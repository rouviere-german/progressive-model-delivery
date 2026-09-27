"""Anytime-valid e-values for Bernoulli guardrails (error rate, subgroup error, latency > SLO).

Every function is vectorized: counts may be scalars or NumPy arrays of any broadcastable shape,
so the controller evaluates one rollout and the statistical study evaluates thousands with the
same code.

Conditional one-parameter tests (what the controller uses)
----------------------------------------------------------
Both rollout designs reduce a two-model comparison to a single Bernoulli parameter ``pi`` whose
null value is known, so no nuisance parameter has to be learned:

* **Shadow (paired).** Both models score the same request. Among *discordant* requests (exactly
  one model made the bad event), ``pi = P(it was the challenger)`` ``= p01 / (p01 + p10)``. The
  challenger is no worse (``p_challenger <= p_champion``) exactly when ``pi <= 1/2``. This is
  the sequential form of McNemar's test.
* **Canary (two arms).** A request goes to the challenger with known probability ``w`` (the
  stage weight). Given that a bad event happened, ``pi = P(it was on the challenger arm)``
  ``= w r / (w r + 1 - w)`` with ``r = p_challenger / p_champion``, independently of the
  champion's own rate. ``r <= r0`` is equivalent to ``pi <= pi0(w, r0)``.

For H0 ``pi <= pi0`` the e-value mixes the likelihood ratio ``(pi / pi0)^k ((1 - pi) /
(1 - pi0))^j`` over a Beta prior truncated to ``(pi0, 1]``. Each component is a supermartingale
under every null ``pi' <= pi0`` (its conditional expectation is linear in ``pi'`` and equals 1 at
``pi0``), so the mixture is an e-process for the whole composite null and Ville's inequality gives
``P(exists t: E_t >= 1 / alpha) <= alpha`` under continuous monitoring. The mixture has the closed
form

    E = B(a + k, b + j) * T(a + k, b + j) / (B(a, b) * T(a, b) * pi0^k * (1 - pi0)^j)

where ``T`` is the upper (or, for H0 ``pi >= pi0``, lower) regularized incomplete beta tail at
``pi0`` (Robbins 1970; Howard et al. 2021). The prior is ``Beta(c pi0, c (1 - pi0))``, centred on
the null with strength ``c``: the analogue of the mixing scale ``tau`` of a normal-mixture SPRT.

E-values from different stages are computed on disjoint requests, so their product is again an
e-value (optional continuation); the controller multiplies per-stage e-values.

Unconditional comparator
------------------------
:func:`two_sample_log_e` is the universal-inference e-value ``M_t / sup_{H0} L_t`` with a
beta-binomial mixture per arm (Wasserman, Ramdas and Balakrishnan 2020). It is valid for
difference margins but has to learn the champion's rate as well, which costs about
``0.5 * log(n)`` nats; the power study reports how much slower it detects regressions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy import special

__all__ = [
    "JEFFREYS",
    "LOG_E_FLOOR",
    "Alternative",
    "BetaPrior",
    "beta_mixture_log_e",
    "discordant_null",
    "log_beta_binomial",
    "ratio_null",
    "two_sample_log_e",
]

FloatArray = npt.NDArray[np.float64]
DerivativeFn = Callable[[FloatArray], FloatArray]
_BISECTION_STEPS = 64
_CF_STEPS = 400
LOG_E_FLOOR = -700.0
"""Log e-values are clipped here: an e-value of exp(-700) is zero for every decision."""


class Alternative(StrEnum):
    """Direction of the evidence an e-value collects."""

    GREATER = "greater"
    LESS = "less"


@dataclass(frozen=True, slots=True)
class BetaPrior:
    """Beta(a, b) mixing distribution for one Bernoulli arm (default: Jeffreys / KT)."""

    a: float = 0.5
    b: float = 0.5

    def __post_init__(self) -> None:
        if not (self.a > 0 and self.b > 0):
            raise ValueError("beta prior parameters must be positive")


JEFFREYS = BetaPrior()


def _counts(value: npt.ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return array


def ratio_null(weight: npt.ArrayLike, ratio: float) -> FloatArray:
    """``pi0`` for a canary stage: P(event is on the challenger arm) when ``r = ratio``."""
    w = np.asarray(weight, dtype=np.float64)
    if np.any((w <= 0) | (w >= 1)):
        raise ValueError("canary weight must lie strictly between 0 and 1")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    result: FloatArray = w * ratio / (w * ratio + 1.0 - w)
    return result


def discordant_null(odds_ratio: float = 1.0) -> float:
    """``pi0`` for the shadow stage: P(challenger erred | discordant) at a discordant odds ratio."""
    if odds_ratio <= 0:
        raise ValueError("odds_ratio must be positive")
    return odds_ratio / (1.0 + odds_ratio)


def _beta_continued_fraction(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    """Continued fraction of the incomplete beta function (modified Lentz, vectorized).

    Converges quickly for ``x < (a + 1) / (a + b + 2)`` (Press et al., Numerical Recipes 6.4).
    """
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
    h = d.copy()
    for m in range(1, _CF_STEPS + 1):
        m2 = 2.0 * m
        for numerator in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
            c = 1.0 + numerator / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            h = h * d * c
    return h


def _log_beta_tail(a: FloatArray, b: FloatArray, x: FloatArray, *, upper: bool) -> FloatArray:
    """``log`` of the lower (or upper) regularized incomplete beta tail at ``x``.

    SciPy's value is used where it is comfortably representable; tails below 1e-200 (where
    subnormal floats lose precision) are recomputed in log space from the continued fraction,
    which is exactly the region where it converges fastest.
    """
    direct = special.betaincc(a, b, x) if upper else special.betainc(a, b, x)
    with np.errstate(divide="ignore"):
        log_direct = np.log(direct)
    small = direct < 1e-200
    if not np.any(small):
        return np.asarray(log_direct, dtype=np.float64)
    # Upper tail of Beta(a, b) at x is the lower tail of Beta(b, a) at 1 - x.
    aa, bb, xx = (b[small], a[small], 1.0 - x[small]) if upper else (a[small], b[small], x[small])
    log_front = (
        special.gammaln(aa + bb)
        - special.gammaln(aa)
        - special.gammaln(bb)
        + aa * np.log(xx)
        + bb * np.log1p(-xx)
        - np.log(aa)
    )
    result = np.asarray(log_direct, dtype=np.float64).copy()
    result[small] = log_front + np.log(_beta_continued_fraction(aa, bb, xx))
    return result


def beta_mixture_log_e(
    successes: npt.ArrayLike,
    failures: npt.ArrayLike,
    *,
    null_p: npt.ArrayLike,
    alternative: Alternative | str,
    strength: float = 50.0,
) -> FloatArray:
    """Log e-value of a truncated beta-binomial mixture for a one-parameter Bernoulli null.

    ``successes`` count events on the challenger side (discordant pairs where the challenger
    erred, or events on the challenger arm), ``failures`` count the champion side.
    ``GREATER`` tests H0 ``pi <= null_p``; ``LESS`` tests H0 ``pi >= null_p``.
    """
    alt = Alternative(alternative)
    if strength <= 0:
        raise ValueError("strength must be positive")
    k = _counts(successes, "successes")
    j = _counts(failures, "failures")
    p0 = np.asarray(null_p, dtype=np.float64)
    if np.any((p0 <= 0) | (p0 >= 1)):
        raise ValueError("null_p must lie strictly between 0 and 1")
    k, j, p0 = np.broadcast_arrays(k, j, p0)
    a = strength * p0
    b = strength * (1.0 - p0)
    upper = alt is Alternative.GREATER
    value = (
        special.betaln(a + k, b + j)
        - special.betaln(a, b)
        + _log_beta_tail(a + k, b + j, p0, upper=upper)
        - _log_beta_tail(a, b, p0, upper=upper)
        - special.xlogy(k, p0)
        - special.xlog1py(j, -p0)
    )
    result: FloatArray = np.maximum(np.nan_to_num(value, nan=LOG_E_FLOOR), LOG_E_FLOOR)
    return result


def log_beta_binomial(
    successes: npt.ArrayLike, failures: npt.ArrayLike, prior: BetaPrior = JEFFREYS
) -> FloatArray:
    """Log marginal probability of an ordered Bernoulli sequence under a Beta prior."""
    s = _counts(successes, "successes")
    f = _counts(failures, "failures")
    result: FloatArray = special.betaln(prior.a + s, prior.b + f) - special.betaln(prior.a, prior.b)
    return result


def _bisect_decreasing(
    derivative: DerivativeFn, lo: FloatArray, hi: FloatArray, steps: int = _BISECTION_STEPS
) -> FloatArray:
    """Argmax of a concave function on [lo, hi] given its (decreasing) derivative."""
    lo = lo.copy()
    hi = hi.copy()
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        rising = derivative(mid) > 0.0
        lo = np.where(rising, mid, lo)
        hi = np.where(rising, hi, mid)
    result: FloatArray = 0.5 * (lo + hi)
    return result


def _two_sample_slope(
    sa: FloatArray, fa: FloatArray, sb: FloatArray, fb: FloatArray, margin: float
) -> DerivativeFn:
    """d/dpa of the two-sample log-likelihood on the boundary ``pb = pa + margin``."""

    def slope(point: FloatArray) -> FloatArray:
        pb = point + margin
        with np.errstate(divide="ignore", invalid="ignore"):
            value: FloatArray = sa / point - fa / (1.0 - point) + sb / pb - fb / (1.0 - pb)
        return np.nan_to_num(value, nan=0.0)

    return slope


def _binomial_loglik(s: FloatArray, f: FloatArray, p: FloatArray) -> FloatArray:
    result: FloatArray = np.asarray(special.xlogy(s, p) + special.xlog1py(f, -p), np.float64)
    return result


def _unconstrained_loglik(s: FloatArray, f: FloatArray) -> FloatArray:
    n = s + f
    p = np.where(n > 0, s / np.where(n > 0, n, 1.0), 0.0)
    return np.where(n > 0, _binomial_loglik(s, f, p), 0.0)


def two_sample_log_e(
    successes_champion: npt.ArrayLike,
    n_champion: npt.ArrayLike,
    successes_challenger: npt.ArrayLike,
    n_challenger: npt.ArrayLike,
    *,
    margin: float,
    alternative: Alternative | str,
    prior: BetaPrior = JEFFREYS,
) -> FloatArray:
    """Universal-inference log e-value for ``delta = p_challenger - p_champion`` vs ``margin``.

    ``GREATER`` tests H0 ``delta <= margin``; ``LESS`` tests H0 ``delta >= margin``.
    """
    alt = Alternative(alternative)
    if not -1.0 < margin < 1.0:
        raise ValueError("margin must lie strictly between -1 and 1")
    m = float(margin)
    sa = _counts(successes_champion, "successes_champion")
    na = _counts(n_champion, "n_champion")
    sb = _counts(successes_challenger, "successes_challenger")
    nb = _counts(n_challenger, "n_challenger")
    if np.any(sa > na) or np.any(sb > nb):
        raise ValueError("successes cannot exceed the number of observations")
    sa, na, sb, nb = np.broadcast_arrays(sa, na, sb, nb)
    fa, fb = na - sa, nb - sb

    log_mixture = log_beta_binomial(sa, fa, prior) + log_beta_binomial(sb, fb, prior)
    log_unconstrained = _unconstrained_loglik(sa, fa) + _unconstrained_loglik(sb, fb)

    pa_hat = np.where(na > 0, sa / np.where(na > 0, na, 1.0), np.nan)
    pb_hat = np.where(nb > 0, sb / np.where(nb > 0, nb, 1.0), np.nan)
    # An arm without data is free: pick the value that best satisfies the constraint.
    if alt is Alternative.GREATER:  # H0: pb - pa <= m
        feasible = (
            np.where(np.isnan(pb_hat), 0.0, pb_hat) - np.where(np.isnan(pa_hat), 1.0, pa_hat) <= m
        )
    else:  # H0: pb - pa >= m
        feasible = (
            np.where(np.isnan(pb_hat), 1.0, pb_hat) - np.where(np.isnan(pa_hat), 0.0, pa_hat) >= m
        )

    lo = np.full(sa.shape, max(0.0, -m) + 1e-15)
    hi = np.full(sa.shape, min(1.0, 1.0 - m) - 1e-15)
    pa_star = _bisect_decreasing(_two_sample_slope(sa, fa, sb, fb, m), lo, hi)
    log_boundary = _binomial_loglik(sa, fa, pa_star) + _binomial_loglik(sb, fb, pa_star + m)
    log_null = np.where(feasible, log_unconstrained, np.minimum(log_boundary, log_unconstrained))
    result: FloatArray = np.maximum(log_mixture - log_null, LOG_E_FLOOR)
    return result
