"""Time-uniform confidence sequences.

* :func:`beta_binomial_cs` inverts the beta-binomial mixture martingale for a single Bernoulli
  rate: ``C_t = {p : M_t / L_t(p) < 1 / alpha}`` where ``M_t`` is the beta-binomial marginal
  likelihood. By Ville's inequality ``P(exists t: p not in C_t) <= alpha`` exactly (Robbins 1970;
  Howard, Ramdas, McAuliffe and Sekhon 2021, "Time-uniform, nonparametric, nonasymptotic
  confidence sequences"). The set is an interval because the log-likelihood is concave in ``p``.
* :func:`normal_mixture_cs` inverts the two-sided normal-mixture mSPRT, giving the asymptotic
  confidence sequence ``d +- sqrt(2 V (V + tau^2) / tau^2 * log(sqrt((V + tau^2) / V) / alpha))``
  for an estimate ``d`` with sampling variance ``V``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import special

from progressive_model_delivery.stats.evalues import JEFFREYS, BetaPrior, log_beta_binomial

__all__ = ["beta_binomial_cs", "normal_mixture_cs"]

FloatArray = npt.NDArray[np.float64]
_STEPS = 60


def normal_mixture_cs(
    estimate: npt.ArrayLike,
    estimate_variance: npt.ArrayLike,
    *,
    tau: float,
    alpha: float,
) -> tuple[FloatArray, FloatArray]:
    """Two-sided asymptotic confidence sequence around an estimate (inverted mSPRT)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if tau <= 0:
        raise ValueError("tau must be positive")
    d = np.asarray(estimate, dtype=np.float64)
    v = np.asarray(estimate_variance, dtype=np.float64)
    if np.any(v <= 0):
        raise ValueError("estimate_variance must be positive")
    t2 = tau * tau
    radius = np.sqrt(2.0 * v * (v + t2) / t2 * np.log(np.sqrt((v + t2) / v) / alpha))
    lower: FloatArray = d - radius
    upper: FloatArray = d + radius
    return lower, upper


def beta_binomial_cs(
    successes: npt.ArrayLike,
    n: npt.ArrayLike,
    *,
    alpha: float,
    prior: BetaPrior = JEFFREYS,
) -> tuple[FloatArray, FloatArray]:
    """Exact time-uniform confidence sequence for a Bernoulli rate after ``n`` observations."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    s = np.asarray(successes, dtype=np.float64)
    total = np.asarray(n, dtype=np.float64)
    s, total = np.broadcast_arrays(s, total)
    if np.any(s < 0) or np.any(s > total):
        raise ValueError("successes must lie in [0, n]")
    f = total - s
    threshold = log_beta_binomial(s, f, prior) + np.log(alpha)
    p_hat = np.where(total > 0, s / np.maximum(total, 1.0), 0.5)

    def loglik(p: FloatArray) -> FloatArray:
        result: FloatArray = np.asarray(special.xlogy(s, p) + special.xlog1py(f, -p), np.float64)
        return result

    # Lower end: loglik increases on [0, p_hat]; find the first p with loglik >= threshold.
    lo = np.zeros(s.shape)
    hi = p_hat.copy()
    inside_at_zero = loglik(lo) >= threshold
    for _ in range(_STEPS):
        mid = 0.5 * (lo + hi)
        above = loglik(mid) >= threshold
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    lower: FloatArray = np.where(inside_at_zero | (total == 0), 0.0, hi)

    lo = p_hat.copy()
    hi = np.ones(s.shape)
    inside_at_one = loglik(hi) >= threshold
    for _ in range(_STEPS):
        mid = 0.5 * (lo + hi)
        above = loglik(mid) >= threshold
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    upper: FloatArray = np.where(inside_at_one | (total == 0), 1.0, lo)
    return lower, upper
