from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import integrate, optimize, special

from progressive_model_delivery.stats.evalues import (
    Alternative,
    BetaPrior,
    beta_mixture_log_e,
    discordant_null,
    ratio_null,
    two_sample_log_e,
)

pytestmark = pytest.mark.unit


def _integrated_log_e(k: int, j: int, p0: float, strength: float, greater: bool) -> float:
    """Oracle: integrate the truncated-beta mixture of likelihood ratios numerically."""
    a, b = strength * p0, strength * (1 - p0)
    lo, hi = (p0, 1.0) if greater else (0.0, p0)

    def density(p: float) -> float:
        return float(np.exp((a - 1) * np.log(p) + (b - 1) * np.log1p(-p) - special.betaln(a, b)))

    def ratio(p: float) -> float:
        return float(
            np.exp(special.xlogy(k, p / p0) + special.xlogy(j, (1 - p) / (1 - p0))) * density(p)
        )

    mass = integrate.quad(density, lo, hi, limit=200)[0]
    value = integrate.quad(ratio, lo, hi, limit=200, points=[k / max(k + j, 1)])[0]
    return math.log(value / mass)


@pytest.mark.parametrize(
    ("k", "j", "p0", "greater"),
    [(12, 5, 0.5, True), (3, 30, 0.25, True), (0, 0, 0.1, True), (4, 40, 0.3, False)],
)
def test_beta_mixture_matches_numerical_integration(
    k: int, j: int, p0: float, greater: bool
) -> None:
    alternative = Alternative.GREATER if greater else Alternative.LESS
    closed = float(beta_mixture_log_e(k, j, null_p=p0, alternative=alternative, strength=20.0))
    assert closed == pytest.approx(_integrated_log_e(k, j, p0, 20.0, greater), abs=1e-6)


def test_no_data_means_no_evidence() -> None:
    for alternative in Alternative:
        assert float(beta_mixture_log_e(0, 0, null_p=0.3, alternative=alternative)) == (
            pytest.approx(0.0, abs=1e-12)
        )


@given(
    k=st.integers(0, 400),
    j=st.integers(0, 400),
    p0=st.floats(0.01, 0.99),
    shrink=st.floats(0.0, 1.0),
)
def test_mixture_is_a_supermartingale_under_every_null_rate(
    k: int, j: int, p0: float, shrink: float
) -> None:
    """Validity invariant: E[E_{t+1} / E_t] <= 1 for every pi' <= pi0 and every history."""
    true_p = p0 * shrink
    now = beta_mixture_log_e(k, j, null_p=p0, alternative="greater")
    up = beta_mixture_log_e(k + 1, j, null_p=p0, alternative="greater")
    down = beta_mixture_log_e(k, j + 1, null_p=p0, alternative="greater")
    if float(now) <= -600:  # both branches underflow; the ratio is meaningless there
        return
    expected = true_p * math.exp(float(up - now)) + (1 - true_p) * math.exp(float(down - now))
    assert expected <= 1.0 + 1e-7


@given(k=st.integers(0, 300), j=st.integers(0, 300), p0=st.floats(0.02, 0.98))
def test_more_challenger_events_never_reduce_harm_evidence(k: int, j: int, p0: float) -> None:
    now = float(beta_mixture_log_e(k, j, null_p=p0, alternative="greater"))
    more = float(beta_mixture_log_e(k + 1, j, null_p=p0, alternative="greater"))
    fewer_champion = float(beta_mixture_log_e(k, j + 1, null_p=p0, alternative="greater"))
    assert more >= now - 1e-9
    assert fewer_champion <= now + 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("null_p", [0.5, 0.05, 0.25])
def test_continuous_monitoring_keeps_false_rejections_below_alpha(null_p: float) -> None:
    rng = np.random.default_rng(7)
    reps, looks, batch, alpha = 3_000, 150, 40, 0.05
    k = np.zeros(reps)
    j = np.zeros(reps)
    rejected = np.zeros(reps, dtype=bool)
    for _ in range(looks):
        challenger = rng.binomial(batch, null_p, size=reps)
        k += challenger
        j += batch - challenger
        rejected |= beta_mixture_log_e(k, j, null_p=null_p, alternative="greater") >= math.log(
            1 / alpha
        )
    # Upper edge of a 99.9% binomial band around alpha.
    assert rejected.mean() <= alpha + 3.3 * math.sqrt(alpha * (1 - alpha) / reps)


def test_ratio_and_discordant_nulls() -> None:
    assert float(ratio_null(0.25, 1.0)) == pytest.approx(0.25)
    assert float(ratio_null(0.5, 1.5)) == pytest.approx(0.6)
    assert discordant_null() == pytest.approx(0.5)
    assert discordant_null(3.0) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="weight"):
        ratio_null(1.0, 1.0)
    with pytest.raises(ValueError, match="ratio"):
        ratio_null(0.5, 0.0)
    with pytest.raises(ValueError, match="odds_ratio"):
        discordant_null(0.0)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        beta_mixture_log_e(-1, 2, null_p=0.5, alternative="greater")
    with pytest.raises(ValueError, match="null_p"):
        beta_mixture_log_e(1, 2, null_p=1.0, alternative="greater")
    with pytest.raises(ValueError, match="strength"):
        beta_mixture_log_e(1, 2, null_p=0.5, alternative="greater", strength=0)
    with pytest.raises(ValueError, match="Alternative"):
        beta_mixture_log_e(1, 2, null_p=0.5, alternative="sideways")
    with pytest.raises(ValueError, match="prior"):
        BetaPrior(0.0, 1.0)


def _oracle_two_sample(
    *, sa: int, na: int, sb: int, nb: int, margin: float, greater: bool
) -> float:
    """Independent oracle: grid search inside the null plus scipy's optimum on its boundary."""
    grid = np.linspace(1e-6, 1 - 1e-6, 1501)
    pa, pb = np.meshgrid(grid, grid, indexing="ij")
    feasible = (pb - pa <= margin) if greater else (pb - pa >= margin)

    def loglik(p_a: npt.ArrayLike, p_b: npt.ArrayLike) -> npt.NDArray[np.float64]:
        a = np.asarray(p_a, dtype=np.float64)
        b = np.asarray(p_b, dtype=np.float64)
        return np.asarray(
            special.xlogy(sa, a)
            + special.xlog1py(na - sa, -a)
            + special.xlogy(sb, b)
            + special.xlog1py(nb - sb, -b),
            dtype=np.float64,
        )

    inside = float(np.where(feasible, loglik(pa, pb), -np.inf).max())
    boundary = optimize.minimize_scalar(
        lambda x: -float(loglik(x, x + margin)),
        bounds=(max(0.0, -margin) + 1e-9, min(1.0, 1.0 - margin) - 1e-9),
        method="bounded",
        options={"xatol": 1e-12},
    )
    mixture = (
        special.betaln(0.5 + sa, 0.5 + na - sa)
        + special.betaln(0.5 + sb, 0.5 + nb - sb)
        - 2 * special.betaln(0.5, 0.5)
    )
    return float(mixture - max(inside, -float(boundary.fun)))


@pytest.mark.parametrize(
    ("sa", "na", "sb", "nb", "margin", "greater"),
    [
        (12, 100, 25, 90, 0.0, True),
        (12, 100, 25, 90, 0.05, True),
        (12, 100, 25, 90, 0.0, False),
        (30, 200, 10, 180, 0.02, False),
    ],
)
def test_unconditional_e_value_matches_an_optimization_oracle(
    *, sa: int, na: int, sb: int, nb: int, margin: float, greater: bool
) -> None:
    alternative = "greater" if greater else "less"
    closed = float(two_sample_log_e(sa, na, sb, nb, margin=margin, alternative=alternative))
    assert closed == pytest.approx(
        _oracle_two_sample(sa=sa, na=na, sb=sb, nb=nb, margin=margin, greater=greater),
        abs=2e-3,
    )


@given(
    sa=st.integers(0, 50),
    extra_a=st.integers(0, 200),
    sb=st.integers(0, 50),
    extra_b=st.integers(0, 200),
)
def test_unconditional_e_value_is_at_most_one_on_the_null_side(
    sa: int, extra_a: int, sb: int, extra_b: int
) -> None:
    na, nb = sa + extra_a, sb + extra_b
    log_e = float(two_sample_log_e(sa, na, sb, nb, margin=0.0, alternative="greater"))
    if na and nb and sb / nb <= sa / na:
        assert log_e <= 1e-9


def test_unconditional_handles_empty_arms_and_validates() -> None:
    assert float(two_sample_log_e(0, 0, 0, 0, margin=0.0, alternative="less")) <= 1e-9
    assert float(two_sample_log_e(0, 0, 3, 10, margin=0.1, alternative="less")) <= 1e-9
    with pytest.raises(ValueError, match="exceed"):
        two_sample_log_e(5, 3, 0, 1, margin=0.0, alternative="greater")
    with pytest.raises(ValueError, match="margin"):
        two_sample_log_e(1, 3, 0, 1, margin=1.0, alternative="greater")


def test_log_space_tails_match_scipy_and_survive_underflow() -> None:
    from scipy import special as sp

    from progressive_model_delivery.stats.evalues import _log_beta_tail

    a = np.array([20.0, 5.0, 300.0, 47.66, 2.0])
    b = np.array([400.0, 900.0, 40.0, 1_284.34, 3.0])
    x = np.array([0.5, 0.2, 0.5, 0.953, 0.4])
    for upper in (True, False):
        ours = _log_beta_tail(a, b, x, upper=upper)
        with np.errstate(divide="ignore"):
            reference = np.log(sp.betaincc(a, b, x) if upper else sp.betainc(a, b, x))
        representable = np.isfinite(reference) & (reference > -600)
        np.testing.assert_allclose(ours[representable], reference[representable], rtol=1e-9)
        assert np.all(np.isfinite(ours))
    # Far beyond double-precision underflow, and still monotone in the shape parameter.
    deep = _log_beta_tail(
        np.array([47.66, 47.66]), np.array([3_000.0, 3_001.0]), np.array([0.953, 0.953]), upper=True
    )
    assert deep[0] < -5_000
    assert deep[1] < deep[0]
