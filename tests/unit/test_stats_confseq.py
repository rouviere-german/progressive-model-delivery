from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from progressive_model_delivery.stats.confseq import beta_binomial_cs, normal_mixture_cs
from progressive_model_delivery.stats.msprt import mixture_log_e

pytestmark = pytest.mark.unit


@given(n=st.integers(1, 5_000), frac=st.floats(0, 1), alpha=st.floats(0.01, 0.2))
def test_beta_binomial_cs_contains_the_estimate(n: int, frac: float, alpha: float) -> None:
    s = round(frac * n)
    lower, upper = beta_binomial_cs(s, n, alpha=alpha)
    assert 0.0 <= float(lower) <= s / n <= float(upper) <= 1.0


def test_beta_binomial_cs_edges_and_narrowing() -> None:
    lower, upper = beta_binomial_cs(0, 0, alpha=0.05)
    assert (float(lower), float(upper)) == (0.0, 1.0)
    lower, _ = beta_binomial_cs(0, 50, alpha=0.05)
    assert float(lower) == 0.0
    _, upper = beta_binomial_cs(50, 50, alpha=0.05)
    assert float(upper) == 1.0
    widths = [
        float(np.subtract(*beta_binomial_cs(n // 10, n, alpha=0.05)[::-1]))
        for n in (100, 1_000, 10_000)
    ]
    assert widths[0] > widths[1] > widths[2]
    with pytest.raises(ValueError, match="alpha"):
        beta_binomial_cs(1, 2, alpha=1.5)
    with pytest.raises(ValueError, match="successes"):
        beta_binomial_cs(3, 2, alpha=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("rate", [0.1, 0.5])
def test_beta_binomial_cs_has_time_uniform_coverage(rate: float) -> None:
    """Property from the spec: >= 95% of paths never leave the sequence, checked at every step."""
    rng = np.random.default_rng(3)
    reps, horizon, alpha = 600, 1_500, 0.05
    successes = np.cumsum(rng.random((reps, horizon)) < rate, axis=1)
    lower, upper = beta_binomial_cs(successes, np.arange(1, horizon + 1), alpha=alpha)
    miss = ((lower > rate) | (upper < rate)).any(axis=1).mean()
    assert miss <= alpha + 3.3 * math.sqrt(alpha * (1 - alpha) / reps)


@pytest.mark.slow
def test_normal_mixture_cs_has_time_uniform_coverage_after_burn_in() -> None:
    """Skewed, log-loss-like data: asymptotic coverage holds after the 200-sample burn-in."""
    rng = np.random.default_rng(5)
    reps, horizon, alpha, burn_in = 600, 2_000, 0.05, 200
    data = rng.exponential(0.4, size=(reps, horizon)) - 0.4
    n = np.arange(1, horizon + 1)
    csum = np.cumsum(data, axis=1)
    csq = np.cumsum(data * data, axis=1)
    mean = csum / n
    var = np.maximum(csq - csum * csum / n, 1e-12) / np.maximum(n - 1, 1)
    lower, upper = normal_mixture_cs(mean, var / n, tau=0.05, alpha=alpha)
    miss = ((lower[:, burn_in:] > 0) | (upper[:, burn_in:] < 0)).any(axis=1).mean()
    assert miss <= alpha + 3.3 * math.sqrt(alpha * (1 - alpha) / reps)


@given(d=st.floats(-0.2, 0.2), v=st.floats(1e-6, 1e-2), alpha=st.floats(0.01, 0.2))
def test_normal_mixture_cs_is_the_inverted_two_sided_msprt(
    d: float, v: float, alpha: float
) -> None:
    lower, upper = normal_mixture_cs(d, v, tau=0.05, alpha=alpha)
    for edge in (float(lower), float(upper)):
        boundary = float(mixture_log_e(d, v, null_value=edge, tau=0.05, sided="two-sided"))
        assert boundary == pytest.approx(math.log(1 / alpha), abs=1e-6)


def test_normal_mixture_cs_validates() -> None:
    with pytest.raises(ValueError, match="alpha"):
        normal_mixture_cs(0.0, 1.0, tau=0.1, alpha=0.0)
    with pytest.raises(ValueError, match="tau"):
        normal_mixture_cs(0.0, 1.0, tau=-1.0, alpha=0.1)
    with pytest.raises(ValueError, match="variance"):
        normal_mixture_cs(0.0, 0.0, tau=0.1, alpha=0.1)
