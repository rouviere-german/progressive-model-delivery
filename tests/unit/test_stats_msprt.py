from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import integrate, stats

from progressive_model_delivery.stats.msprt import (
    Moments,
    Sided,
    mixture_log_e,
    paired_mean_log_e,
    two_sample_mean_log_e,
)

pytestmark = pytest.mark.unit


def _integrated(
    d: float, v: float, tau: float, bounds: tuple[float, float], weight: float
) -> float:
    def integrand(theta: float) -> float:
        lr = math.exp(theta * d / v - theta * theta / (2 * v))
        return lr * weight * stats.norm.pdf(theta, scale=tau)

    return math.log(integrate.quad(integrand, *bounds, limit=200)[0])


@pytest.mark.parametrize(("d", "v", "tau"), [(0.03, 1e-4, 0.05), (-0.01, 4e-4, 0.02)])
def test_closed_forms_match_numerical_mixtures(d: float, v: float, tau: float) -> None:
    two_sided = float(mixture_log_e(d, v, null_value=0.0, tau=tau, sided=Sided.TWO_SIDED))
    greater = float(mixture_log_e(d, v, null_value=0.0, tau=tau, sided=Sided.GREATER))
    less = float(mixture_log_e(d, v, null_value=0.0, tau=tau, sided=Sided.LESS))
    assert two_sided == pytest.approx(_integrated(d, v, tau, (-1.0, 1.0), 1.0), abs=1e-6)
    assert greater == pytest.approx(_integrated(d, v, tau, (0.0, 1.0), 2.0), abs=1e-6)
    assert less == pytest.approx(_integrated(d, v, tau, (-1.0, 0.0), 2.0), abs=1e-6)


@given(
    values=st.lists(st.floats(-5, 5, allow_nan=False), min_size=2, max_size=80),
    cut=st.integers(0, 80),
)
def test_moments_are_additive_and_match_numpy(values: list[float], cut: int) -> None:
    array = np.array(values)
    cut = min(cut, len(values))
    split = Moments.zeros().add(array[:cut]) + Moments.zeros().add(array[cut:])
    whole = Moments.zeros().add(array)
    assert float(split.count) == len(values)
    assert float(split.mean) == pytest.approx(float(np.mean(array)), abs=1e-9)
    assert float(split.variance) == pytest.approx(float(np.var(array, ddof=1)), abs=1e-7)
    assert float(whole.total) == pytest.approx(float(split.total))


def test_masked_batches_and_small_samples() -> None:
    moments = Moments.zeros((2,)).add(
        np.array([[1.0, 2.0, 9.0], [4.0, 4.0, 4.0]]),
        [
            [True, True, False],
            [True, False, False],
        ],
    )
    assert moments.count.tolist() == [2.0, 1.0]
    assert moments.mean.tolist() == [1.5, 4.0]
    assert moments.variance.tolist() == [0.5, 0.0]
    empty = Moments.of(0, 0, 0)
    assert float(empty.mean) == 0.0
    assert float(empty.variance) == 0.0


def test_burn_in_holds_the_statistic_at_zero() -> None:
    few = Moments.zeros().add(np.array([5.0, 6.0, 7.0]))
    assert float(paired_mean_log_e(few, null_value=0.0, tau=0.1, sided="greater")) == 0.0
    arm = Moments.zeros().add(np.full(300, 1.0))
    assert float(two_sample_mean_log_e(arm, few, null_value=0.0, tau=0.1, sided="greater")) == 0.0


def test_strong_effects_accumulate_evidence_in_the_right_direction() -> None:
    rng = np.random.default_rng(1)
    worse = Moments.zeros().add(rng.normal(0.05, 0.2, size=2_000))
    assert float(paired_mean_log_e(worse, null_value=0.0, tau=0.05, sided="greater")) > 10
    assert float(paired_mean_log_e(worse, null_value=0.0, tau=0.05, sided="less")) < 0
    champion = Moments.zeros().add(rng.normal(0.30, 0.3, size=3_000))
    better = Moments.zeros().add(rng.normal(0.25, 0.3, size=3_000))
    assert (
        float(two_sample_mean_log_e(champion, better, null_value=0.02, tau=0.05, sided="less")) > 10
    )


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="tau"):
        mixture_log_e(0.1, 0.01, null_value=0.0, tau=0.0, sided="greater")
    with pytest.raises(ValueError, match="variance"):
        mixture_log_e(0.1, 0.0, null_value=0.0, tau=0.1, sided="greater")


@pytest.mark.slow
def test_plug_in_variance_msprt_keeps_false_harm_rate_below_alpha_on_skewed_losses() -> None:
    """A/A with log-loss-like (skewed, heavy right tail) data, checked after every batch."""
    rng = np.random.default_rng(11)
    reps, looks, batch, alpha = 2_000, 100, 30, 0.05
    champion = Moments.zeros((reps,))
    challenger = Moments.zeros((reps,))
    paired = Moments.zeros((reps,))
    rejected_two = np.zeros(reps, dtype=bool)
    rejected_paired = np.zeros(reps, dtype=bool)
    for _ in range(looks):
        a = rng.exponential(0.35, size=(reps, batch))
        b = rng.exponential(0.35, size=(reps, batch))
        champion = champion.add(a)
        challenger = challenger.add(b)
        paired = paired.add(b - rng.exponential(0.35, size=(reps, batch)))
        threshold = math.log(1 / alpha)
        rejected_two |= (
            two_sample_mean_log_e(champion, challenger, null_value=0.0, tau=0.05, sided="greater")
            >= threshold
        )
        rejected_paired |= (
            paired_mean_log_e(paired, null_value=0.0, tau=0.05, sided="greater") >= threshold
        )
    slack = 3.3 * math.sqrt(alpha * (1 - alpha) / reps)
    assert rejected_two.mean() <= alpha + slack
    assert rejected_paired.mean() <= alpha + slack
