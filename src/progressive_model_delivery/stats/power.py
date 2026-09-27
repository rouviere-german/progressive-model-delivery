"""Power calculator: fixed-horizon sample sizes and simulated sequential detection.

The fixed-horizon formulas are the textbook one-sided two-proportion z-test with unequal
allocation (for example Chow, Shao and Wang, "Sample Size Calculations in Clinical Research",
section 4.2). The sequential side is estimated by simulating the same conditional beta-mixture
e-value the controller uses (and, for comparison, the unconditional universal-inference one),
which is the only honest way to plan an anytime-valid test: its stopping time has no simple
closed form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from progressive_model_delivery.stats.evalues import (
    Alternative,
    beta_mixture_log_e,
    ratio_null,
    two_sample_log_e,
)

__all__ = [
    "SequentialPlan",
    "fixed_horizon_power",
    "fixed_horizon_sample_size",
    "simulate_sequential_detection",
    "two_proportion_z",
]

FloatArray = npt.NDArray[np.float64]


def _check_rate(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")


def fixed_horizon_sample_size(
    p_control: float,
    delta: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> int:
    """Total requests a one-sided fixed-horizon z-test needs to detect ``+delta`` harm.

    ``allocation`` is the share of traffic on the challenger.
    """
    _check_rate(p_control, "p_control")
    _check_rate(p_control + delta, "p_control + delta")
    _check_rate(alpha, "alpha")
    _check_rate(power, "power")
    _check_rate(allocation, "allocation")
    if delta <= 0:
        raise ValueError("delta must be positive")
    w = allocation
    p1, p2 = p_control, p_control + delta
    p_bar = (1 - w) * p1 + w * p2
    z_a = stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    numerator = z_a * math.sqrt(p_bar * (1 - p_bar) * (1 / w + 1 / (1 - w))) + z_b * math.sqrt(
        p1 * (1 - p1) / (1 - w) + p2 * (1 - p2) / w
    )
    return math.ceil((numerator / delta) ** 2)


def two_proportion_z(
    successes_a: npt.ArrayLike,
    n_a: npt.ArrayLike,
    successes_b: npt.ArrayLike,
    n_b: npt.ArrayLike,
) -> FloatArray:
    """Pooled two-proportion z statistic for ``p_b - p_a`` (0 where undefined)."""
    sa = np.asarray(successes_a, dtype=np.float64)
    na = np.asarray(n_a, dtype=np.float64)
    sb = np.asarray(successes_b, dtype=np.float64)
    nb = np.asarray(n_b, dtype=np.float64)
    safe_na, safe_nb = np.maximum(na, 1.0), np.maximum(nb, 1.0)
    pooled = (sa + sb) / np.maximum(na + nb, 1.0)
    se = np.sqrt(pooled * (1 - pooled) * (1 / safe_na + 1 / safe_nb))
    z = np.where(se > 0, (sb / safe_nb - sa / safe_na) / np.where(se > 0, se, 1.0), 0.0)
    result: FloatArray = np.where((na > 0) & (nb > 0), z, 0.0)
    return result


def fixed_horizon_power(
    p_control: float, p_challenger: float, n_total: int, *, alpha: float, allocation: float
) -> float:
    """Normal-approximation power of the one-sided z-test at a fixed total sample size."""
    _check_rate(p_control, "p_control")
    _check_rate(p_challenger, "p_challenger")
    w = allocation
    na, nb = n_total * (1 - w), n_total * w
    p_bar = (1 - w) * p_control + w * p_challenger
    se0 = math.sqrt(p_bar * (1 - p_bar) * (1 / na + 1 / nb))
    se1 = math.sqrt(p_control * (1 - p_control) / na + p_challenger * (1 - p_challenger) / nb)
    z_a = stats.norm.ppf(1 - alpha)
    return float(stats.norm.sf((z_a * se0 - (p_challenger - p_control)) / se1))


@dataclass(frozen=True, slots=True)
class SequentialPlan:
    """Simulated behaviour of the anytime-valid harm test for one effect size."""

    p_control: float
    p_challenger: float
    allocation: float
    alpha: float
    max_requests: int
    detection_probability: float
    median_requests_to_detection: float | None
    unconditional_detection_probability: float
    unconditional_median_requests: float | None
    fixed_horizon_requests_80: int | None


def simulate_sequential_detection(
    p_control: float,
    p_challenger: float,
    *,
    allocation: float = 0.5,
    alpha: float = 0.05,
    max_requests: int = 20_000,
    look_every: int = 200,
    reps: int = 400,
    seed: int = 0,
) -> SequentialPlan:
    """Detection probability and median requests to detection of the harm tests."""
    _check_rate(p_control, "p_control")
    _check_rate(p_challenger, "p_challenger")
    _check_rate(allocation, "allocation")
    rng = np.random.default_rng(seed)
    looks = max_requests // look_every
    sa = np.zeros(reps)
    na = np.zeros(reps)
    sb = np.zeros(reps)
    nb = np.zeros(reps)
    stopped = np.full(reps, np.nan)
    stopped_unconditional = np.full(reps, np.nan)
    threshold = math.log(1 / alpha)
    null_p = ratio_null(allocation, 1.0)
    for look in range(1, looks + 1):
        to_b = rng.binomial(look_every, allocation, size=reps)
        to_a = look_every - to_b
        sb += rng.binomial(to_b, p_challenger)
        sa += rng.binomial(to_a, p_control)
        nb += to_b
        na += to_a
        conditional = beta_mixture_log_e(sb, sa, null_p=null_p, alternative=Alternative.GREATER)
        unconditional = two_sample_log_e(sa, na, sb, nb, margin=0.0, alternative="greater")
        stopped[np.isnan(stopped) & (conditional >= threshold)] = look * look_every
        stopped_unconditional[np.isnan(stopped_unconditional) & (unconditional >= threshold)] = (
            look * look_every
        )
    detected = ~np.isnan(stopped)
    detected_unconditional = ~np.isnan(stopped_unconditional)
    fixed: int | None = None
    if p_challenger > p_control:
        fixed = fixed_horizon_sample_size(
            p_control, p_challenger - p_control, alpha=alpha, power=0.8, allocation=allocation
        )
    return SequentialPlan(
        p_control=p_control,
        p_challenger=p_challenger,
        allocation=allocation,
        alpha=alpha,
        max_requests=looks * look_every,
        detection_probability=float(detected.mean()),
        median_requests_to_detection=float(np.median(stopped[detected]))
        if detected.any()
        else None,
        unconditional_detection_probability=float(detected_unconditional.mean()),
        unconditional_median_requests=float(
            np.median(stopped_unconditional[detected_unconditional])
        )
        if detected_unconditional.any()
        else None,
        fixed_horizon_requests_80=fixed,
    )
