"""Mixture sequential probability ratio tests (mSPRT) for mean differences such as log-loss.

The statistic follows Robbins' normal mixture as used for A/B testing by Johari, Koomen, Pekelis
and Walsh (2017, "Peeking at A/B tests"). For an estimate ``d`` of a mean difference with
sampling variance ``V`` and ``x = d - theta0``, mixing the Gaussian likelihood ratio over
``theta - theta0 ~ N(0, tau^2)`` gives

    log Lambda = 0.5 * log(V / (V + tau^2)) + tau^2 x^2 / (2 V (V + tau^2)).

The one-sided version mixes over a half-normal (only ``theta > theta0``, or only
``theta < theta0``). Every component likelihood ratio is then a supermartingale under the whole
composite null, and the mixture evaluates in closed form to

    log Lambda_+ = log 2 + 0.5 * log(V / (V + tau^2)) + tau^2 x^2 / (2 V (V + tau^2))
                   + log Phi(x tau / sqrt(V (V + tau^2))).

``V`` uses running sample variances, which makes the guarantee asymptotic (Waudby-Smith, Arbour,
Sinha, Kennedy and Ramdas 2024, "Time-uniform central limit theory and asymptotic confidence
sequences"); ``min_n`` is a burn-in per arm before which the statistic is held at 0. Log-losses
are right-skewed, so early sample means and variances are biased together; with a 200-sample
burn-in the simulated coverage in ``tests/unit/test_stats_confseq.py`` stays at or above 95%.

Designs
-------
* Shadow stage (paired): the stream ``loss_challenger_i - loss_champion_i``, ``V = s^2 / n``.
* Canary stage with a fixed weight: two independent arms, ``d = mean_b - mean_a`` and
  ``V = s_b^2 / n_b + s_a^2 / n_a``. The weight is constant within a stage, so the Welch
  estimate is the natural one; per-stage e-values are multiplied across stages.

All statistics are functions of per-arm sums (count, sum, sum of squares), which the controller
reads with one SQL aggregation and the study accumulates with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy import special

__all__ = [
    "Moments",
    "Sided",
    "mixture_log_e",
    "paired_mean_log_e",
    "two_sample_mean_log_e",
]

FloatArray = npt.NDArray[np.float64]
_LOG2 = float(np.log(2.0))


class Sided(StrEnum):
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


@dataclass(frozen=True, slots=True)
class Moments:
    """Vectorized additive sufficient statistics of a stream: count, sum, sum of squares."""

    count: FloatArray
    total: FloatArray
    total_sq: FloatArray

    @classmethod
    def zeros(cls, shape: tuple[int, ...] = ()) -> Moments:
        return cls(np.zeros(shape), np.zeros(shape), np.zeros(shape))

    @classmethod
    def of(cls, count: npt.ArrayLike, total: npt.ArrayLike, total_sq: npt.ArrayLike) -> Moments:
        c, s, q = np.broadcast_arrays(
            np.asarray(count, dtype=np.float64),
            np.asarray(total, dtype=np.float64),
            np.asarray(total_sq, dtype=np.float64),
        )
        return cls(c.copy(), s.copy(), q.copy())

    def add(self, values: npt.ArrayLike, valid: npt.ArrayLike | None = None) -> Moments:
        """Fold a batch along the last axis (``valid`` masks records that do not count)."""
        data = np.asarray(values, dtype=np.float64)
        mask = np.ones(data.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        filled = np.where(mask, data, 0.0)
        return Moments(
            self.count + mask.sum(axis=-1),
            self.total + filled.sum(axis=-1),
            self.total_sq + (filled * filled).sum(axis=-1),
        )

    def __add__(self, other: Moments) -> Moments:
        return Moments(
            self.count + other.count, self.total + other.total, self.total_sq + other.total_sq
        )

    @property
    def mean(self) -> FloatArray:
        result: FloatArray = np.where(self.count > 0, self.total / np.maximum(self.count, 1.0), 0.0)
        return result

    @property
    def variance(self) -> FloatArray:
        """Unbiased sample variance (0 with fewer than two observations)."""
        n = self.count
        centred = self.total_sq - self.total * self.total / np.maximum(n, 1.0)
        result: FloatArray = np.where(n > 1, np.maximum(centred, 0.0) / np.maximum(n - 1, 1.0), 0.0)
        return result


def mixture_log_e(
    estimate: npt.ArrayLike,
    estimate_variance: npt.ArrayLike,
    *,
    null_value: float,
    tau: float,
    sided: Sided | str,
) -> FloatArray:
    """Log mSPRT statistic for an estimate with the given sampling variance."""
    side = Sided(sided)
    if tau <= 0:
        raise ValueError("tau must be positive")
    d, v = np.broadcast_arrays(
        np.asarray(estimate, dtype=np.float64), np.asarray(estimate_variance, dtype=np.float64)
    )
    if np.any(v <= 0):
        raise ValueError("estimate_variance must be positive")
    t2 = tau * tau
    x = d - null_value
    base = 0.5 * np.log(v / (v + t2)) + t2 * x * x / (2.0 * v * (v + t2))
    if side is Sided.TWO_SIDED:
        result: FloatArray = base
        return result
    z = x * tau / np.sqrt(v * (v + t2))
    signed = z if side is Sided.GREATER else -z
    one_sided: FloatArray = _LOG2 + base + special.log_ndtr(signed)
    return one_sided


def paired_mean_log_e(
    differences: Moments,
    *,
    null_value: float,
    tau: float,
    sided: Sided | str,
    min_n: int = 200,
    variance_floor: float = 1e-6,
) -> FloatArray:
    """mSPRT on a stream of paired differences (challenger minus champion)."""
    n = differences.count
    v = np.maximum(differences.variance, variance_floor) / np.maximum(n, 1.0)
    value = mixture_log_e(differences.mean, v, null_value=null_value, tau=tau, sided=sided)
    result: FloatArray = np.where(n >= max(min_n, 2), value, 0.0)
    return result


def two_sample_mean_log_e(
    champion: Moments,
    challenger: Moments,
    *,
    null_value: float,
    tau: float,
    sided: Sided | str,
    min_n: int = 200,
    variance_floor: float = 1e-6,
) -> FloatArray:
    """mSPRT for ``mean_challenger - mean_champion`` from two independent arms (Welch variance)."""
    na, nb = champion.count, challenger.count
    v = np.maximum(champion.variance, variance_floor) / np.maximum(na, 1.0) + np.maximum(
        challenger.variance, variance_floor
    ) / np.maximum(nb, 1.0)
    value = mixture_log_e(
        challenger.mean - champion.mean, v, null_value=null_value, tau=tau, sided=sided
    )
    ready = (na >= max(min_n, 2)) & (nb >= max(min_n, 2))
    result: FloatArray = np.where(ready, value, 0.0)
    return result
