"""Synthetic customer population with a known label model and delayed, lossy labels.

Everything is seeded, so datasets, trained models and simulated traffic are reproducible. The
label model has a region-specific interaction (in ``south`` the effect of ``usage_trend``
reverses), which is what lets a challenger regress in one subgroup only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from progressive_model_delivery.models.features import REGION_SHARES, REGIONS

__all__ = [
    "LabelDelay",
    "label_probability",
    "make_dataset",
    "sample_features",
    "sample_label_delays",
    "sample_labels",
    "subject_ids",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
_REGION_OFFSET = {"north": 0.0, "south": 0.7, "east": -0.5, "west": 0.2}


@dataclass(frozen=True, slots=True)
class LabelDelay:
    """Label arrival model: exponential delays, slower for positives, some labels never arrive."""

    mean_negative_seconds: float = 240.0
    mean_positive_seconds: float = 600.0
    loss_rate: float = 0.03

    def __post_init__(self) -> None:
        if self.mean_negative_seconds <= 0 or self.mean_positive_seconds <= 0:
            raise ValueError("mean delays must be positive")
        if not 0.0 <= self.loss_rate < 1.0:
            raise ValueError("loss_rate must be in [0, 1)")


def sample_features(
    n: int,
    rng: np.random.Generator,
    *,
    region_shares: tuple[float, ...] = REGION_SHARES,
    shift: float = 0.0,
) -> pd.DataFrame:
    """Draw ``n`` feature rows. ``shift`` moves discount and usage (covariate drift knob)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    shares = np.asarray(region_shares, dtype=np.float64)
    if shares.shape != (len(REGIONS),) or not np.isclose(shares.sum(), 1.0):
        raise ValueError("region_shares must have one entry per region and sum to 1")
    return pd.DataFrame(
        {
            "tenure_months": rng.integers(0, 121, size=n).astype(np.int64),
            "monthly_spend": np.round(np.clip(rng.gamma(2.0, 40.0, size=n), 0, 500), 2),
            "support_tickets": np.clip(rng.poisson(2.0, size=n), 0, 20).astype(np.int64),
            "num_products": (1 + rng.binomial(5, 0.3, size=n)).astype(np.int64),
            "discount_rate": np.round(
                np.clip(rng.beta(2, 8, size=n) * 0.5 + shift * 0.2, 0, 0.5), 4
            ),
            "usage_trend": np.round(np.clip(rng.normal(shift, 0.4, size=n), -1, 1), 4),
            "region": pd.Categorical(
                rng.choice(np.array(REGIONS), size=n, p=shares), categories=list(REGIONS)
            ).astype(str),
        }
    )


def label_probability(frame: pd.DataFrame, *, south_interaction: bool = True) -> FloatArray:
    """True P(label = 1 | features) of the synthetic population.

    ``south_interaction=False`` gives the stale behaviour (before the reversal in ``south``),
    which the ``subgroup_worse`` challenger is trained on for that region.
    """
    region = frame["region"].to_numpy()
    usage = frame["usage_trend"].to_numpy(dtype=np.float64)
    logit = (
        -0.9
        - 0.06 * frame["tenure_months"].to_numpy(dtype=np.float64)
        + 0.60 * frame["support_tickets"].to_numpy(dtype=np.float64)
        - 0.70 * (frame["num_products"].to_numpy(dtype=np.float64) - 1.0)
        + 6.0 * frame["discount_rate"].to_numpy(dtype=np.float64)
        - 3.0 * usage
        + 0.006 * frame["monthly_spend"].to_numpy(dtype=np.float64)
        + np.array([_REGION_OFFSET[str(r)] for r in region])
        + np.where((region == "south") & south_interaction, 6.0 * usage, 0.0)
    )
    result: FloatArray = 1.0 / (1.0 + np.exp(-logit))
    return result


def sample_labels(frame: pd.DataFrame, rng: np.random.Generator) -> IntArray:
    result: IntArray = (rng.random(len(frame)) < label_probability(frame)).astype(np.int64)
    return result


def sample_label_delays(
    labels: npt.ArrayLike, rng: np.random.Generator, delay: LabelDelay | None = None
) -> FloatArray:
    """Seconds until each label arrives; ``inf`` for labels that never arrive."""
    delay = delay or LabelDelay()
    y = np.asarray(labels, dtype=np.int64)
    means = np.where(y == 1, delay.mean_positive_seconds, delay.mean_negative_seconds)
    delays = rng.exponential(1.0, size=y.shape) * means
    lost = rng.random(y.shape) < delay.loss_rate
    result: FloatArray = np.where(lost, np.inf, delays)
    return result


def subject_ids(n: int, rng: np.random.Generator, *, population: int = 2_000_000) -> list[str]:
    """Customer identifiers drawn from a large population (repeat visits are rare)."""
    return [f"cust-{value:07d}" for value in rng.integers(0, population, size=n)]


def make_dataset(n: int, *, seed: int, shift: float = 0.0) -> pd.DataFrame:
    """Features plus sampled labels, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    frame = sample_features(n, rng, shift=shift)
    frame["label"] = sample_labels(frame, rng)
    return frame
