"""The request feature schema shared by the generator, the contracts, training and serving.

One definition feeds the pandera contract for training and logged frames, the pydantic contract
for single requests, the synthetic generator and the model feature order. Three of the four import
these specs; the pydantic contract cannot (pydantic needs literal annotations), so it hand-copies
the bounds and ``tests/unit/test_serving_units.py`` holds the copy to them.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FEATURES",
    "FEATURE_ORDER",
    "NUMERIC_FEATURES",
    "REGIONS",
    "REGION_SHARES",
    "SUBGROUP_FEATURE",
    "FeatureSpec",
    "region_code",
]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    low: float
    high: float
    integer: bool = False
    description: str = ""


NUMERIC_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("tenure_months", 0, 120, integer=True, description="months as a customer"),
    FeatureSpec("monthly_spend", 0, 500, description="average monthly spend"),
    FeatureSpec("support_tickets", 0, 20, integer=True, description="tickets in 90 days"),
    FeatureSpec("num_products", 1, 6, integer=True, description="active products"),
    FeatureSpec("discount_rate", 0, 0.5, description="current discount"),
    FeatureSpec("usage_trend", -1, 1, description="normalized 30-day usage change"),
)
REGIONS: tuple[str, ...] = ("north", "south", "east", "west")
REGION_SHARES: tuple[float, ...] = (0.40, 0.12, 0.28, 0.20)
SUBGROUP_FEATURE = "region"
FEATURES: tuple[str, ...] = (*(spec.name for spec in NUMERIC_FEATURES), SUBGROUP_FEATURE)
FEATURE_ORDER: tuple[str, ...] = FEATURES
"""Column order used to train the demo models (the region is encoded by :func:`region_code`)."""


def region_code(region: str) -> int:
    """Stable integer code of a region (LightGBM categorical feature)."""
    try:
        return REGIONS.index(region)
    except ValueError:
        raise ValueError(f"unknown region {region!r}") from None
