"""The request contracts must agree with the one feature definition.

``models/features.py`` is the single definition of the feature schema: it feeds the pandera contract
for training frames, the synthetic generator and the model's feature order. The pydantic contract in
``serving/schemas.py`` cannot import those bounds -- pydantic needs literal annotations to build
the validator -- so it hand-copies them, and a hand copy drifts. A widened bound in
``features.py`` would otherwise leave the API rejecting traffic the model was trained on, with
nothing failing.

These tests are the check the ``serving/schemas.py`` and ``models/features.py`` docstrings claim.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest

from progressive_model_delivery.models.features import (
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    REGIONS,
    SUBGROUP_FEATURE,
    FeatureSpec,
)
from progressive_model_delivery.serving.schemas import Region, RequestFeatures

pytestmark = pytest.mark.unit


def _constraint(field: Any, name: str) -> float | None:
    """The ``ge``/``le`` an ``annotated_types`` marker on a pydantic field carries."""
    for marker in field.metadata:
        value = getattr(marker, name, None)
        if value is not None:
            return float(value)
    return None


@pytest.mark.parametrize("spec", NUMERIC_FEATURES, ids=[s.name for s in NUMERIC_FEATURES])
def test_the_pydantic_bounds_are_the_feature_definition_bounds(spec: FeatureSpec) -> None:
    field = RequestFeatures.model_fields[spec.name]
    assert _constraint(field, "ge") == spec.low, f"{spec.name}: lower bound drifted"
    assert _constraint(field, "le") == spec.high, f"{spec.name}: upper bound drifted"
    assert field.annotation is (int if spec.integer else float), f"{spec.name}: type drifted"
    assert field.is_required(), f"{spec.name}: a served request must carry every feature"


def test_the_request_contract_has_exactly_the_defined_features() -> None:
    assert tuple(RequestFeatures.model_fields) == FEATURE_ORDER


def test_the_region_literal_is_the_subgroup_definition() -> None:
    assert get_args(Region) == REGIONS
    assert get_args(RequestFeatures.model_fields[SUBGROUP_FEATURE].annotation) == REGIONS
