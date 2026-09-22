from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from progressive_model_delivery.ml.contracts import (
    PREDICTION_LOG_SCHEMA,
    DataContractError,
    feature_frame_schema,
    validate_frame,
)
from progressive_model_delivery.models.synthetic import make_dataset

pytestmark = pytest.mark.unit


def failures_for(frame: pd.DataFrame, *, labeled: bool = True) -> DataContractError:
    with pytest.raises(DataContractError) as info:
        validate_frame(frame, feature_frame_schema(labeled=labeled))
    return info.value


def test_synthetic_datasets_satisfy_the_contract() -> None:
    frame = make_dataset(300, seed=1)
    assert len(validate_frame(frame, feature_frame_schema(labeled=True))) == 300
    features = frame.drop(columns=["label"])
    assert len(validate_frame(features, feature_frame_schema(labeled=False))) == 300


@settings(max_examples=15)
@given(
    n=st.integers(min_value=1, max_value=60), seed=st.integers(0, 10_000), shift=st.floats(-1, 1)
)
def test_generator_always_satisfies_the_contract(n: int, seed: int, shift: float) -> None:
    validate_frame(make_dataset(n, seed=seed, shift=shift), feature_frame_schema(labeled=True))


def test_violations_are_summarized_without_the_offending_values() -> None:
    frame = make_dataset(20, seed=2)
    frame.loc[3, "monthly_spend"] = 123456.789
    frame.loc[4, "region"] = "atlantis"
    error = failures_for(frame)
    assert {"monthly_spend", "region"} <= {str(item["column"]) for item in error.failures}
    for leaked in ("123456", "atlantis"):
        assert leaked not in str(error)
        assert leaked not in repr(error.failures)


def test_unexpected_missing_and_mistyped_columns_are_rejected() -> None:
    frame = make_dataset(5, seed=3)
    frame["email"] = "someone@example.com"
    assert failures_for(frame).failures
    assert failures_for(make_dataset(5, seed=3).drop(columns=["label"])).failures
    mistyped = make_dataset(5, seed=3)
    mistyped["tenure_months"] = mistyped["tenure_months"].astype("float64")
    assert failures_for(mistyped).failures
    bad_label = make_dataset(5, seed=3)
    bad_label.loc[0, "label"] = 2
    assert any(item["column"] == "label" for item in failures_for(bad_label).failures)


def _log_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "request_id": "a" * 32,
        "rollout_id": "b" * 32,
        "ts": 10.0,
        "stage_index": 1,
        "arm": "challenger",
        "weight": 0.05,
        "release_id": "c" * 64,
        "subject_hash": "d" * 32,
        "subgroup": "south",
        "score": 0.3,
        "latency_ms": 0.4,
        "f_tenure_months": 12.0,
        "f_monthly_spend": 80.0,
        "f_support_tickets": 1.0,
        "f_num_products": 2.0,
        "f_discount_rate": 0.1,
        "f_usage_trend": 0.0,
    }
    row.update(overrides)
    return row


def test_prediction_log_contract() -> None:
    good = pd.DataFrame([_log_row(), _log_row(request_id="e" * 32, arm="champion")])
    assert len(validate_frame(good, PREDICTION_LOG_SCHEMA)) == 2
    for bad in (
        _log_row(score=1.5),
        _log_row(arm="treatment"),
        _log_row(subject_hash="cust-0001"),
        _log_row(f_discount_rate=0.9),
    ):
        with pytest.raises(DataContractError):
            validate_frame(pd.DataFrame([bad]), PREDICTION_LOG_SCHEMA)
    duplicated = pd.DataFrame([_log_row(), _log_row()])
    with pytest.raises(DataContractError):
        validate_frame(duplicated, PREDICTION_LOG_SCHEMA)
