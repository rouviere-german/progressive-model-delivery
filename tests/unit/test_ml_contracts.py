from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from progressive_model_delivery.ml.contracts import (
    DataContractError,
    make_synthetic_events,
    validate_events,
)

pytestmark = pytest.mark.unit


def failures_for(frame: pd.DataFrame) -> DataContractError:
    with pytest.raises(DataContractError) as info:
        validate_events(frame)
    return info.value


def test_synthetic_events_satisfy_the_contract() -> None:
    assert len(validate_events(make_synthetic_events(200, seed=1))) == 200


@settings(max_examples=20)
@given(n=st.integers(min_value=1, max_value=40), seed=st.integers(min_value=0, max_value=10_000))
def test_generator_always_satisfies_the_contract(n: int, seed: int) -> None:
    validate_events(make_synthetic_events(n, seed=seed))


def test_violations_are_summarized_without_the_offending_values() -> None:
    frame = make_synthetic_events(20, seed=2)
    frame.loc[3, "amount"] = -123456.789
    frame.loc[4, "user_id"] = ""
    error = failures_for(frame)
    assert {"amount", "user_id"} <= {str(item["column"]) for item in error.failures}
    assert "123456" not in str(error)
    assert "123456" not in repr(error.failures)


def test_unexpected_columns_are_rejected() -> None:
    frame = make_synthetic_events(5, seed=3)
    frame["email"] = "someone@example.com"
    assert failures_for(frame).failures


def test_missing_columns_are_rejected() -> None:
    assert failures_for(make_synthetic_events(5, seed=3).drop(columns=["label"])).failures


def test_timezone_naive_timestamps_are_rejected() -> None:
    frame = make_synthetic_events(5, seed=4)
    frame["event_time"] = frame["event_time"].dt.tz_localize(None)
    assert failures_for(frame).failures


def test_duplicate_event_ids_are_rejected() -> None:
    frame = make_synthetic_events(5, seed=5)
    frame.loc[1, "event_id"] = frame.loc[0, "event_id"]
    assert any(item["column"] == "event_id" for item in failures_for(frame).failures)


def test_positive_labels_require_a_positive_amount() -> None:
    frame = make_synthetic_events(5, seed=6)
    frame.loc[0, "label"] = 1
    frame.loc[0, "amount"] = 0.0
    assert failures_for(frame).failures


def test_generator_rejects_empty_requests() -> None:
    with pytest.raises(ValueError, match="n must be"):
        make_synthetic_events(0, seed=1)
