"""pandera data contracts with PII-safe failure reporting.

Two contracts guard the frames this project moves around:

* :func:`feature_frame_schema` for training, holdout and simulated traffic frames (optionally
  with the label). It is built from :mod:`progressive_model_delivery.models.features`, the same
  definition that feeds the router's pydantic request model, so both contracts agree.
* :data:`PREDICTION_LOG_SCHEMA` for prediction-log segments before they are written to Parquet:
  request ids, arms, scores in [0, 1], latencies, assignment weights and model versions.

Failure summaries report column, check and count only. The failing values themselves are never
copied into the exception, because they are often exactly the sensitive data being validated.
Types are checked strictly (no coercion), and unexpected columns are rejected.
"""

from __future__ import annotations

from functools import cache

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from progressive_model_delivery.models.features import NUMERIC_FEATURES, REGIONS

__all__ = [
    "ARMS",
    "PREDICTION_LOG_SCHEMA",
    "DataContractError",
    "feature_frame_schema",
    "validate_frame",
]

ARMS = ("champion", "challenger", "shadow")


class DataContractError(ValueError):
    """A frame violated its contract. ``failures`` holds (column, check, count) summaries."""

    def __init__(self, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.failures = failures


def _feature_columns() -> dict[str, pa.Column]:
    columns: dict[str, pa.Column] = {}
    for spec in NUMERIC_FEATURES:
        columns[spec.name] = pa.Column(
            "int64" if spec.integer else "float64",
            pa.Check.in_range(spec.low, spec.high),
            nullable=False,
        )
    columns["region"] = pa.Column(str, pa.Check.isin(list(REGIONS)), nullable=False)
    return columns


@cache
def feature_frame_schema(*, labeled: bool) -> pa.DataFrameSchema:
    """Contract for feature frames; ``labeled`` adds a binary ``label`` column."""
    columns = _feature_columns()
    if labeled:
        columns["label"] = pa.Column("int64", pa.Check.isin([0, 1]), nullable=False)
    return pa.DataFrameSchema(
        columns, strict=True, coerce=False, name="labeled_features" if labeled else "features"
    )


PREDICTION_LOG_SCHEMA = pa.DataFrameSchema(
    {
        "request_id": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{32}$"), unique=True),
        "rollout_id": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{32}$")),
        "ts": pa.Column("float64", pa.Check.ge(0)),
        "stage_index": pa.Column("int64", pa.Check.in_range(0, 16)),
        "arm": pa.Column(str, pa.Check.isin(list(ARMS))),
        "weight": pa.Column("float64", pa.Check.in_range(0.0, 1.0)),
        "release_id": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{64}$")),
        "subject_hash": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{32}$")),
        "subgroup": pa.Column(str, pa.Check.isin(list(REGIONS))),
        "score": pa.Column("float64", pa.Check.in_range(0.0, 1.0)),
        "latency_ms": pa.Column("float64", pa.Check.ge(0)),
        **{
            f"f_{spec.name}": pa.Column("float64", pa.Check.in_range(spec.low, spec.high))
            for spec in NUMERIC_FEATURES
        },
    },
    strict=True,
    coerce=False,
    name="prediction_log",
)


def validate_frame(frame: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate lazily (all failures at once) and raise a summarized DataContractError."""
    try:
        return schema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases[["column", "check"]].astype(str)
        counts = cases.value_counts().sort_index()
        failures: list[dict[str, str | int]] = [
            {"column": column, "check": check, "count": int(count)}
            for (column, check), count in counts.items()
        ]
        raise DataContractError(failures) from None
