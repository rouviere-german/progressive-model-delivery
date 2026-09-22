"""pandera data contract example with PII-safe failure reporting.

Failure summaries report column, check and count only. The failing values themselves are never
copied into the exception, because they are often exactly the sensitive data being validated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.typing.pandas import Series

__all__ = ["DataContractError", "EventSchema", "make_synthetic_events", "validate_events"]


class DataContractError(ValueError):
    """A frame violated its contract. ``failures`` holds (column, check, count) summaries."""

    def __init__(self, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.failures = failures


class EventSchema(pa.DataFrameModel):
    """Example contract for a synthetic transaction-event table.

    Types are checked strictly (no coercion): ``event_time`` must already be timezone-aware UTC at
    nanosecond resolution, so naive or local timestamps are rejected instead of silently converted.
    """

    event_id: Series[str] = pa.Field(unique=True, str_matches=r"^evt_[0-9a-f]{8}$")
    user_id: Series[str] = pa.Field(str_length={"min_value": 1, "max_value": 64})
    amount: Series[float] = pa.Field(ge=0.0, le=1_000_000.0)
    event_time: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "UTC"})
    label: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        name = "events"

    @pa.dataframe_check
    @classmethod
    def positive_labels_have_amount(cls, frame: pd.DataFrame) -> Series[bool]:
        """A positive label requires a strictly positive amount."""
        return Series[bool]((frame["label"] == 0) | (frame["amount"] > 0))


def validate_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate lazily (all failures at once) and raise a summarized DataContractError."""
    try:
        return EventSchema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases[["column", "check"]].astype(str)
        counts = cases.value_counts().sort_index()
        failures: list[dict[str, str | int]] = [
            {"column": column, "check": check, "count": int(count)}
            for (column, check), count in counts.items()
        ]
        raise DataContractError(failures) from None


def make_synthetic_events(n: int, *, seed: int) -> pd.DataFrame:
    """Deterministic synthetic events that satisfy :class:`EventSchema`."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, 30 * 86_400, size=n)
    return pd.DataFrame(
        {
            "event_id": [f"evt_{i:08x}" for i in range(n)],
            "user_id": [f"user_{u}" for u in rng.integers(0, n // 10 + 1, size=n)],
            "amount": np.maximum(np.round(rng.gamma(2.0, 20.0, size=n), 2), 0.01),
            "event_time": (
                pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(offsets, unit="s")
            ).astype("datetime64[ns, UTC]"),
            "label": (rng.random(n) < 0.1).astype(np.int64),
        }
    )
