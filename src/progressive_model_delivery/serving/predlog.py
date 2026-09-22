"""Prediction and label logs: contract-checked Parquet segments, read back with DuckDB.

Each writer (the router, each shadow worker, the feedback endpoint) owns its own directory and
buffers records in memory; :meth:`SegmentWriter.flush` validates the batch against its pandera
contract and writes an immutable Parquet segment (temporary file, then atomic rename). Readers
never see a partial segment, several processes can write side by side without sharing a DuckDB
file lock, and the controller reads everything with ``read_parquet(<dir>/**/*.parquet)``.

Records buffered but not yet flushed are lost if the process crashes (at most one flush interval,
one second by default); that is an accepted limit of the single-router design.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from progressive_model_delivery.ml.contracts import PREDICTION_LOG_SCHEMA, validate_frame

__all__ = [
    "LABEL_SCHEMA",
    "NO_ROLLOUT",
    "FeedbackOutcome",
    "RequestIndex",
    "SegmentWriter",
]

NO_ROLLOUT = "0" * 32
LABEL_SCHEMA = pa.DataFrameSchema(
    {
        "request_id": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{32}$"), unique=True),
        "label": pa.Column("int64", pa.Check.isin([0, 1])),
        "received_at": pa.Column("float64", pa.Check.ge(0)),
    },
    strict=True,
    coerce=False,
    name="labels",
)


class SegmentWriter:
    """Buffered, thread-safe writer of contract-checked Parquet segments."""

    def __init__(
        self,
        directory: Path,
        *,
        schema: pa.DataFrameSchema = PREDICTION_LOG_SCHEMA,
        prefix: str = "segment",
        max_buffer: int = 5_000,
    ) -> None:
        self.directory = directory
        self.schema = schema
        self.prefix = prefix
        self.max_buffer = max_buffer
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._sequence = 0
        directory.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._buffer.append(dict(record))
            full = len(self._buffer) >= self.max_buffer
        if full:
            self.flush()

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._buffer)

    def flush(self) -> int:
        """Write buffered records as one segment; returns the number of records written."""
        with self._flush_lock:
            with self._lock:
                batch, self._buffer = self._buffer, []
            if not batch:
                return 0
            frame = validate_frame(pd.DataFrame(batch), self.schema)
            self._sequence += 1
            name = f"{self.prefix}-{os.getpid()}-{self._sequence:08d}-{id(self):x}.parquet"
            fd, tmp_name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=self.directory)
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                frame.to_parquet(tmp, index=False)
                os.replace(tmp, self.directory / name)
            finally:
                tmp.unlink(missing_ok=True)
            return len(frame)


class FeedbackOutcome(StrEnum):
    ACCEPTED = "accepted"
    UNKNOWN = "unknown_or_expired"
    DUPLICATE = "duplicate"


@dataclass(slots=True)
class _Served:
    served_at: float
    labeled: bool = False


class RequestIndex:
    """Recently served request ids, for the feedback endpoint's integrity checks.

    A label is accepted only for a request this router served, only within ``ttl_seconds`` of
    serving it, and only once. The index is bounded (oldest entries are evicted first), so a
    flood of requests cannot exhaust memory; evicted ids simply stop accepting labels.
    """

    def __init__(self, *, ttl_seconds: float, capacity: int, clock: Callable[[], float]) -> None:
        if ttl_seconds <= 0 or capacity < 1:
            raise ValueError("ttl_seconds and capacity must be positive")
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self._clock = clock
        self._entries: OrderedDict[str, _Served] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _expire(self, now: float) -> None:
        while self._entries:
            request_id, served = next(iter(self._entries.items()))
            if now - served.served_at <= self.ttl_seconds and len(self._entries) <= self.capacity:
                break
            del self._entries[request_id]

    def remember(self, request_id: str, served_at: float) -> None:
        with self._lock:
            self._entries[request_id] = _Served(served_at)
            self._expire(self._clock())

    def claim(self, request_id: str) -> FeedbackOutcome:
        """Mark a request as labeled if the label is admissible."""
        now = self._clock()
        with self._lock:
            self._expire(now)
            served = self._entries.get(request_id)
            if served is None or now - served.served_at > self.ttl_seconds:
                return FeedbackOutcome.UNKNOWN
            if served.labeled:
                return FeedbackOutcome.DUPLICATE
            served.labeled = True
            return FeedbackOutcome.ACCEPTED
