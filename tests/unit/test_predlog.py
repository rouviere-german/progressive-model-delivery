from __future__ import annotations

import threading
from pathlib import Path

import duckdb
import pytest

from progressive_model_delivery.ml.contracts import DataContractError
from progressive_model_delivery.serving.predlog import (
    LABEL_SCHEMA,
    FeedbackOutcome,
    RequestIndex,
    SegmentWriter,
)

pytestmark = pytest.mark.unit


def _record(i: int, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "request_id": f"{i:032x}",
        "rollout_id": "b" * 32,
        "ts": float(i),
        "stage_index": 0,
        "arm": "champion",
        "weight": 0.0,
        "release_id": "c" * 64,
        "subject_hash": "d" * 32,
        "subgroup": "north",
        "score": 0.2,
        "latency_ms": 0.3,
        "f_tenure_months": 12.0,
        "f_monthly_spend": 80.0,
        "f_support_tickets": 1.0,
        "f_num_products": 2.0,
        "f_discount_rate": 0.1,
        "f_usage_trend": 0.0,
    }
    record.update(overrides)
    return record


def test_segments_are_contract_checked_and_readable_with_duckdb(tmp_path: Path) -> None:
    writer = SegmentWriter(tmp_path / "router", max_buffer=10_000)

    def produce(k: int) -> None:
        for i in range(50):
            writer.append(_record(k * 100 + i))

    threads = [threading.Thread(target=produce, args=(k,)) for k in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert writer.pending == 200
    assert writer.flush() == 200
    assert writer.flush() == 0
    files = [str(p) for p in (tmp_path / "router").glob("*.parquet")]
    assert len(files) == 1
    with duckdb.connect() as con:
        count, ids = con.execute(
            "SELECT count(*), count(DISTINCT request_id) FROM read_parquet($f)", {"f": files}
        ).fetchone() or (0, 0)
    assert (count, ids) == (200, 200)


def test_a_bad_batch_is_refused_without_writing_a_partial_segment(tmp_path: Path) -> None:
    writer = SegmentWriter(tmp_path / "router")
    writer.append(_record(1))
    writer.append(_record(2, score=1.7))
    with pytest.raises(DataContractError):
        writer.flush()
    assert list((tmp_path / "router").iterdir()) == []


def test_full_buffers_flush_automatically(tmp_path: Path) -> None:
    writer = SegmentWriter(tmp_path / "labels", schema=LABEL_SCHEMA, max_buffer=3)
    for i in range(7):
        writer.append({"request_id": f"{i:032x}", "label": i % 2, "received_at": float(i)})
    assert len(list((tmp_path / "labels").glob("*.parquet"))) == 2
    assert writer.pending == 1


def test_request_index_accepts_each_known_request_once_within_the_ttl() -> None:
    now = [100.0]
    index = RequestIndex(ttl_seconds=60, capacity=3, clock=lambda: now[0])
    for i in range(3):
        index.remember(f"{i:032x}", now[0])
    assert index.claim(f"{0:032x}") is FeedbackOutcome.ACCEPTED
    assert index.claim(f"{0:032x}") is FeedbackOutcome.DUPLICATE
    assert index.claim("f" * 32) is FeedbackOutcome.UNKNOWN
    index.remember(f"{3:032x}", now[0])  # evicts the oldest entry (capacity 3)
    assert len(index) == 3
    assert index.claim(f"{0:032x}") is FeedbackOutcome.UNKNOWN
    now[0] += 61
    assert index.claim(f"{1:032x}") is FeedbackOutcome.UNKNOWN  # expired
    assert len(index) == 0
    with pytest.raises(ValueError, match="positive"):
        RequestIndex(ttl_seconds=0, capacity=1, clock=lambda: 0.0)
