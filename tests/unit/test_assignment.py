from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from progressive_model_delivery.serving.assignment import (
    BUCKETS,
    Arm,
    assign,
    bucket_of,
    subject_hash,
)

pytestmark = pytest.mark.unit

SECRET = b"k" * 32
WEIGHTS = [100, 500, 2_500, 5_000]


@given(
    subject=st.text(min_size=1, max_size=40),
    salt=st.text(alphabet="0123456789abcdef", min_size=32, max_size=32),
    low=st.sampled_from(WEIGHTS),
    high=st.sampled_from(WEIGHTS),
)
def test_assignment_is_sticky_when_the_canary_weight_increases(
    subject: str, salt: str, low: int, high: int
) -> None:
    hashed = subject_hash(SECRET, subject)
    if low <= high and assign(hashed, salt, low) is Arm.CHALLENGER:
        assert assign(hashed, salt, high) is Arm.CHALLENGER
    assert assign(hashed, salt, 0) is Arm.CHAMPION
    assert assign(hashed, salt, BUCKETS) is Arm.CHALLENGER
    assert assign(hashed, salt, low) == assign(hashed, salt, low)  # deterministic


@pytest.mark.parametrize("weight_bp", WEIGHTS)
def test_split_stays_within_binomial_bounds(weight_bp: int) -> None:
    n = 20_000
    salt = "ab" * 16
    hits = sum(
        assign(subject_hash(SECRET, f"cust-{i}"), salt, weight_bp) is Arm.CHALLENGER
        for i in range(n)
    )
    p = weight_bp / BUCKETS
    # 99.99% two-sided binomial band.
    assert abs(hits - n * p) <= 3.9 * math.sqrt(n * p * (1 - p))


def test_a_new_salt_redraws_the_partition() -> None:
    subjects = [subject_hash(SECRET, f"cust-{i}") for i in range(8_000)]
    first = {s for s in subjects if assign(s, "11" * 16, 5_000) is Arm.CHALLENGER}
    second = {s for s in subjects if assign(s, "22" * 16, 5_000) is Arm.CHALLENGER}
    overlap = len(first & second) / len(subjects)
    assert overlap == pytest.approx(0.25, abs=0.03)


def test_subject_hash_is_keyed_and_validated() -> None:
    a = subject_hash(SECRET, "cust-1")
    assert len(a) == 32
    assert a == subject_hash(SECRET, "cust-1")
    assert a != subject_hash(b"z" * 32, "cust-1")
    assert "cust-1" not in a
    with pytest.raises(ValueError, match="16 bytes"):
        subject_hash(b"short", "cust-1")
    with pytest.raises(ValueError, match="empty"):
        subject_hash(SECRET, "")
    with pytest.raises(ValueError, match="weight_bp"):
        assign(a, "salt", 10_001)
    assert 0 <= bucket_of(a, "salt") < BUCKETS
