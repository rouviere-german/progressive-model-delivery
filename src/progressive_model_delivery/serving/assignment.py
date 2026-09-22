"""Subject hashing and consistent-hash traffic assignment.

* ``subject_hash = HMAC-SHA256(secret, subject_id)`` (first 128 bits, hex). Raw subject ids never
  reach logs or the assignment function; without the secret the hash cannot be recomputed from a
  guessed id.
* ``bucket = SHA-256(rollout_salt || ":" || subject_hash) mod 10_000``. A subject is on the
  challenger when ``bucket < weight_bp``. Because the threshold only moves up (1% -> 5% -> 25% ->
  50%), every subject on the challenger stays there when the weight increases (sticky), and a new
  rollout salt re-draws the partition so the same customers are not always the guinea pigs.
"""

from __future__ import annotations

import hashlib
import hmac
from enum import StrEnum

__all__ = ["BUCKETS", "Arm", "assign", "bucket_of", "subject_hash"]

BUCKETS = 10_000


class Arm(StrEnum):
    CHAMPION = "champion"
    CHALLENGER = "challenger"


def subject_hash(secret: bytes, subject_id: str) -> str:
    if len(secret) < 16:
        raise ValueError("the subject hash secret must be at least 16 bytes")
    if not subject_id:
        raise ValueError("subject_id must not be empty")
    return hmac.new(secret, subject_id.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def bucket_of(hashed_subject: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{hashed_subject}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS


def assign(hashed_subject: str, salt: str, weight_bp: int) -> Arm:
    """Arm for a subject at a canary weight in basis points (0 = champion only)."""
    if not 0 <= weight_bp <= BUCKETS:
        raise ValueError("weight_bp must be within [0, 10000]")
    return Arm.CHALLENGER if bucket_of(hashed_subject, salt) < weight_bp else Arm.CHAMPION
