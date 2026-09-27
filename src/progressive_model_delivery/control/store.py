"""SQLite control plane: champion, release lifecycle, rollouts, approvals, lease and audit chain.

Every state change happens inside one ``BEGIN IMMEDIATE`` transaction together with the audit
entry that justifies it, so a crash either leaves both or neither. Rollout rows carry a revision
number and are updated with compare-and-swap, and controller writes also check the fencing token
of the lease they hold, so a controller that lost its lease (paused, partitioned, restarted)
cannot overwrite a newer decision.

The audit table is a hash chain: ``hash = sha256(canonical_json(entry without hash))`` and each
entry names its predecessor's hash. :meth:`ControlStore.verify_audit` recomputes the chain and
reports the first broken entry.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from progressive_model_delivery.control.spec import RolloutSpec, StageKind
from progressive_model_delivery.core.canonical import canonical_json, sha256_hex
from progressive_model_delivery.registry.lifecycle import (
    LifecycleState,
    from_record,
    to_record,
)

__all__ = [
    "GENESIS_HASH",
    "AuditEntry",
    "AuditVerification",
    "ControlStore",
    "Lease",
    "LeaseLostError",
    "RolloutRecord",
    "RolloutStatus",
    "StageWindow",
    "StaleWriteError",
]

GENESIS_HASH = "0" * 64
RolloutStatus = Literal["active", "awaiting_approval", "promoted", "rolled_back", "aborted"]
OPEN_STATUSES: tuple[RolloutStatus, ...] = ("active", "awaiting_approval")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS champion (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    release_id TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS lifecycle (
    release_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    weight_bp INTEGER NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id TEXT PRIMARY KEY,
    record TEXT NOT NULL,
    status TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_rollout ON rollouts ((1))
    WHERE status IN ('active', 'awaiting_approval');
CREATE TABLE IF NOT EXISTS approvals (
    request_id TEXT PRIMARY KEY,
    rollout_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    record TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lease (
    name TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    token INTEGER NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    rollout_id TEXT,
    data TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
"""


class StaleWriteError(RuntimeError):
    """A compare-and-swap update lost against a concurrent writer."""


class LeaseLostError(RuntimeError):
    """The controller's lease expired or was taken over; its writes are refused."""


class StageWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    kind: StageKind
    weight_bp: int = Field(ge=0, le=10_000)
    started_at: float
    ended_at: float | None = None


class RolloutRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rollout_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    champion: str = Field(pattern=r"^[0-9a-f]{64}$")
    challenger: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec: RolloutSpec
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RolloutStatus
    stage_index: int = Field(ge=0)
    stages: tuple[StageWindow, ...]
    proposed_by: str
    created_at: float
    salt: str = Field(pattern=r"^[0-9a-f]{32}$")
    decided_at: float | None = None
    decision_reason: str | None = None
    revision: int = Field(ge=0)

    @property
    def current(self) -> StageWindow:
        return self.stages[-1]

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES


@dataclass(frozen=True, slots=True)
class Lease:
    name: str
    holder: str
    token: int
    expires_at: float


class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=0)
    ts: str
    actor: str
    action: str
    rollout_id: str | None
    data: dict[str, JsonValue]
    prev_hash: str
    hash: str


@dataclass(frozen=True, slots=True)
class AuditVerification:
    ok: bool
    entries: int
    head: str
    error: str | None = None
    seq: int | None = None


def _entry_hash(fields: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json({k: v for k, v in fields.items() if k != "hash"}))


class ControlStore:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)

    # --- connections -------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One write transaction (``BEGIN IMMEDIATE``): commit on success, roll back on error."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    # --- champion and lifecycle --------------------------------------------------------------

    def champion(self, conn: sqlite3.Connection | None = None) -> str | None:
        query = "SELECT release_id FROM champion WHERE id = 1"
        if conn is not None:
            row = conn.execute(query).fetchone()
        else:
            with self._read() as reader:
                row = reader.execute(query).fetchone()
        return None if row is None else str(row[0])

    def set_champion(self, conn: sqlite3.Connection, release_id: str) -> None:
        conn.execute(
            "INSERT INTO champion (id, release_id, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET release_id = excluded.release_id, "
            "updated_at = excluded.updated_at",
            (release_id, self.clock()),
        )

    def lifecycle(self, release_id: str) -> LifecycleState | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT stage, weight_bp FROM lifecycle WHERE release_id = ?", (release_id,)
            ).fetchone()
        return None if row is None else from_record(str(row[0]), int(row[1]))

    def lifecycle_in(self, conn: sqlite3.Connection, release_id: str) -> LifecycleState | None:
        row = conn.execute(
            "SELECT stage, weight_bp FROM lifecycle WHERE release_id = ?", (release_id,)
        ).fetchone()
        return None if row is None else from_record(str(row[0]), int(row[1]))

    def set_lifecycle(
        self, conn: sqlite3.Connection, release_id: str, state: LifecycleState
    ) -> None:
        stage, weight = to_record(state)
        conn.execute(
            "INSERT INTO lifecycle (release_id, stage, weight_bp, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(release_id) DO UPDATE SET stage = excluded.stage, "
            "weight_bp = excluded.weight_bp, updated_at = excluded.updated_at",
            (release_id, stage, weight, self.clock()),
        )

    # --- rollouts ----------------------------------------------------------------------------

    @staticmethod
    def new_rollout_id() -> str:
        return uuid.uuid4().hex

    def insert_rollout(self, conn: sqlite3.Connection, record: RolloutRecord) -> None:
        try:
            conn.execute(
                "INSERT INTO rollouts (rollout_id, record, status, revision, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record.rollout_id,
                    record.model_dump_json(),
                    record.status,
                    record.revision,
                    record.created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise StaleWriteError("another rollout is already open") from exc

    def update_rollout(
        self, conn: sqlite3.Connection, record: RolloutRecord, *, expected_revision: int
    ) -> RolloutRecord:
        """Compare-and-swap on the revision; returns the stored record (revision + 1)."""
        updated = record.model_copy(update={"revision": expected_revision + 1})
        cursor = conn.execute(
            "UPDATE rollouts SET record = ?, status = ?, revision = ? "
            "WHERE rollout_id = ? AND revision = ?",
            (
                updated.model_dump_json(),
                updated.status,
                updated.revision,
                updated.rollout_id,
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleWriteError(f"rollout {record.rollout_id} changed concurrently")
        return updated

    def rollout(self, rollout_id: str, conn: sqlite3.Connection | None = None) -> RolloutRecord:
        query = "SELECT record FROM rollouts WHERE rollout_id = ?"
        if conn is not None:
            row = conn.execute(query, (rollout_id,)).fetchone()
        else:
            with self._read() as reader:
                row = reader.execute(query, (rollout_id,)).fetchone()
        if row is None:
            raise KeyError(rollout_id)
        return RolloutRecord.model_validate_json(str(row[0]))

    def open_rollout(self, conn: sqlite3.Connection | None = None) -> RolloutRecord | None:
        query = "SELECT record FROM rollouts WHERE status IN ('active', 'awaiting_approval')"
        if conn is not None:
            row = conn.execute(query).fetchone()
        else:
            with self._read() as reader:
                row = reader.execute(query).fetchone()
        return None if row is None else RolloutRecord.model_validate_json(str(row[0]))

    def rollouts(self) -> list[RolloutRecord]:
        with self._read() as conn:
            rows = conn.execute("SELECT record FROM rollouts ORDER BY created_at").fetchall()
        return [RolloutRecord.model_validate_json(str(row[0])) for row in rows]

    # --- approvals ---------------------------------------------------------------------------

    def put_approval(
        self,
        conn: sqlite3.Connection,
        *,
        request_id: str,
        rollout_id: str,
        kind: str,
        record: Mapping[str, JsonValue],
        status: str,
    ) -> None:
        conn.execute(
            "INSERT INTO approvals (request_id, rollout_id, kind, record, status) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(request_id) DO UPDATE SET "
            "record = excluded.record, status = excluded.status",
            (request_id, rollout_id, kind, canonical_json(dict(record)), status),
        )

    def approval(
        self, request_id: str, conn: sqlite3.Connection | None = None
    ) -> dict[str, Any] | None:
        query = "SELECT rollout_id, kind, record, status FROM approvals WHERE request_id = ?"
        if conn is not None:
            row = conn.execute(query, (request_id,)).fetchone()
        else:
            with self._read() as reader:
                row = reader.execute(query, (request_id,)).fetchone()
        if row is None:
            return None
        return {
            "request_id": request_id,
            "rollout_id": str(row[0]),
            "kind": str(row[1]),
            "record": json.loads(str(row[2])),
            "status": str(row[3]),
        }

    # --- lease -------------------------------------------------------------------------------

    def acquire_lease(self, name: str, holder: str, ttl_seconds: float) -> Lease | None:
        """Take or renew the lease; a takeover after expiry increments the fencing token."""
        now = self.clock()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT holder, token, expires_at FROM lease WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                token = 1
            elif str(row[0]) == holder and float(row[2]) > now:
                token = int(row[1])
            elif float(row[2]) <= now:
                token = int(row[1]) + 1
            else:
                return None
            conn.execute(
                "INSERT INTO lease (name, holder, token, expires_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET holder = excluded.holder, "
                "token = excluded.token, expires_at = excluded.expires_at",
                (name, holder, token, now + ttl_seconds),
            )
        return Lease(name, holder, token, now + ttl_seconds)

    def check_lease(self, conn: sqlite3.Connection, lease: Lease) -> None:
        """Inside a write transaction: refuse unless ``lease`` is still the current, live lease."""
        row = conn.execute(
            "SELECT holder, token, expires_at FROM lease WHERE name = ?", (lease.name,)
        ).fetchone()
        if (
            row is None
            or str(row[0]) != lease.holder
            or int(row[1]) != lease.token
            or float(row[2]) <= self.clock()
        ):
            raise LeaseLostError(f"lease {lease.name} is no longer held by {lease.holder}")

    def lease_holder(self, name: str) -> Lease | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT holder, token, expires_at FROM lease WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return None
        return Lease(name, str(row[0]), int(row[1]), float(row[2]))

    def release_lease(self, lease: Lease) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE lease SET expires_at = 0 WHERE name = ? AND holder = ? AND token = ?",
                (lease.name, lease.holder, lease.token),
            )

    # --- audit chain -------------------------------------------------------------------------

    def append_audit(
        self,
        conn: sqlite3.Connection,
        *,
        actor: str,
        action: str,
        rollout_id: str | None,
        data: Mapping[str, JsonValue],
    ) -> AuditEntry:
        row = conn.execute("SELECT seq, hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        seq, prev = (0, GENESIS_HASH) if row is None else (int(row[0]) + 1, str(row[1]))
        fields: dict[str, Any] = {
            "seq": seq,
            "ts": datetime.fromtimestamp(self.clock(), UTC).isoformat(),
            "actor": actor,
            "action": action,
            "rollout_id": rollout_id,
            "data": dict(data),
            "prev_hash": prev,
        }
        fields["hash"] = _entry_hash(fields)
        conn.execute(
            "INSERT INTO audit (seq, ts, actor, action, rollout_id, data, prev_hash, hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                seq,
                fields["ts"],
                actor,
                action,
                rollout_id,
                canonical_json(fields["data"]),
                prev,
                fields["hash"],
            ),
        )
        return AuditEntry.model_validate(fields)

    def audit_entries(self, rollout_id: str | None = None) -> list[AuditEntry]:
        query = "SELECT seq, ts, actor, action, rollout_id, data, prev_hash, hash FROM audit"
        params: tuple[str, ...] = ()
        if rollout_id is not None:
            query += " WHERE rollout_id = ?"
            params = (rollout_id,)
        with self._read() as conn:
            rows = conn.execute(query + " ORDER BY seq", params).fetchall()
        return [
            AuditEntry(
                seq=int(r[0]),
                ts=str(r[1]),
                actor=str(r[2]),
                action=str(r[3]),
                rollout_id=None if r[4] is None else str(r[4]),
                data=json.loads(str(r[5])),
                prev_hash=str(r[6]),
                hash=str(r[7]),
            )
            for r in rows
        ]

    def verify_audit(self, *, expect_head: str | None = None) -> AuditVerification:
        prev = GENESIS_HASH
        count = 0
        for entry in self.audit_entries():
            if entry.seq != count:
                return AuditVerification(False, count, prev, "sequence gap or reordering", count)
            if entry.prev_hash != prev:
                return AuditVerification(False, count, prev, "broken hash chain", entry.seq)
            if _entry_hash(entry.model_dump(mode="json")) != entry.hash:
                return AuditVerification(False, count, prev, "entry content modified", entry.seq)
            prev = entry.hash
            count += 1
        if expect_head is not None and expect_head != prev:
            return AuditVerification(False, count, prev, "head mismatch (truncated)", None)
        return AuditVerification(True, count, prev)

    def export_audit(self, path: Path) -> int:
        """Write the chain as canonical JSONL (one entry per line) for offline verification."""
        entries = self.audit_entries()
        text = "".join(canonical_json(e.model_dump(mode="json")) + "\n" for e in entries)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return len(entries)
