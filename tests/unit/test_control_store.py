"""The SQLite control plane: audit chain, lease fencing, compare-and-swap, single open rollout."""

from __future__ import annotations

import itertools
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from progressive_model_delivery.control.spec import RolloutSpec, StageKind, bundled_spec
from progressive_model_delivery.control.store import (
    GENESIS_HASH,
    ControlStore,
    LeaseLostError,
    RolloutRecord,
    RolloutStatus,
    StageWindow,
    StaleWriteError,
)
from progressive_model_delivery.registry.lifecycle import (
    Canary,
    CanaryWeight,
    IllegalTransitionError,
    Production,
    Shadow,
    Staged,
)

pytestmark = pytest.mark.unit


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> ControlStore:
    return ControlStore(tmp_path / "nested" / "control.sqlite", clock=clock)


def _record(
    store: ControlStore,
    spec: RolloutSpec,
    *,
    challenger: str = "b" * 64,
    status: RolloutStatus = "active",
) -> RolloutRecord:
    return RolloutRecord(
        rollout_id=ControlStore.new_rollout_id(),
        champion="a" * 64,
        challenger=challenger,
        spec=spec,
        spec_sha256=spec.sha256(),
        status=status,
        stage_index=0,
        stages=(StageWindow(index=0, kind=StageKind.SHADOW, weight_bp=0, started_at=1.0),),
        proposed_by="dana",
        created_at=1.0,
        salt="c" * 32,
        revision=0,
    )


@pytest.fixture
def spec() -> RolloutSpec:
    return bundled_spec("demo")


# --- champion and lifecycle -------------------------------------------------------------------


def test_the_store_creates_its_directory_and_starts_empty(store: ControlStore) -> None:
    assert store.path.parent.is_dir()
    assert store.champion() is None
    assert store.open_rollout() is None
    assert store.rollouts() == []
    assert store.lifecycle("a" * 64) is None
    assert store.verify_audit().ok


def test_champion_and_lifecycle_round_trip(store: ControlStore) -> None:
    with store.transaction() as conn:
        store.set_champion(conn, "a" * 64)
        store.set_lifecycle(conn, "a" * 64, Production())
        assert store.champion(conn) == "a" * 64
        assert store.lifecycle_in(conn, "a" * 64) == Production()
    assert store.champion() == "a" * 64
    assert store.lifecycle("a" * 64) == Production()
    with store.transaction() as conn:
        store.set_champion(conn, "b" * 64)
        store.set_lifecycle(conn, "a" * 64, Canary(CanaryWeight.W25))
    assert store.champion() == "b" * 64
    assert store.lifecycle("a" * 64) == Canary(CanaryWeight.W25)


def test_an_impossible_stored_lifecycle_is_refused_on_read(store: ControlStore) -> None:
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO lifecycle (release_id, stage, weight_bp, updated_at) VALUES (?,?,?,?)",
            ("d" * 64, "production", 2_500, 1.0),
        )
    with pytest.raises(IllegalTransitionError, match="cannot carry a traffic weight"):
        store.lifecycle("d" * 64)


def test_a_failed_transaction_leaves_nothing_behind(store: ControlStore) -> None:
    def write_then_fail() -> None:
        with store.transaction() as conn:
            store.set_champion(conn, "a" * 64)
            store.append_audit(conn, actor="x", action="y", rollout_id=None, data={})
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        write_then_fail()
    assert store.champion() is None
    assert store.audit_entries() == []


# --- rollouts ---------------------------------------------------------------------------------


def test_only_one_rollout_can_be_open_at_a_time(store: ControlStore, spec: RolloutSpec) -> None:
    first = _record(store, spec)
    with store.transaction() as conn:
        store.insert_rollout(conn, first)
    with pytest.raises(StaleWriteError, match="already open"), store.transaction() as conn:
        store.insert_rollout(conn, _record(store, spec, challenger="e" * 64))
    assert store.open_rollout() is not None
    with store.transaction() as conn:
        store.update_rollout(
            conn, first.model_copy(update={"status": "rolled_back"}), expected_revision=0
        )
    assert store.open_rollout() is None
    with store.transaction() as conn:  # a closed rollout frees the slot
        store.insert_rollout(conn, _record(store, spec, challenger="e" * 64))
    assert store.open_rollout() is not None
    assert len(store.rollouts()) == 2


def test_updates_are_compare_and_swap_on_the_revision(
    store: ControlStore, spec: RolloutSpec
) -> None:
    record = _record(store, spec)
    with store.transaction() as conn:
        store.insert_rollout(conn, record)
        stored = store.update_rollout(conn, record, expected_revision=0)
    assert stored.revision == 1
    assert store.rollout(record.rollout_id).revision == 1
    with pytest.raises(StaleWriteError, match="changed concurrently"), store.transaction() as conn:
        store.update_rollout(conn, record, expected_revision=0)
    with pytest.raises(KeyError):
        store.rollout("f" * 32)


def test_rollout_helpers_accept_a_shared_connection(store: ControlStore, spec: RolloutSpec) -> None:
    record = _record(store, spec)
    with store.transaction() as conn:
        store.insert_rollout(conn, record)
        assert store.rollout(record.rollout_id, conn).rollout_id == record.rollout_id
        assert store.open_rollout(conn) is not None


def test_record_helpers_expose_the_current_window_and_openness(
    store: ControlStore, spec: RolloutSpec
) -> None:
    record = _record(store, spec)
    assert record.current.index == 0
    assert record.is_open
    assert not record.model_copy(update={"status": "promoted"}).is_open
    assert record.model_copy(update={"status": "awaiting_approval"}).is_open


# --- approvals --------------------------------------------------------------------------------


def test_approvals_are_upserted_by_request_id(store: ControlStore, spec: RolloutSpec) -> None:
    record = _record(store, spec)
    with store.transaction() as conn:
        store.insert_rollout(conn, record)
        store.put_approval(
            conn,
            request_id="r1",
            rollout_id=record.rollout_id,
            kind="override",
            record={"approvals": ["ravi"]},
            status="pending",
        )
        store.put_approval(
            conn,
            request_id="r1",
            rollout_id=record.rollout_id,
            kind="override",
            record={"approvals": ["ravi", "rita"]},
            status="executed",
        )
    stored = store.approval("r1")
    assert stored is not None
    assert stored["status"] == "executed"
    assert stored["record"]["approvals"] == ["ravi", "rita"]
    assert store.approval("missing") is None


# --- lease ------------------------------------------------------------------------------------


def test_a_live_lease_cannot_be_taken_over_and_renewal_keeps_the_token(
    store: ControlStore, clock: Clock
) -> None:
    first = store.acquire_lease("controller", "c1", 30.0)
    assert first is not None
    assert first.token == 1
    assert store.acquire_lease("controller", "c2", 30.0) is None
    renewed = store.acquire_lease("controller", "c1", 30.0)
    assert renewed is not None
    assert renewed.token == 1
    held = store.lease_holder("controller")
    assert held is not None
    assert held.holder == "c1"
    assert store.lease_holder("other") is None


def test_an_expired_lease_is_taken_over_with_a_new_fencing_token(
    store: ControlStore, clock: Clock
) -> None:
    first = store.acquire_lease("controller", "c1", 10.0)
    assert first is not None
    clock.now += 11.0
    second = store.acquire_lease("controller", "c2", 10.0)
    assert second is not None
    assert second.token == 2
    with store.transaction() as conn:
        store.check_lease(conn, second)
        with pytest.raises(LeaseLostError, match="no longer held"):
            store.check_lease(conn, first)


def test_a_released_lease_is_immediately_available(store: ControlStore) -> None:
    lease = store.acquire_lease("controller", "c1", 60.0)
    assert lease is not None
    store.release_lease(lease)
    taken = store.acquire_lease("controller", "c2", 60.0)
    assert taken is not None
    assert taken.token == 2
    with store.transaction() as conn, pytest.raises(LeaseLostError):
        store.check_lease(conn, lease)


def test_check_lease_refuses_an_unknown_and_an_expired_lease(
    store: ControlStore, clock: Clock
) -> None:
    lease = store.acquire_lease("controller", "c1", 10.0)
    assert lease is not None
    with store.transaction() as conn:
        store.check_lease(conn, lease)
    clock.now += 11.0
    with store.transaction() as conn, pytest.raises(LeaseLostError):
        store.check_lease(conn, lease)


# --- audit chain ------------------------------------------------------------------------------


def _append(store: ControlStore, count: int, rollout_id: str | None = None) -> None:
    with store.transaction() as conn:
        for index in range(count):
            store.append_audit(
                conn,
                actor=f"actor-{index}",
                action="stage_advanced",
                rollout_id=rollout_id,
                data={"index": index},
            )


def test_the_audit_chain_links_every_entry_to_its_predecessor(store: ControlStore) -> None:
    _append(store, 4, "r" * 32)
    entries = store.audit_entries()
    assert [entry.seq for entry in entries] == [0, 1, 2, 3]
    assert entries[0].prev_hash == GENESIS_HASH
    for previous, entry in itertools.pairwise(entries):
        assert entry.prev_hash == previous.hash
    verified = store.verify_audit()
    assert verified.ok
    assert verified.entries == 4
    assert verified.head == entries[-1].hash
    assert store.verify_audit(expect_head=entries[-1].hash).ok


def test_audit_entries_can_be_filtered_by_rollout(store: ControlStore) -> None:
    _append(store, 2, "a" * 32)
    _append(store, 3, "b" * 32)
    _append(store, 1, None)
    assert len(store.audit_entries()) == 6
    assert len(store.audit_entries("a" * 32)) == 2
    assert len(store.audit_entries("b" * 32)) == 3


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("UPDATE audit SET data = '{\"index\": 99}' WHERE seq = 1", "entry content modified"),
        ("UPDATE audit SET actor = 'mallory' WHERE seq = 2", "entry content modified"),
        ("UPDATE audit SET prev_hash = '" + "0" * 64 + "' WHERE seq = 2", "broken hash chain"),
        ("DELETE FROM audit WHERE seq = 1", "sequence gap or reordering"),
    ],
)
def test_tampering_with_the_audit_log_is_detected(
    store: ControlStore, sql: str, reason: str
) -> None:
    _append(store, 4)
    assert store.verify_audit().ok
    with closing(sqlite3.connect(store.path)) as conn:
        conn.execute(sql)
        conn.commit()
    result = store.verify_audit()
    assert not result.ok
    assert result.error == reason


def test_truncating_the_tail_is_only_detectable_against_a_known_head(
    store: ControlStore,
) -> None:
    _append(store, 4)
    head = store.verify_audit().head
    with closing(sqlite3.connect(store.path)) as conn:
        conn.execute("DELETE FROM audit WHERE seq = 3")
        conn.commit()
    assert store.verify_audit().ok  # a truncated prefix is internally consistent
    truncated = store.verify_audit(expect_head=head)
    assert not truncated.ok
    assert truncated.error == "head mismatch (truncated)"


def test_the_audit_log_exports_as_canonical_jsonl(store: ControlStore, tmp_path: Path) -> None:
    _append(store, 3, "a" * 32)
    target = tmp_path / "export" / "audit.jsonl"
    assert store.export_audit(target) == 3
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [entry["seq"] for entry in parsed] == [0, 1, 2]
    assert all(
        line == json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))
        for line in lines
    )


@given(actions=st.lists(st.sampled_from(["a", "bb", "ccc"]), min_size=0, max_size=12))
def test_appending_any_sequence_of_entries_keeps_the_chain_verifiable(
    tmp_path_factory: pytest.TempPathFactory, actions: list[str]
) -> None:
    store = ControlStore(tmp_path_factory.mktemp("chain") / "control.sqlite")
    with store.transaction() as conn:
        for index, action in enumerate(actions):
            store.append_audit(conn, actor="a", action=action, rollout_id=None, data={"i": index})
    result = store.verify_audit()
    assert result.ok
    assert result.entries == len(actions)
    if actions:
        assert result.head == store.audit_entries()[-1].hash
    else:
        assert result.head == GENESIS_HASH


def test_two_stores_over_the_same_file_share_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "control.sqlite"
    first, second = ControlStore(path), ControlStore(path)
    _append(first, 2)
    _append(second, 2)
    assert first.verify_audit().ok
    assert second.verify_audit().entries == 4


def test_shadow_lifecycle_transitions_used_by_the_store(store: ControlStore) -> None:
    with store.transaction() as conn:
        store.set_lifecycle(conn, "a" * 64, Staged().start_shadow())
        assert store.lifecycle_in(conn, "a" * 64) == Shadow()
