"""Transactional rollout operations: the only legal paths through the control plane."""

from __future__ import annotations

from pathlib import Path

import pytest

from progressive_model_delivery.control.operations import (
    OperationError,
    advance,
    bootstrap_champion,
    close,
    promote,
    propose_rollout,
)
from progressive_model_delivery.control.spec import RolloutSpec, bundled_spec
from progressive_model_delivery.control.store import ControlStore, RolloutRecord
from progressive_model_delivery.registry.lifecycle import (
    Canary,
    CanaryWeight,
    IllegalTransitionError,
    Production,
    Retired,
    RolledBack,
    Shadow,
    Staged,
)

pytestmark = pytest.mark.unit

CHAMPION = "a" * 64
CHALLENGER = "b" * 64


class Clock:
    def __init__(self) -> None:
        self.now = 10_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def spec() -> RolloutSpec:
    return bundled_spec("demo")


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> ControlStore:
    store = ControlStore(tmp_path / "control.sqlite", clock=clock)
    with store.transaction() as conn:
        bootstrap_champion(store, conn, CHAMPION, actor="ravi")
    return store


def _propose(store: ControlStore, spec: RolloutSpec, challenger: str = CHALLENGER) -> RolloutRecord:
    with store.transaction() as conn:
        return propose_rollout(store, conn, challenger=challenger, spec=spec, actor="dana")


def _advance(store: ControlStore, record: RolloutRecord) -> RolloutRecord:
    with store.transaction() as conn:
        return advance(store, conn, record, actor="controller:c1", evidence={"e": 1})


def test_bootstrap_installs_a_production_champion_exactly_once(store: ControlStore) -> None:
    assert store.champion() == CHAMPION
    assert store.lifecycle(CHAMPION) == Production()
    entries = store.audit_entries()
    assert [entry.action for entry in entries] == ["champion_bootstrapped"]
    with pytest.raises(OperationError, match="already exists"), store.transaction() as conn:
        bootstrap_champion(store, conn, "c" * 64, actor="ravi")


def test_proposing_a_rollout_starts_the_shadow_stage_and_records_the_spec_hash(
    store: ControlStore, spec: RolloutSpec, clock: Clock
) -> None:
    record = _propose(store, spec)
    assert record.status == "active"
    assert record.stage_index == 0
    assert record.current.kind.value == "shadow"
    assert record.spec_sha256 == spec.sha256()
    assert record.created_at == clock.now
    assert len(record.salt) == 32
    assert store.lifecycle(CHALLENGER) == Shadow()
    proposed = store.audit_entries(record.rollout_id)[0]
    assert proposed.action == "rollout_proposed"
    assert proposed.data["spec_sha256"] == spec.sha256()
    assert proposed.data["lifecycle"] == "staged -> shadow"


def test_a_rollout_needs_a_champion_and_a_different_challenger(
    tmp_path: Path, spec: RolloutSpec
) -> None:
    empty = ControlStore(tmp_path / "empty.sqlite")
    with pytest.raises(OperationError, match="no champion"), empty.transaction() as conn:
        propose_rollout(empty, conn, challenger=CHALLENGER, spec=spec, actor="dana")


def test_the_champion_cannot_challenge_itself(store: ControlStore, spec: RolloutSpec) -> None:
    with pytest.raises(OperationError, match="is the current champion"):
        _propose(store, spec, CHAMPION)


def test_only_one_rollout_at_a_time_and_only_staged_releases(
    store: ControlStore, spec: RolloutSpec
) -> None:
    _propose(store, spec)
    with pytest.raises(OperationError, match="already open"):
        _propose(store, spec, "c" * 64)


def test_a_rolled_back_release_cannot_be_proposed_again(
    store: ControlStore, spec: RolloutSpec
) -> None:
    record = _propose(store, spec)
    with store.transaction() as conn:
        close(
            store, conn, record, actor="dana", status="rolled_back", reason="manual", evidence={}
        )
    assert store.lifecycle(CHALLENGER) == RolledBack()
    with pytest.raises(OperationError, match="already rolled_back"):
        _propose(store, spec)


def test_advancing_walks_the_lifecycle_and_closes_each_window(
    store: ControlStore, spec: RolloutSpec, clock: Clock
) -> None:
    record = _propose(store, spec)
    weights = [stage.weight_bp for stage in spec.stages if stage.weight_bp]
    for expected in weights:
        clock.now += 100.0
        record = _advance(store, record)
        assert record.current.weight_bp == expected
        assert record.stages[-2].ended_at == clock.now
        assert store.lifecycle(CHALLENGER) == Canary(CanaryWeight(expected))
    assert record.stage_index == len(spec.stages) - 1
    assert record.revision == len(weights)
    actions = [entry.action for entry in store.audit_entries(record.rollout_id)]
    assert actions == ["rollout_proposed", *["stage_advanced"] * len(weights)]


def test_advancing_past_the_last_stage_asks_for_promotion_instead(
    store: ControlStore, spec: RolloutSpec
) -> None:
    record = _propose(store, spec)
    for _ in range(len(spec.stages) - 1):
        record = _advance(store, record)
    record = _advance(store, record)
    assert record.status == "awaiting_approval"
    assert record.stage_index == len(spec.stages) - 1  # the stage does not move
    last = store.audit_entries(record.rollout_id)[-1]
    assert last.action == "promotion_requested"
    assert last.data["evidence"] == {"e": 1}
    with pytest.raises(OperationError, match="awaiting_approval"):
        _advance(store, record)


def test_promotion_makes_the_challenger_champion_and_retires_the_old_one(
    store: ControlStore, spec: RolloutSpec
) -> None:
    record = _propose(store, spec)
    for _ in range(len(spec.stages)):
        record = _advance(store, record)
    with store.transaction() as conn:
        promoted = promote(store, conn, record, approved_by=("ravi",), evidence={"seq": 3})
    assert promoted.status == "promoted"
    assert promoted.decision_reason == "promotion approved"
    assert store.champion() == CHALLENGER
    assert store.lifecycle(CHALLENGER) == Production()
    assert store.lifecycle(CHAMPION) == Retired()
    entry = store.audit_entries(record.rollout_id)[-1]
    assert entry.action == "promoted"
    assert entry.actor == "ravi"
    assert entry.data["retired"] == CHAMPION
    assert store.open_rollout() is None


def test_promotion_requires_the_awaiting_approval_state_and_a_canary_stage(
    store: ControlStore, spec: RolloutSpec
) -> None:
    record = _propose(store, spec)
    with pytest.raises(OperationError, match="awaiting approval"), store.transaction() as conn:
        promote(store, conn, record, approved_by=("ravi",), evidence={})
    forced = record.model_copy(update={"status": "awaiting_approval"})
    with pytest.raises(IllegalTransitionError, match="not in a canary stage"), store.transaction() as conn:
        promote(store, conn, forced, approved_by=("ravi",), evidence={})


def test_closing_rolls_the_release_back_and_is_not_repeatable(
    store: ControlStore, spec: RolloutSpec, clock: Clock
) -> None:
    record = _propose(store, spec)
    record = _advance(store, record)
    clock.now += 50.0
    with store.transaction() as conn:
        closed = close(
            store,
            conn,
            record,
            actor="controller:c1",
            status="rolled_back",
            reason="guardrail breach: worst_subgroup_error[south]",
            evidence={"tests": []},
        )
    assert closed.status == "rolled_back"
    assert closed.decided_at == clock.now
    assert closed.current.ended_at == clock.now
    assert store.lifecycle(CHALLENGER) == RolledBack()
    assert store.champion() == CHAMPION
    with pytest.raises(OperationError, match="already rolled_back"), store.transaction() as conn:
        close(store, conn, closed, actor="x", status="aborted", reason="r", evidence={})


def test_close_only_accepts_terminal_failure_states(
    store: ControlStore, spec: RolloutSpec
) -> None:
    record = _propose(store, spec)
    with pytest.raises(OperationError, match="rolled_back or aborted"), store.transaction() as conn:
        close(store, conn, record, actor="x", status="promoted", reason="r", evidence={})


def test_close_needs_a_lifecycle_record(store: ControlStore, spec: RolloutSpec) -> None:
    record = _propose(store, spec)
    with store.transaction() as conn:
        conn.execute("DELETE FROM lifecycle WHERE release_id = ?", (CHALLENGER,))
    with pytest.raises(OperationError, match="no lifecycle record"), store.transaction() as conn:
        close(store, conn, record, actor="x", status="aborted", reason="r", evidence={})


def test_aborting_a_rollout_keeps_the_champion(store: ControlStore, spec: RolloutSpec) -> None:
    record = _propose(store, spec)
    with store.transaction() as conn:
        aborted = close(
            store, conn, record, actor="controller:c1", status="aborted", reason="timeout",
            evidence={},
        )
    assert aborted.status == "aborted"
    assert store.champion() == CHAMPION
    assert [e.action for e in store.audit_entries(record.rollout_id)][-1] == "aborted"


def test_advancing_a_release_that_left_the_canary_path_is_refused(
    store: ControlStore, spec: RolloutSpec
) -> None:
    record = _propose(store, spec)
    with store.transaction() as conn:
        store.set_lifecycle(conn, CHALLENGER, Staged())
    with pytest.raises(IllegalTransitionError, match="cannot advance"):
        _advance(store, record)
