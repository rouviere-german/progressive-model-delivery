"""Transactional rollout operations shared by the controller and the control API.

Each function takes an open write transaction, updates the rollout (compare-and-swap on its
revision), moves the release lifecycle through a legal edge only, and appends the audit entry
with its evidence in the same transaction. A caller that holds the controller lease checks it in
the same transaction first (see :mod:`progressive_model_delivery.control.controller`).
"""

from __future__ import annotations

import secrets
import sqlite3
from collections.abc import Mapping

from pydantic import JsonValue

from progressive_model_delivery.control.spec import RolloutSpec, StageKind
from progressive_model_delivery.control.store import (
    ControlStore,
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
    apply,
)

__all__ = [
    "OperationError",
    "advance",
    "bootstrap_champion",
    "close",
    "promote",
    "propose_rollout",
]


class OperationError(ValueError):
    """The operation is not allowed in the current control-plane state."""


def bootstrap_champion(
    store: ControlStore, conn: sqlite3.Connection, release_id: str, *, actor: str
) -> None:
    """Install the very first champion (only when none exists). Later champions are promoted."""
    if store.champion(conn) is not None:
        raise OperationError("a champion already exists; replace it through a rollout")
    store.set_champion(conn, release_id)
    store.set_lifecycle(conn, release_id, Production())
    store.append_audit(
        conn,
        actor=actor,
        action="champion_bootstrapped",
        rollout_id=None,
        data={"release_id": release_id},
    )


def propose_rollout(
    store: ControlStore,
    conn: sqlite3.Connection,
    *,
    challenger: str,
    spec: RolloutSpec,
    actor: str,
    salt: str | None = None,
) -> RolloutRecord:
    """Open a rollout for a (verified) challenger; it starts in the shadow stage.

    The assignment salt is drawn fresh per rollout so the same customers are not always the test
    group. ``salt`` overrides it, which only the demo does (``demo --seed``), to make a run's canary
    numbers reproducible; a deployment must leave it unset.
    """
    champion = store.champion(conn)
    if champion is None:
        raise OperationError("no champion is installed")
    if challenger == champion:
        raise OperationError("the challenger is the current champion")
    if store.open_rollout(conn) is not None:
        raise OperationError("another rollout is already open")
    current = store.lifecycle_in(conn, challenger)
    if current is not None and not isinstance(current, Staged):
        raise OperationError(f"release is already {current.stage}; publish a new release")
    now = store.clock()
    record = RolloutRecord(
        rollout_id=ControlStore.new_rollout_id(),
        champion=champion,
        challenger=challenger,
        spec=spec,
        spec_sha256=spec.sha256(),
        status="active",
        stage_index=0,
        stages=(StageWindow(index=0, kind=StageKind.SHADOW, weight_bp=0, started_at=now),),
        proposed_by=actor,
        created_at=now,
        salt=salt if salt is not None else secrets.token_hex(16),
        revision=0,
    )
    try:
        store.insert_rollout(conn, record)
    except StaleWriteError as exc:
        raise OperationError("another rollout is already open") from exc
    store.set_lifecycle(conn, challenger, Staged().start_shadow())
    store.append_audit(
        conn,
        actor=actor,
        action="rollout_proposed",
        rollout_id=record.rollout_id,
        data={
            "challenger": challenger,
            "champion": champion,
            "spec_name": spec.name,
            "spec_version": spec.version,
            "spec_sha256": record.spec_sha256,
            "lifecycle": "staged -> shadow",
        },
    )
    return record


def _close_window(record: RolloutRecord, now: float) -> tuple[StageWindow, ...]:
    windows = list(record.stages)
    windows[-1] = windows[-1].model_copy(update={"ended_at": now})
    return tuple(windows)


def advance(
    store: ControlStore,
    conn: sqlite3.Connection,
    record: RolloutRecord,
    *,
    actor: str,
    evidence: Mapping[str, JsonValue],
) -> RolloutRecord:
    """Move to the next stage, or to ``awaiting_approval`` after the last canary stage."""
    if record.status != "active":
        raise OperationError(f"rollout is {record.status}")
    now = store.clock()
    next_index = record.stage_index + 1
    if next_index >= len(record.spec.stages):
        updated = record.model_copy(update={"status": "awaiting_approval"})
        stored = store.update_rollout(conn, updated, expected_revision=record.revision)
        store.append_audit(
            conn,
            actor=actor,
            action="promotion_requested",
            rollout_id=record.rollout_id,
            data={"stage_index": record.stage_index, "evidence": dict(evidence)},
        )
        return stored
    stage = record.spec.stages[next_index]
    weight = CanaryWeight(stage.weight_bp)
    lifecycle = store.lifecycle_in(conn, record.challenger)
    if isinstance(lifecycle, Shadow):
        new_state = lifecycle.start_canary(weight)
    elif isinstance(lifecycle, Canary):
        new_state = lifecycle.increase(weight)
    else:
        raise IllegalTransitionError(f"cannot advance a release that is {lifecycle}")
    windows = (
        *_close_window(record, now),
        StageWindow(index=next_index, kind=stage.kind, weight_bp=stage.weight_bp, started_at=now),
    )
    updated = record.model_copy(update={"stage_index": next_index, "stages": windows})
    stored = store.update_rollout(conn, updated, expected_revision=record.revision)
    store.set_lifecycle(conn, record.challenger, new_state)
    store.append_audit(
        conn,
        actor=actor,
        action="stage_advanced",
        rollout_id=record.rollout_id,
        data={
            "from_stage": record.stage_index,
            "to_stage": next_index,
            "canary_weight": stage.weight,
            "evidence": dict(evidence),
        },
    )
    return stored


def close(
    store: ControlStore,
    conn: sqlite3.Connection,
    record: RolloutRecord,
    *,
    actor: str,
    status: RolloutStatus,
    reason: str,
    evidence: Mapping[str, JsonValue],
) -> RolloutRecord:
    """Roll back (``rolled_back``) or abort (``aborted``): the champion keeps all traffic."""
    if status not in ("rolled_back", "aborted"):
        raise OperationError("close() ends a rollout with rolled_back or aborted")
    if not record.is_open:
        raise OperationError(f"rollout is already {record.status}")
    now = store.clock()
    lifecycle = store.lifecycle_in(conn, record.challenger)
    if lifecycle is None:
        raise OperationError("challenger has no lifecycle record")
    updated = record.model_copy(
        update={
            "status": status,
            "stages": _close_window(record, now),
            "decided_at": now,
            "decision_reason": reason,
        }
    )
    stored = store.update_rollout(conn, updated, expected_revision=record.revision)
    store.set_lifecycle(conn, record.challenger, apply(lifecycle, "roll_back"))
    store.append_audit(
        conn,
        actor=actor,
        action=status,
        rollout_id=record.rollout_id,
        data={"reason": reason, "stage_index": record.stage_index, "evidence": dict(evidence)},
    )
    return stored


def promote(
    store: ControlStore,
    conn: sqlite3.Connection,
    record: RolloutRecord,
    *,
    approved_by: tuple[str, ...],
    evidence: Mapping[str, JsonValue],
) -> RolloutRecord:
    """Make the challenger the champion (after approval) and retire the old champion."""
    if record.status != "awaiting_approval":
        raise OperationError("only a rollout awaiting approval can be promoted")
    now = store.clock()
    lifecycle = store.lifecycle_in(conn, record.challenger)
    if not isinstance(lifecycle, Canary):
        raise IllegalTransitionError("the challenger is not in a canary stage")
    old = store.lifecycle_in(conn, record.champion)
    updated = record.model_copy(
        update={
            "status": "promoted",
            "stages": _close_window(record, now),
            "decided_at": now,
            "decision_reason": "promotion approved",
        }
    )
    stored = store.update_rollout(conn, updated, expected_revision=record.revision)
    store.set_lifecycle(conn, record.challenger, lifecycle.promote())
    if isinstance(old, Production):
        store.set_lifecycle(conn, record.champion, old.retire())
    store.set_champion(conn, record.challenger)
    store.append_audit(
        conn,
        actor=",".join(approved_by),
        action="promoted",
        rollout_id=record.rollout_id,
        data={
            "new_champion": record.challenger,
            "retired": record.champion,
            "approved_by": list(approved_by),
            "evidence": dict(evidence),
        },
    )
    return stored
