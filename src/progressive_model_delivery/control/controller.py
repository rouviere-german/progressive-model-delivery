"""The reconciliation controller.

One pass (:meth:`Controller.reconcile`):

1. Take or renew the SQLite lease; a controller without the lease does nothing.
2. Read the open rollout and the signature-verified manifests of champion and challenger.
3. Collect per-stage evidence from the logs as of ``now`` (matured labels only), evaluate every
   guardrail e-value and apply the stage rule (:func:`~.guardrails.decide`).
4. In one write transaction: re-check the lease (fencing token) and the rollout revision, then
   apply the decision (advance, request promotion, roll back, abort) together with an audit entry
   that carries the evidence: every e-value and threshold, sample counts, drift statistics and
   the spec hash.

A pass is a pure function of the control-plane state, the logs and the clock, so running it
twice at the same time changes nothing the second time, a replay on a copy of the state produces
the same records and audit hashes, and a crash anywhere before the commit leaves the previous
state intact for the next controller (see ``tests/integration/test_controller.py``).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import JsonValue

from progressive_model_delivery.control.evidence import CollectedEvidence, EvidenceReader
from progressive_model_delivery.control.guardrails import (
    Decision,
    DecisionResult,
    TestResult,
    decide,
    evaluate,
)
from progressive_model_delivery.control.operations import advance, close
from progressive_model_delivery.control.store import ControlStore, RolloutRecord
from progressive_model_delivery.core.logging import get_logger
from progressive_model_delivery.models.features import REGIONS
from progressive_model_delivery.observability import STAGE_CODES, RolloutMetrics
from progressive_model_delivery.registry.lifecycle import Stage
from progressive_model_delivery.registry.release import ReleaseStore
from progressive_model_delivery.registry.signing import TrustStore

__all__ = ["LEASE_NAME", "Controller", "ReconcileOutcome", "ReconcileReport"]

LEASE_NAME = "controller"
_log = get_logger("progressive_model_delivery.controller")


class ReconcileOutcome(StrEnum):
    NOT_LEADER = "not_leader"
    IDLE = "idle"
    HOLD = "hold"
    ADVANCED = "advanced"
    PROMOTION_REQUESTED = "promotion_requested"
    ROLLED_BACK = "rolled_back"
    ABORTED = "aborted"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    outcome: ReconcileOutcome
    rollout_id: str | None = None
    stage_index: int | None = None
    evidence: dict[str, JsonValue] = field(default_factory=dict)


def _round(value: float) -> float:
    return round(value, 4) if math.isfinite(value) else -1e9


def evidence_payload(
    record: RolloutRecord,
    now: float,
    collected: CollectedEvidence,
    results: list[TestResult],
    decision: DecisionResult,
) -> dict[str, JsonValue]:
    """The audit evidence of one evaluation (JSON, deterministic)."""
    window = record.current
    tests: list[JsonValue] = [
        {
            "test": result.test,
            "guardrail": result.label,
            "log_e": _round(float(result.log_e)),
            "threshold": _round(result.threshold),
            "crossed": bool(result.crossed),
        }
        for result in results
    ]
    return {
        "evaluated_at": now,
        "spec_sha256": record.spec_sha256,
        "stage_index": record.stage_index,
        "stage_elapsed_seconds": round(now - window.started_at, 3),
        "stage_samples": int(collected.stages[record.stage_index].samples),
        "decision": Decision(int(decision.decision)).name,
        "harm_breached": bool(decision.harm_breached),
        "promotion_evidence": bool(decision.promotion_evidence),
        "minimums_met": bool(decision.minimums_met),
        "tests": tests,
        "drift": {
            "feature_psi": dict(collected.drift.feature_psi),
            "score_psi": collected.drift.score_psi,
            "samples": collected.drift.samples,
            "drift_ok": collected.drift.drift_ok,
            "score_breach": collected.drift.score_breach,
        },
        "counts": collected.counts,
        "challenger_exposure": int(sum(float(stage.exposure) for stage in collected.stages)),
    }


class Controller:
    def __init__(
        self,
        *,
        control: ControlStore,
        evidence: EvidenceReader,
        releases: ReleaseStore,
        trust: TrustStore,
        controller_id: str,
        lease_seconds: float,
        metrics: RolloutMetrics | None = None,
    ) -> None:
        self.control = control
        self.evidence = evidence
        self.releases = releases
        self.trust = trust
        self.controller_id = controller_id
        self.lease_seconds = lease_seconds
        self.metrics = metrics
        self.last_report: ReconcileReport | None = None
        self.holds_lease = False

    def reconcile(self) -> ReconcileReport:
        report = self._reconcile()
        self.last_report = report
        if self.metrics is not None:
            self.metrics.reconcile_total.labels(str(report.outcome)).inc()
            self.metrics.lease_held.set(1.0 if self.holds_lease else 0.0)
        return report

    def _reconcile(self) -> ReconcileReport:
        lease = self.control.acquire_lease(LEASE_NAME, self.controller_id, self.lease_seconds)
        self.holds_lease = lease is not None
        if lease is None:
            return ReconcileReport(ReconcileOutcome.NOT_LEADER)
        record = self.control.open_rollout()
        if record is None:
            return ReconcileReport(ReconcileOutcome.IDLE)
        now = self.control.clock()
        champion = self.releases.verify(record.champion, self.trust).manifest
        challenger = self.releases.verify(record.challenger, self.trust).manifest
        collected = self.evidence.collect(record, now, champion=champion, challenger=challenger)
        results = evaluate(record.spec, collected.stages, REGIONS)
        window = record.current
        decision = decide(
            record.spec,
            stage_index=record.stage_index,
            elapsed_seconds=now - window.started_at,
            stage_samples=collected.stages[record.stage_index].samples,
            results=results,
            drift_ok=collected.drift.drift_ok,
            score_breach=collected.drift.score_breach,
            awaiting_approval=record.status == "awaiting_approval",
        )
        payload = evidence_payload(record, now, collected, results, decision)
        self._observe(record, collected, results)
        chosen = Decision(int(decision.decision))
        if chosen is Decision.HOLD:
            return ReconcileReport(
                ReconcileOutcome.HOLD, record.rollout_id, record.stage_index, payload
            )
        with self.control.transaction() as conn:
            self.control.check_lease(conn, lease)
            current = self.control.rollout(record.rollout_id, conn)
            if current.revision != record.revision:
                return ReconcileReport(
                    ReconcileOutcome.STALE, record.rollout_id, record.stage_index, payload
                )
            actor = f"controller:{self.controller_id}"
            if chosen is Decision.ROLLBACK:
                breached = [r.label for r in results if r.test == "harm" and bool(r.crossed)]
                if collected.drift.score_breach:
                    breached.append("score_psi")
                reason = "guardrail breach: " + ", ".join(breached)
                close(
                    self.control, conn, record, actor=actor, status="rolled_back",
                    reason=reason, evidence=payload,
                )
                outcome = ReconcileOutcome.ROLLED_BACK
            elif chosen is Decision.ABORT:
                close(
                    self.control, conn, record, actor=actor, status="aborted",
                    reason="stage maximum duration exceeded without meeting its gate",
                    evidence=payload,
                )
                outcome = ReconcileOutcome.ABORTED
            else:
                advance(self.control, conn, record, actor=actor, evidence=payload)
                outcome = (
                    ReconcileOutcome.PROMOTION_REQUESTED
                    if chosen is Decision.REQUEST_PROMOTION
                    else ReconcileOutcome.ADVANCED
                )
        _log.info(
            "rollout_transition",
            rollout_id=record.rollout_id,
            outcome=str(outcome),
            stage_index=record.stage_index,
        )
        if self.metrics is not None and outcome is ReconcileOutcome.ROLLED_BACK:
            for result in results:
                if result.test == "harm" and bool(result.crossed):
                    self.metrics.guardrail_breaches_total.labels(result.label).inc()
        return ReconcileReport(outcome, record.rollout_id, record.stage_index, payload)

    def _observe(
        self, record: RolloutRecord, collected: CollectedEvidence, results: list[TestResult]
    ) -> None:
        if self.metrics is None:
            return
        self.metrics.rollout_stage.labels(record.challenger[:12]).set(
            STAGE_CODES[Stage(record.current.kind.value)]
        )
        self.metrics.canary_weight.set(record.current.weight_bp / 10_000)
        for result in results:
            self.metrics.set_e_value(result.label, result.test, float(result.log_e))
        for feature, value in collected.drift.feature_psi.items():
            self.metrics.feature_psi.labels(feature).set(value)
        if collected.drift.score_psi is not None:
            self.metrics.score_psi.set(collected.drift.score_psi)

    def run(self, stop: threading.Event, interval_seconds: float) -> None:
        """Reconcile every ``interval_seconds`` until ``stop`` is set (never raises)."""
        while not stop.is_set():
            try:
                self.reconcile()
            except Exception as exc:  # noqa: BLE001 - the loop must survive transient faults
                _log.warning("reconcile_failed", error_type=type(exc).__name__)
            stop.wait(interval_seconds)
