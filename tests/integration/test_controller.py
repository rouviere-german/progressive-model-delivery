"""The reconciliation controller: idempotence, replayability, crash safety and leader election.

These tests drive the controller over a real prediction log written by the real router, with a
simulated clock, so they exercise the whole path: Parquet segments -> DuckDB join -> maturation ->
e-values -> stage rule -> transactional write with audit evidence.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from progressive_model_delivery.control.controller import (
    LEASE_NAME,
    Controller,
    ReconcileOutcome,
)
from progressive_model_delivery.control.evidence import EvidenceReader
from progressive_model_delivery.control.spec import RolloutSpec, load_spec
from progressive_model_delivery.control.store import ControlStore, RolloutRecord
from progressive_model_delivery.models.synthetic import LabelDelay
from progressive_model_delivery.observability import RolloutMetrics
from progressive_model_delivery.registry.release import ReleaseStore
from progressive_model_delivery.serving.router import RouterRuntime, build_router_app
from progressive_model_delivery.serving.shadow_worker import ShadowRuntime, build_shadow_app
from progressive_model_delivery.sim.traffic import SimulatedClock, TrafficConfig, TrafficSimulator
from tests.integration.conftest import API_KEY, SHADOW_TOKEN, ControlEnv, number, sub

pytestmark = [pytest.mark.integration, pytest.mark.slow]

FAST_SPEC = """
name: controller-test
version: 1
alpha_rollback: 0.025
alpha_promotion: 0.025
label_window_seconds: 60
prior_strength: 10
stages:
  - kind: shadow
    min_samples: 1200
    min_duration_seconds: 0
    max_duration_seconds: 100000
  - kind: canary
    weight_percent: 25
    min_samples: 40
    min_duration_seconds: 0
    max_duration_seconds: 100000
  - kind: canary
    weight_percent: 50
    min_samples: 80
    min_duration_seconds: 0
    max_duration_seconds: 100000
guardrails:
  - name: error_rate
    metric: error
    non_inferiority_ratio: 2.0
  - name: worst_subgroup_error
    metric: error
    subgroups: true
  - name: log_loss
    metric: log_loss
    non_inferiority_margin: 0.3
drift:
  feature_psi_threshold: 0.4
  score_psi_threshold: 0.6
  min_samples: 200
"""


@pytest.fixture
def fast_spec() -> RolloutSpec:
    return load_spec(FAST_SPEC)


class Harness:
    """A router plus a controller over one control plane, driven by simulated traffic."""

    def __init__(self, env: ControlEnv, spec: RolloutSpec, *, seed: int = 7) -> None:
        self.env = env
        self.spec = spec
        self.clock = SimulatedClock(start=env.clock[0])
        env.clock[0] = self.clock.now
        self.settings = env.settings(
            shadow_url="http://shadow.internal:8001",
            shadow_token=SHADOW_TOKEN,
            label_ttl_seconds=spec.label_window_seconds,
            feedback_rate_per_second=1e6,
            feedback_burst=1_000_000,
        )
        self.control = ControlStore(self.settings.control_db, clock=self.clock)
        self.registry_metrics = CollectorRegistry()
        self.controller = Controller(
            control=self.control,
            evidence=EvidenceReader(self.settings.predlog_dir),
            releases=ReleaseStore(self.settings.registry_dir),
            trust=env.registry.trust,
            controller_id="c1",
            lease_seconds=60.0,
            metrics=RolloutMetrics.create(self.registry_metrics),
        )
        self.seed = seed

    def __enter__(self) -> Harness:
        self._shadow_app = build_shadow_app(self.settings, background=False)
        shadow_runtime = self._shadow_app.state.runtime
        assert isinstance(shadow_runtime, ShadowRuntime)
        self._shadow_runtime = shadow_runtime
        app = build_router_app(
            self.settings,
            clock=self.clock,
            shadow_transport=httpx.ASGITransport(app=self._shadow_app),
            background=False,
        )
        self.client = TestClient(app)
        self.client.__enter__()
        runtime = self.client.app.state.runtime  # type: ignore[attr-defined]
        assert isinstance(runtime, RouterRuntime)
        self.runtime = runtime
        self.traffic = TrafficSimulator(
            self.client,
            api_key=API_KEY,
            clock=self.clock,
            config=TrafficConfig(
                seed=self.seed,
                requests_per_tick=250,
                tick_seconds=30.0,
                delay=LabelDelay(
                    mean_negative_seconds=5.0, mean_positive_seconds=10.0, loss_rate=0.02
                ),
            ),
            flush=self.flush,
        )
        self.traffic.assert_ttl_fits(self.settings.label_ttl_seconds)
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.__exit__()

    def flush(self) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and self._pending_shadow() > 0:
            time.sleep(0.01)
        self.runtime.flush()
        self._shadow_runtime.writer.flush()

    def _pending_shadow(self) -> float:
        text = self.client.get("/metrics", headers={"X-API-Key": API_KEY}).text
        totals = {"enq": 0.0, "done": 0.0}
        for line in text.splitlines():
            if line.startswith("app_shadow_enqueued_total "):
                totals["enq"] += float(line.rsplit(" ", 1)[1])
            elif line.startswith(("app_shadow_delivered_total ", "app_shadow_failed_total{")):
                totals["done"] += float(line.rsplit(" ", 1)[1])
        return totals["enq"] - totals["done"]

    def run(self, ticks: int) -> list[ReconcileOutcome]:
        outcomes = []
        for _ in range(ticks):
            self.traffic.tick()
            report = self.controller.reconcile()
            outcomes.append(report.outcome)
            if report.outcome is not ReconcileOutcome.HOLD:
                self.runtime.refresh()
            if report.outcome in (
                ReconcileOutcome.ROLLED_BACK,
                ReconcileOutcome.PROMOTION_REQUESTED,
                ReconcileOutcome.ABORTED,
            ):
                break
        return outcomes

    def metric(self, name: str, *contains: str) -> float:
        total = 0.0
        for metric in self.registry_metrics.collect():
            for sample in metric.samples:
                if sample.name != name:
                    continue
                rendered = str(sorted(sample.labels.items()))
                if all(needle in rendered for needle in contains):
                    total += sample.value
        return total


@pytest.fixture
def harness(control_env: ControlEnv, fast_spec: RolloutSpec) -> Iterator[Harness]:
    with Harness(control_env, fast_spec) as harness:
        yield harness


def _propose(harness: Harness, challenger: str) -> RolloutRecord:
    """Open a rollout and let the router pick it up (its refresh loop is off in tests)."""
    from progressive_model_delivery.control.operations import propose_rollout

    with harness.control.transaction() as conn:
        record = propose_rollout(
            harness.control,
            conn,
            challenger=harness.env.release(challenger),
            spec=harness.spec,
            actor="dana",
        )
    harness.runtime.refresh()
    return record


# --- basic outcomes ---------------------------------------------------------------------------


def test_with_no_open_rollout_the_controller_is_idle_but_holds_the_lease(
    harness: Harness,
) -> None:
    report = harness.controller.reconcile()
    assert report.outcome is ReconcileOutcome.IDLE
    assert report.evidence == {}
    assert harness.controller.holds_lease
    assert harness.metric("app_lease_held") == 1.0
    assert harness.metric("app_reconcile_total", "idle") == 1.0


def test_a_pipeline_bug_challenger_is_rolled_back_with_evidence_in_the_audit_log(
    harness: Harness,
) -> None:
    """``feature_order_bug`` passes every schema check and is caught by the guardrails instead."""
    record = _propose(harness, "feature_order_bug")
    outcomes = harness.run(20)
    assert outcomes[-1] is ReconcileOutcome.ROLLED_BACK
    final = harness.control.rollout(record.rollout_id)
    assert final.status == "rolled_back"
    assert final.decision_reason is not None
    assert final.decision_reason.startswith("guardrail breach:")

    entry = harness.control.audit_entries(record.rollout_id)[-1]
    assert entry.action == "rolled_back"
    evidence = entry.data["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["spec_sha256"] == harness.spec.sha256()
    assert evidence["decision"] == "ROLLBACK"
    assert evidence["harm_breached"] is True
    # The evidence records the exposure the decision was made under, not just the verdict.
    assert evidence["stage_kind"] == "shadow"
    assert evidence["canary_weight_percent"] == 0.0
    tests = evidence["tests"]
    assert isinstance(tests, list)
    crossed = [t["guardrail"] for t in tests if isinstance(t, dict) and t["crossed"]]
    assert "error_rate" in crossed
    assert harness.control.verify_audit().ok
    assert harness.metric("app_guardrail_breaches_total", "error_rate") >= 1.0
    assert harness.metric("app_e_value", "error_rate", "harm") > 0.0


def test_a_better_challenger_walks_the_stages_and_asks_for_promotion(harness: Harness) -> None:
    record = _propose(harness, "better")
    outcomes = harness.run(40)
    assert outcomes[-1] is ReconcileOutcome.PROMOTION_REQUESTED
    assert outcomes.count(ReconcileOutcome.ADVANCED) == len(harness.spec.stages) - 1
    final = harness.control.rollout(record.rollout_id)
    assert final.status == "awaiting_approval"
    assert final.stage_index == len(harness.spec.stages) - 1
    assert [w.weight_bp for w in final.stages] == [0, 2_500, 5_000]
    assert harness.metric("app_canary_weight") == 0.5
    assert harness.metric("app_rollout_stage", record.challenger[:12]) == 2.0
    # The challenger really served traffic in the canary stages.
    served = sum(1 for window in final.stages if window.weight_bp)
    assert served == 2
    last = harness.control.audit_entries(record.rollout_id)[-1]
    assert number(last.data, "evidence", "canary_weight_percent") == 50.0
    assert sub(last.data, "evidence")["stage_kind"] == "canary"
    report = harness.controller.last_report
    assert report is not None
    assert number(report.evidence, "challenger_exposure") > 0


def test_an_identical_challenger_is_neither_rolled_back_nor_harmed(harness: Harness) -> None:
    _propose(harness, "identical")
    outcomes = harness.run(10)
    assert ReconcileOutcome.ROLLED_BACK not in outcomes
    report = harness.controller.last_report
    assert report is not None
    tests = report.evidence["tests"]
    assert isinstance(tests, list)
    for test in tests:
        assert isinstance(test, dict)
        if test["test"] == "harm":
            assert test["crossed"] is False


# --- idempotence and replay --------------------------------------------------------------------


def test_reconciling_twice_at_the_same_clock_changes_nothing_the_second_time(
    harness: Harness,
) -> None:
    record = _propose(harness, "better")
    harness.run(6)
    before = harness.control.rollout(record.rollout_id)
    audit_before = harness.control.verify_audit()
    second = harness.controller.reconcile()
    after = harness.control.rollout(record.rollout_id)
    audit_after = harness.control.verify_audit()
    assert second.outcome is ReconcileOutcome.HOLD
    assert after == before
    assert audit_after.head == audit_before.head
    assert audit_after.entries == audit_before.entries


def test_replaying_a_rollout_on_a_copy_of_the_state_reproduces_every_decision(
    control_env: ControlEnv, fast_spec: RolloutSpec, tmp_path: Path
) -> None:
    """The evidence is a pure function of the logs and the clock, so a replay is deterministic."""
    with Harness(control_env, fast_spec, seed=11) as harness:
        record = _propose(harness, "feature_order_bug")
        harness.run(20)
        original = harness.control.audit_entries(record.rollout_id)
        stamps = [
            number(entry.data, "evidence", "evaluated_at")
            for entry in original
            if "evidence" in entry.data
        ]
        predlog = harness.settings.predlog_dir

    replay_dir = tmp_path / "replay"
    shutil.copytree(predlog, replay_dir / "predlog")
    clock = SimulatedClock(start=stamps[0])
    replay_control = ControlStore(replay_dir / "control.sqlite", clock=clock)
    with replay_control.transaction() as conn:
        replay_control.set_champion(conn, control_env.release("champion"))
        from progressive_model_delivery.registry.lifecycle import (
            Production,
            Shadow,
        )

        replay_control.set_lifecycle(conn, control_env.release("champion"), Production())
        replay_control.set_lifecycle(conn, control_env.release("feature_order_bug"), Shadow())
        replay_control.insert_rollout(conn, record)
    replay = Controller(
        control=replay_control,
        evidence=EvidenceReader(replay_dir / "predlog"),
        releases=ReleaseStore(harness.settings.registry_dir),
        trust=control_env.registry.trust,
        controller_id="replay",
        lease_seconds=60.0,
    )
    outcomes = []
    for stamp in stamps:
        clock.advance(stamp - clock.now)
        outcomes.append(replay.reconcile().outcome)

    replayed = replay_control.audit_entries(record.rollout_id)
    assert [e.action for e in replayed] == [
        e.action for e in original if e.action != "rollout_proposed"
    ]
    assert outcomes[-1] is ReconcileOutcome.ROLLED_BACK
    for first, again in zip((e for e in original if "evidence" in e.data), replayed, strict=False):
        assert again.data["evidence"] == first.data["evidence"]
    assert replay_control.rollout(record.rollout_id).status == "rolled_back"


# --- crash safety and leases -------------------------------------------------------------------


def test_a_crash_before_the_commit_leaves_the_previous_state_intact(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _propose(harness, "feature_order_bug")
    harness.run(2)  # not enough matured labels for a decision yet
    before = harness.control.rollout(record.rollout_id)
    head = harness.control.verify_audit().head
    assert before.status == "active"

    from progressive_model_delivery.control import controller as controller_module

    def exploding(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("controller crashed mid-transaction")

    monkeypatch.setattr(controller_module, "close", exploding)
    monkeypatch.setattr(controller_module, "advance", exploding)

    def until_decision() -> None:
        for _ in range(12):
            harness.traffic.tick()
            harness.controller.reconcile()

    with pytest.raises(RuntimeError, match="crashed"):
        until_decision()
    monkeypatch.undo()

    assert harness.control.rollout(record.rollout_id) == before
    assert harness.control.verify_audit().head == head
    # The next controller pass decides normally.
    assert harness.run(12)[-1] is ReconcileOutcome.ROLLED_BACK


def test_the_run_loop_survives_a_failing_pass(harness: Harness) -> None:
    calls = {"n": 0}
    original = harness.controller.reconcile

    def flaky() -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return original()

    harness.controller.reconcile = flaky  # type: ignore[method-assign]
    stop = threading.Event()
    thread = threading.Thread(target=harness.controller.run, args=(stop, 0.01), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and calls["n"] < 3:
        time.sleep(0.01)
    stop.set()
    thread.join(5.0)
    assert not thread.is_alive()
    assert calls["n"] >= 3


def test_only_the_lease_holder_reconciles(harness: Harness) -> None:
    _propose(harness, "feature_order_bug")
    assert harness.controller.reconcile().outcome is not ReconcileOutcome.NOT_LEADER
    rival = Controller(
        control=harness.control,
        evidence=harness.controller.evidence,
        releases=harness.controller.releases,
        trust=harness.controller.trust,
        controller_id="c2",
        lease_seconds=60.0,
    )
    assert rival.reconcile().outcome is ReconcileOutcome.NOT_LEADER
    assert rival.holds_lease is False
    # After the lease expires the rival takes over with a new fencing token.
    harness.clock.advance(61.0)
    assert rival.reconcile().outcome is not ReconcileOutcome.NOT_LEADER
    held = harness.control.lease_holder(LEASE_NAME)
    assert held is not None
    # The holder string is the configured id plus a per-process suffix (see Controller.holder).
    assert held.holder.startswith("c2:")
    assert held.holder == rival.holder
    assert held.token == 2


def test_two_controllers_from_the_same_settings_do_not_both_hold_the_lease(
    harness: Harness,
) -> None:
    """``APP_CONTROLLER_ID`` names a deployment, so two processes of it share the string.

    The lease has to separate them anyway: a second process (``docker compose up --scale
    controller=2``, or a rolling restart whose old process has not exited) must see a live lease
    held by somebody else and stand down.
    """
    _propose(harness, "feature_order_bug")
    twin = Controller(
        control=harness.control,
        evidence=harness.controller.evidence,
        releases=harness.controller.releases,
        trust=harness.controller.trust,
        controller_id=harness.controller.controller_id,
        lease_seconds=60.0,
    )
    assert harness.controller.reconcile().outcome is not ReconcileOutcome.NOT_LEADER
    assert twin.reconcile().outcome is ReconcileOutcome.NOT_LEADER
    assert twin.holds_lease is False
    assert harness.controller.reconcile().outcome is not ReconcileOutcome.NOT_LEADER


def test_a_controller_that_lost_its_lease_cannot_write(harness: Harness) -> None:
    record = _propose(harness, "feature_order_bug")
    harness.run(2)
    # Renew under the controller's own holder identity: this is the lease it is about to lose.
    lease = harness.control.acquire_lease(LEASE_NAME, harness.controller.holder, 60.0)
    assert lease is not None
    harness.clock.advance(61.0)
    stolen = harness.control.acquire_lease(LEASE_NAME, "c2", 600.0)
    assert stolen is not None
    assert stolen.token > lease.token

    from progressive_model_delivery.control.store import Lease, LeaseLostError

    def stale_lease(name: str, holder: str, ttl_seconds: float) -> Lease | None:
        """Hand the controller the lease it already lost, as a crashed peer would see it."""
        del name, holder, ttl_seconds
        return lease

    harness.control.acquire_lease = stale_lease  # type: ignore[method-assign]

    def until_decision() -> None:
        for _ in range(12):
            harness.traffic.tick()
            harness.controller.reconcile()

    with pytest.raises(LeaseLostError):
        until_decision()
    assert harness.control.rollout(record.rollout_id).status == "active"


def test_a_concurrent_write_makes_the_pass_stale_instead_of_overwriting_it(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _propose(harness, "feature_order_bug")
    harness.run(2)
    original = ControlStore.rollout
    calls = {"n": 0}

    def racing(self: ControlStore, rollout_id: str, conn: object | None = None) -> RolloutRecord:
        stored = original(self, rollout_id, conn)  # type: ignore[arg-type]
        calls["n"] += 1
        return stored.model_copy(update={"revision": stored.revision + 1})

    monkeypatch.setattr(ControlStore, "rollout", racing)
    outcomes = []
    for _ in range(12):
        harness.traffic.tick()
        outcomes.append(harness.controller.reconcile().outcome)
        if outcomes[-1] is ReconcileOutcome.STALE:
            break
    monkeypatch.undo()
    assert outcomes[-1] is ReconcileOutcome.STALE
    assert harness.control.rollout(record.rollout_id).status == "active"


def test_the_controller_refuses_a_challenger_whose_signature_stopped_verifying(
    harness: Harness, tmp_path: Path
) -> None:
    """Tampering happens on a copy of the registry, so no other test sees a broken release."""
    from progressive_model_delivery.registry.release import (
        ReleaseVerificationError,
    )

    _propose(harness, "feature_order_bug")
    harness.run(2)
    copied = tmp_path / "tampered-registry"
    shutil.copytree(harness.settings.registry_dir, copied)
    release_file = copied / "releases" / f"{harness.env.release('feature_order_bug')}.json"
    release_file.write_text(
        release_file.read_text(encoding="utf-8").replace(
            '"release": "feature_order_bug"', '"release": "feature_order_bugg"'
        ),
        encoding="utf-8",
    )
    harness.controller.releases = ReleaseStore(copied)
    with pytest.raises(ReleaseVerificationError):
        harness.controller.reconcile()


def test_metrics_report_drift_statistics_from_the_live_traffic(harness: Harness) -> None:
    _propose(harness, "better")
    harness.run(4)
    report = harness.controller.last_report
    assert report is not None
    drift = sub(report.evidence, "drift")
    assert number(drift, "samples") > 0
    assert set(sub(drift, "feature_psi")) >= {"monthly_spend", "usage_trend", "region"}
    assert drift["drift_ok"] is True
    assert harness.metric("app_feature_psi", "usage_trend") >= 0.0


def test_a_covariate_shift_stops_the_rollout_from_advancing(
    control_env: ControlEnv, fast_spec: RolloutSpec
) -> None:
    with Harness(control_env, fast_spec, seed=3) as harness:
        harness.traffic = TrafficSimulator(
            harness.client,
            api_key=API_KEY,
            clock=harness.clock,
            config=TrafficConfig(
                seed=5,
                requests_per_tick=250,
                tick_seconds=30.0,
                covariate_shift=1.6,
                delay=LabelDelay(
                    mean_negative_seconds=5.0, mean_positive_seconds=10.0, loss_rate=0.02
                ),
            ),
            flush=harness.flush,
        )
        _propose(harness, "identical")
        outcomes = harness.run(8)
        assert ReconcileOutcome.ADVANCED not in outcomes
        report = harness.controller.last_report
        assert report is not None
        drift = sub(report.evidence, "drift")
        assert drift["drift_ok"] is False
        psi = number(drift, "feature_psi", "usage_trend")
        assert psi > fast_spec.drift.feature_psi_threshold
