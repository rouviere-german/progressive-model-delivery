"""The offline end-to-end demo: two rollouts, one rolled back and one promoted.

``uv run python tasks.py demo`` (or ``python -m progressive_model_delivery demo``) trains the demo
models, signs and publishes them, and then runs two complete rollouts against a real router, a
real out-of-process-shaped shadow worker and the real reconciliation controller. Nothing is
mocked and nothing leaves the machine: the shadow hop is an in-process ASGI transport, the clock
is simulated, and no API key or network access is needed.

1. **A subgroup-regressed challenger.** ``subgroup_worse`` was trained with stale labels for the
   ``south`` region, so it looks fine globally and is clearly worse for 12% of customers. The
   worst-subgroup e-value crosses its rollback threshold during the shadow stage, before a single
   customer is served by it, and the rollback lands in the hash-chained audit log with the
   e-values that justified it.
2. **A better challenger.** ``better`` walks shadow -> 1% -> 5% -> 25% -> 50%, proves
   non-inferiority on the served arm, and then *waits*: the controller asks for promotion but
   only a release manager who is not the proposer can grant it. The demo shows the proposer being
   refused, then the release manager approving, and the champion changing.

The summary printed at the end is the machine-readable record of that run (stages reached,
e-values at the decision, exposure, audit head and verification).
"""

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from progressive_model_delivery.control.controller import Controller, ReconcileOutcome
from progressive_model_delivery.control.evidence import EvidenceReader
from progressive_model_delivery.control.rbac import ControlService, Principal, Role
from progressive_model_delivery.control.spec import RolloutSpec, bundled_spec
from progressive_model_delivery.control.store import ControlStore
from progressive_model_delivery.core.config import Settings
from progressive_model_delivery.core.errors import AppError
from progressive_model_delivery.core.security import hash_api_key
from progressive_model_delivery.models.pipeline import DemoModels, build_demo_models
from progressive_model_delivery.models.synthetic import LabelDelay
from progressive_model_delivery.registry.release import ReleaseStore
from progressive_model_delivery.registry.signing import SigningKey, TrustStore
from progressive_model_delivery.serving.router import RouterRuntime, build_router_app
from progressive_model_delivery.serving.shadow_worker import ShadowRuntime, build_shadow_app
from progressive_model_delivery.sim.traffic import SimulatedClock, TrafficConfig, TrafficSimulator

__all__ = ["DemoEnvironment", "run_demo"]

API_KEY = "demo-api-key-" + "d" * 24
SUBJECT_SECRET = "demo-subject-secret-" + "s" * 16
SHADOW_TOKEN = "demo-shadow-token-" + "t" * 16
DEPLOYER = Principal("dana", frozenset({Role.DEPLOYER}))
RELEASE_MANAGER = Principal("ravi", frozenset({Role.RELEASE_MANAGER}))
MAX_TICKS = 120
_TERMINAL = (
    ReconcileOutcome.ROLLED_BACK,
    ReconcileOutcome.PROMOTION_REQUESTED,
    ReconcileOutcome.ABORTED,
)


class DemoExpectationError(AssertionError):
    """A demo scenario did not reach the outcome this repository claims for it."""


def _expect(outcome: ReconcileOutcome, wanted: ReconcileOutcome, scenario: str) -> None:
    if outcome is not wanted:
        raise DemoExpectationError(
            f"{scenario}: expected the controller to report {wanted}, got {outcome}"
        )


@dataclass(frozen=True, slots=True)
class DemoEnvironment:
    """Everything one demo run needs: signed releases, a control plane and a simulated clock."""

    workdir: Path
    settings: Settings
    clock: SimulatedClock
    models: DemoModels
    control: ControlStore
    service: ControlService
    controller: Controller
    spec: RolloutSpec

    def release(self, name: str) -> str:
        return self.models.release_id(name)


def _settings(workdir: Path, key: SigningKey, spec: RolloutSpec) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        log_level="WARNING",
        api_key_sha256=[hash_api_key(API_KEY)],
        registry_dir=workdir / "models" / "registry",
        control_db=workdir / "control.sqlite",
        predlog_dir=workdir / "predlog",
        trusted_signing_keys=[key.public().to_b64()],
        subject_hash_secret=SUBJECT_SECRET,
        shadow_url="http://shadow.internal:8001",
        shadow_token=SHADOW_TOKEN,
        shadow_queue_size=4_096,
        # Labels are refused once the maturation window has passed, so a slow label is a lost one.
        label_ttl_seconds=spec.label_window_seconds,
        rate_limit_per_second=1e6,
        rate_limit_burst=1_000_000,
        feedback_rate_per_second=1e6,
        feedback_burst=1_000_000,
        controller_id="demo-controller",
        lease_seconds=3_600.0,
    )


def build_environment(workdir: Path, *, spec_name: str = "demo") -> DemoEnvironment:
    """Train and publish the demo releases, then wire an empty control plane around them."""
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    spec = bundled_spec(spec_name)
    key = SigningKey.generate()
    models = build_demo_models(
        workdir / "models",
        key,
        n_train=20_000,
        n_holdout=8_000,
        names=("champion", "subgroup_worse", "better"),
    )
    settings = _settings(workdir, key, spec)
    clock = SimulatedClock()
    control = ControlStore(settings.control_db, clock=clock)
    releases = ReleaseStore(settings.registry_dir)
    trust = TrustStore.from_b64(settings.trusted_signing_keys)
    return DemoEnvironment(
        workdir=workdir,
        settings=settings,
        clock=clock,
        models=models,
        control=control,
        service=ControlService(control, releases, trust),
        controller=Controller(
            control=control,
            evidence=EvidenceReader(settings.predlog_dir),
            releases=releases,
            trust=trust,
            controller_id=settings.controller_id,
            lease_seconds=settings.lease_seconds,
        ),
        spec=spec,
    )


@contextmanager
def _serving(
    env: DemoEnvironment,
) -> Iterator[tuple[TestClient, Callable[[], None], Callable[[], None]]]:
    """A router with the shadow worker behind an in-process ASGI transport."""
    shadow_app = build_shadow_app(env.settings, background=False)
    shadow_runtime = shadow_app.state.runtime
    if not isinstance(shadow_runtime, ShadowRuntime):  # pragma: no cover - wiring guard
        raise TypeError("shadow app did not expose its runtime")
    router_app = build_router_app(
        env.settings,
        clock=env.clock,
        shadow_transport=httpx.ASGITransport(app=shadow_app),
        background=False,
    )
    with ExitStack() as stack:
        client = stack.enter_context(TestClient(router_app))
        router_runtime = client.app.state.runtime  # type: ignore[attr-defined]
        if not isinstance(router_runtime, RouterRuntime):  # pragma: no cover - wiring guard
            raise TypeError("router app did not expose its runtime")

        def flush() -> None:
            _drain_shadow(client, router_runtime)
            router_runtime.flush()
            shadow_runtime.writer.flush()

        def refresh() -> None:
            router_runtime.refresh()

        yield client, flush, refresh


def _metric(client: TestClient, name: str, *contains: str) -> float:
    """Sum every exported sample of ``name`` whose label set contains all of ``contains``."""
    total = 0.0
    for line in client.get("/metrics", headers={"X-API-Key": API_KEY}).text.splitlines():
        if not line.startswith((name + " ", name + "{")):
            continue
        if all(needle in line for needle in contains):
            total += float(line.rsplit(" ", 1)[1])
    return total


def _drain_shadow(client: TestClient, runtime: RouterRuntime, timeout: float = 10.0) -> None:
    """Wait until every offered shadow job has been delivered, failed or dropped."""
    if runtime.forwarder is None:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        enqueued = _metric(client, "app_shadow_enqueued_total")
        settled = _metric(client, "app_shadow_delivered_total") + _metric(
            client, "app_shadow_failed_total"
        )
        if settled >= enqueued:
            return
        time.sleep(0.01)


def _reconcile_until_decision(
    env: DemoEnvironment,
    traffic: TrafficSimulator,
    refresh: Callable[[], None],
    *,
    max_ticks: int = MAX_TICKS,
) -> tuple[ReconcileOutcome, dict[str, Any], list[dict[str, Any]]]:
    """Alternate traffic ticks and reconciliation passes until the rollout is decided.

    ``refresh`` re-reads the serving state, which is what the router's background loop does every
    ``state_refresh_seconds`` in production; doing it explicitly keeps the demo deterministic.
    """
    timeline: list[dict[str, Any]] = []
    outcome = ReconcileOutcome.IDLE
    evidence: dict[str, Any] = {}
    for _ in range(max_ticks):
        traffic.tick()
        report = env.controller.reconcile()
        outcome = report.outcome
        evidence = dict(report.evidence)
        if outcome is not ReconcileOutcome.HOLD:
            refresh()
            timeline.append(
                {
                    "tick": traffic.ticks,
                    "outcome": str(outcome),
                    "stage_index": report.stage_index,
                    "stage_samples": evidence.get("stage_samples"),
                    "canary_weight_percent": evidence.get("canary_weight_percent"),
                }
            )
        if outcome in _TERMINAL:
            return outcome, evidence, timeline
    return outcome, evidence, timeline


def _crossed(evidence: dict[str, Any], test: str) -> list[dict[str, Any]]:
    tests = evidence.get("tests")
    if not isinstance(tests, list):
        return []
    return [
        {
            "guardrail": str(item["guardrail"]),
            "log_e": item["log_e"],
            "threshold": item["threshold"],
        }
        for item in tests
        if isinstance(item, dict) and item.get("test") == test and item.get("crossed")
    ]


def _traffic(
    env: DemoEnvironment, client: TestClient, flush: Callable[[], None], *, seed: int
) -> TrafficSimulator:
    # Mean label delays well inside the 120 s maturation window, so most labels arrive in time;
    # positives still lag negatives and 3% never arrive, which is what the maturation rule is for.
    return TrafficSimulator(
        client,
        api_key=API_KEY,
        clock=env.clock,
        config=TrafficConfig(
            seed=seed,
            requests_per_tick=250,
            tick_seconds=90.0,
            delay=LabelDelay(
                mean_negative_seconds=20.0, mean_positive_seconds=45.0, loss_rate=0.03
            ),
        ),
        flush=flush,
    )


def salt_for(seed: int | None, challenger: str) -> str | None:
    """The assignment salt for one scenario, or ``None`` to draw a fresh random one.

    A rollout normally draws its own salt, which is why the canary numbers move between runs: a
    different partition of customers serves a different number of requests and produces different
    e-values at promotion. ``demo --seed`` derives the salt instead, so the whole run -- including
    the pasted README block -- is reproducible.
    """
    if seed is None:
        return None
    return hashlib.sha256(f"{seed}:{challenger}".encode()).hexdigest()[:32]


def _lifecycle_stage(env: DemoEnvironment, name: str) -> str:
    state = env.control.lifecycle(env.release(name))
    return "unknown" if state is None else str(state.stage)


def _rollback_scenario(env: DemoEnvironment, seed: int | None = None) -> dict[str, Any]:
    record = env.service.propose(
        DEPLOYER,
        env.release("subgroup_worse"),
        env.spec,
        salt=salt_for(seed, "subgroup_worse"),
    )
    with _serving(env) as (client, flush, refresh):
        traffic = _traffic(env, client, flush, seed=101)
        traffic.assert_ttl_fits(env.settings.label_ttl_seconds)
        outcome, evidence, timeline = _reconcile_until_decision(env, traffic, refresh)
        served = int(_metric(client, "app_predictions_total", 'arm="challenger"'))
    _expect(outcome, ReconcileOutcome.ROLLED_BACK, "subgroup-regressed challenger")
    final = env.control.rollout(record.rollout_id)
    return {
        "challenger": "subgroup_worse",
        "rollout_id": record.rollout_id,
        "outcome": str(outcome),
        "status": final.status,
        "decision_reason": final.decision_reason,
        "stage_reached": final.stage_index,
        "stages": [window.kind.value for window in final.stages],
        "requests": traffic.total.requests,
        "labels_accepted": traffic.total.labels_accepted,
        "labels_refused_after_window": traffic.total.labels_refused,
        "challenger_served_requests": served,
        "harm_tests_crossed": _crossed(evidence, "harm"),
        "timeline": timeline,
        "lifecycle": _lifecycle_stage(env, "subgroup_worse"),
    }


def _promotion_scenario(env: DemoEnvironment, seed: int | None = None) -> dict[str, Any]:
    record = env.service.propose(
        DEPLOYER, env.release("better"), env.spec, salt=salt_for(seed, "better")
    )
    with _serving(env) as (client, flush, refresh):
        traffic = _traffic(env, client, flush, seed=202)
        traffic.assert_ttl_fits(env.settings.label_ttl_seconds)
        outcome, evidence, timeline = _reconcile_until_decision(env, traffic, refresh)
        served = int(_metric(client, "app_predictions_total", 'arm="challenger"'))
    _expect(outcome, ReconcileOutcome.PROMOTION_REQUESTED, "better challenger")
    refusals: list[str] = []
    try:
        env.service.approve_promotion(DEPLOYER, record.rollout_id)
    except AppError as exc:
        refusals.append(f"{exc.code}: the proposer cannot approve their own rollout")
    else:  # pragma: no cover - the two-person rule is tested in tests/unit/test_control_rbac.py
        raise DemoExpectationError("the proposer was allowed to approve their own rollout")
    status = env.service.approve_promotion(RELEASE_MANAGER, record.rollout_id).status
    final = env.control.rollout(record.rollout_id)
    return {
        "challenger": "better",
        "rollout_id": record.rollout_id,
        "outcome": str(outcome),
        "status": status,
        "stage_reached": final.stage_index,
        "canary_weights": [w.weight_bp / 100 for w in final.stages if w.weight_bp],
        "requests": traffic.total.requests,
        "labels_accepted": traffic.total.labels_accepted,
        "challenger_served_requests": served,
        "non_inferiority_crossed": _crossed(evidence, "non_inferiority"),
        "decision_reason": final.decision_reason,
        "harm_tests_crossed": _crossed(evidence, "harm"),
        "refused_approvals": refusals,
        "approved_by": RELEASE_MANAGER.name,
        "champion_after": env.control.champion(),
        "lifecycle": _lifecycle_stage(env, "better"),
        "timeline": timeline,
    }


def run_demo(workdir: Path, *, seed: int | None = None) -> dict[str, Any]:
    """Run both rollouts under ``workdir`` and return a JSON-serializable summary.

    Without ``seed`` each rollout draws a random assignment salt, as a deployment does: the
    outcomes are deterministic (and the tests assert them) while the canary magnitudes move. With
    ``seed`` the salts are derived from it and the whole summary is reproducible.
    """
    env = build_environment(workdir)
    env.service.bootstrap(RELEASE_MANAGER, env.release("champion"))
    champion = env.release("champion")
    rollback = _rollback_scenario(env, seed)
    promotion = _promotion_scenario(env, seed)
    audit_path = env.workdir / "audit.jsonl"
    entries = env.control.export_audit(audit_path)
    verification = env.control.verify_audit()
    return {
        "spec": {"name": env.spec.name, "version": env.spec.version, "sha256": env.spec.sha256()},
        "champion_before": champion,
        "champion_after": env.control.champion(),
        "champion_changed": env.control.champion() != champion,
        "rollback_scenario": rollback,
        "promotion_scenario": promotion,
        "seed": seed,
        "audit": {
            "entries": entries,
            "head": verification.head,
            "verified": verification.ok,
            "path": str(audit_path),
            "actions": [entry.action for entry in env.control.audit_entries()],
        },
        "simulated_hours": round((env.clock.now - SimulatedClock().now) / 3600.0, 2),
    }
