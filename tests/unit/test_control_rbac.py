"""Role-based control: who may propose, approve, roll back and override, and who may not.

The three security properties under test:

* **Separation of duties.** The release manager who approves a promotion is never the deployer who
  proposed it, and a deployer cannot approve at all.
* **Two-person rule for overrides.** Forcing a stage advance past its statistical gate executes
  only after two *distinct* release managers have signed the same request, and the signatures
  expire if the rollout moved on in between.
* **Rollback is never gated.** Both roles may roll back at any time, because refusing a rollback
  is the dangerous failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from progressive_model_delivery.control.operations import advance
from progressive_model_delivery.control.rbac import (
    ControlService,
    PolicyLimits,
    Principal,
    Role,
    RoleDirectory,
)
from progressive_model_delivery.control.spec import RolloutSpec, bundled_spec
from progressive_model_delivery.control.store import ControlStore, RolloutRecord
from progressive_model_delivery.core.errors import AppError
from progressive_model_delivery.core.security import hash_api_key
from progressive_model_delivery.models.pipeline import DemoRegistry
from progressive_model_delivery.registry.release import ReleaseStore
from progressive_model_delivery.registry.signing import SigningKey, TrustStore

pytestmark = pytest.mark.unit

DANA = Principal("dana", frozenset({Role.DEPLOYER}))
RAVI = Principal("ravi", frozenset({Role.RELEASE_MANAGER}))
RITA = Principal("rita", frozenset({Role.RELEASE_MANAGER}))
VIC = Principal("vic", frozenset({Role.VIEWER}))
BOTH = Principal("both", frozenset({Role.DEPLOYER, Role.RELEASE_MANAGER}))


class Clock:
    def __init__(self) -> None:
        self.now = 5_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def spec() -> RolloutSpec:
    return bundled_spec("demo")


@pytest.fixture
def service(
    tmp_path: Path, trained_registry: DemoRegistry
) -> tuple[ControlService, ControlStore, DemoRegistry]:
    store = ControlStore(tmp_path / "control.sqlite", clock=Clock())
    releases = ReleaseStore(trained_registry.root / "registry")
    service = ControlService(store, releases, trained_registry.trust)
    service.bootstrap(RAVI, trained_registry.models.release_id("champion"))
    return service, store, trained_registry


def _start(
    service: ControlService, registry: DemoRegistry, spec: RolloutSpec, name: str = "better"
) -> RolloutRecord:
    return service.propose(DANA, registry.models.release_id(name), spec)


# --- authentication ---------------------------------------------------------------------------


def test_the_role_directory_maps_key_digests_to_principals_and_roles() -> None:
    key_a, key_b = "deployer-key-" + "a" * 24, "manager-key-" + "b" * 24
    directory = RoleDirectory(
        [
            f"{hash_api_key(key_a)}:dana:deployer",
            f"{hash_api_key(key_b)}:ravi:release_manager+viewer",
        ]
    )
    assert sorted(directory.digests) == sorted([hash_api_key(key_a), hash_api_key(key_b)])
    dana = directory.authenticate(key_a)
    assert dana is not None
    assert dana.name == "dana"
    assert dana.roles == {Role.DEPLOYER}
    ravi = directory.authenticate(key_b)
    assert ravi is not None
    assert ravi.roles == {Role.RELEASE_MANAGER, Role.VIEWER}
    assert directory.authenticate("wrong-key") is None
    assert directory.authenticate(None) is None
    assert directory.authenticate("") is None
    assert directory.authenticate("x" * 513) is None


def test_principal_role_checks() -> None:
    assert BOTH.has(Role.DEPLOYER)
    assert BOTH.has(Role.RELEASE_MANAGER)
    assert not VIC.has(Role.DEPLOYER)
    assert VIC.has(Role.VIEWER, Role.DEPLOYER)


# --- authorisation ----------------------------------------------------------------------------


def test_only_a_release_manager_bootstraps_and_only_a_deployer_proposes(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, _, registry = service
    with pytest.raises(AppError) as forbidden:
        control.bootstrap(DANA, registry.models.release_id("better"))
    assert forbidden.value.status_code == 403
    assert "release_manager" in forbidden.value.message
    with pytest.raises(AppError, match="deployer"):
        control.propose(RAVI, registry.models.release_id("better"), spec)
    with pytest.raises(AppError, match="deployer"):
        control.propose(VIC, registry.models.release_id("better"), spec)
    assert _start(control, registry, spec).status == "active"


def test_an_unverifiable_release_is_refused_before_anything_is_written(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec, tmp_path: Path
) -> None:
    _, store, registry = service
    stranger = TrustStore([SigningKey.generate().public()])
    distrusting = ControlService(store, ReleaseStore(registry.root / "registry"), stranger)
    with pytest.raises(AppError) as error:
        distrusting.propose(DANA, registry.models.release_id("better"), spec)
    assert error.value.status_code == 422
    assert error.value.code == "untrusted_release"
    assert store.open_rollout() is None


def test_a_second_bootstrap_and_a_second_rollout_are_conflicts(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, _, registry = service
    with pytest.raises(AppError) as conflict:
        control.bootstrap(RAVI, registry.models.release_id("better"))
    assert conflict.value.status_code == 409
    _start(control, registry, spec)
    with pytest.raises(AppError) as second:
        control.propose(DANA, registry.models.release_id("worse"), spec)
    assert second.value.status_code == 409


def test_an_unknown_rollout_is_a_404(
    service: tuple[ControlService, ControlStore, DemoRegistry],
) -> None:
    control, _, _ = service
    with pytest.raises(AppError) as error:
        control.rollback(DANA, "f" * 32, "why")
    assert error.value.status_code == 404


# --- promotion approval -----------------------------------------------------------------------


def _awaiting(
    control: ControlService, store: ControlStore, registry: DemoRegistry, spec: RolloutSpec
) -> RolloutRecord:
    record = _start(control, registry, spec)
    for _ in range(len(spec.stages)):
        with store.transaction() as conn:
            record = advance(store, conn, record, actor="controller:c1", evidence={"tests": ["ok"]})
    assert record.status == "awaiting_approval"
    return record


def test_the_proposer_cannot_approve_and_a_deployer_cannot_approve_at_all(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, store, registry = service
    record = _awaiting(control, store, registry, spec)
    with pytest.raises(AppError, match="release_manager"):
        control.approve_promotion(DANA, record.rollout_id)
    self_approving = ControlService(store, control.releases, control.trust)
    with pytest.raises(AppError) as forbidden:
        self_approving.approve_promotion(
            Principal("dana", frozenset({Role.RELEASE_MANAGER})), record.rollout_id
        )
    assert forbidden.value.status_code == 403
    assert "proposer" in forbidden.value.message
    promoted = control.approve_promotion(RAVI, record.rollout_id)
    assert promoted.status == "promoted"
    assert store.champion() == registry.models.release_id("better")


def test_approval_carries_the_controller_evidence_into_the_audit_log(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, store, registry = service
    record = _awaiting(control, store, registry, spec)
    control.approve_promotion(RAVI, record.rollout_id)
    entry = store.audit_entries(record.rollout_id)[-1]
    assert entry.action == "promoted"
    evidence = entry.data["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["evidence"] == {"tests": ["ok"]}
    assert isinstance(evidence["promotion_request_seq"], int)


def test_a_rollout_that_is_not_awaiting_approval_cannot_be_promoted(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, _, registry = service
    record = _start(control, registry, spec)
    with pytest.raises(AppError) as conflict:
        control.approve_promotion(RAVI, record.rollout_id)
    assert conflict.value.status_code == 409
    assert "not awaiting approval" in conflict.value.message


def test_an_approval_against_stale_evidence_is_refused(
    service: tuple[ControlService, ControlStore, DemoRegistry],
    spec: RolloutSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller writing between the read and the transaction must void the approval."""
    control, store, registry = service
    record = _awaiting(control, store, registry, spec)
    original = ControlStore.rollout
    calls = {"n": 0}

    def racing(self: ControlStore, rollout_id: str, conn: object | None = None) -> RolloutRecord:
        stored = original(self, rollout_id, conn)  # type: ignore[arg-type]
        calls["n"] += 1
        if calls["n"] == 2:  # the re-read inside the transaction sees a newer revision
            return stored.model_copy(update={"revision": stored.revision + 1})
        return stored

    monkeypatch.setattr(ControlStore, "rollout", racing)
    with pytest.raises(AppError, match="review the new evidence"):
        control.approve_promotion(RAVI, record.rollout_id)
    monkeypatch.undo()
    assert store.rollout(record.rollout_id).status == "awaiting_approval"


# --- rollback ---------------------------------------------------------------------------------


@pytest.mark.parametrize("actor", [DANA, RAVI, BOTH])
def test_both_roles_may_always_roll_back(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec, actor: Principal
) -> None:
    control, store, registry = service
    record = _start(control, registry, spec)
    rolled = control.rollback(actor, record.rollout_id, "pager went off")
    assert rolled.status == "rolled_back"
    assert rolled.decision_reason == "manual: pager went off"
    assert store.champion() == registry.models.release_id("champion")


def test_a_viewer_cannot_roll_back(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, _, registry = service
    record = _start(control, registry, spec)
    with pytest.raises(AppError, match="deployer"):
        control.rollback(VIC, record.rollout_id, "nope")


def test_rolling_a_closed_rollout_back_twice_is_a_conflict(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, _, registry = service
    record = _start(control, registry, spec)
    control.rollback(DANA, record.rollout_id, "first")
    with pytest.raises(AppError) as conflict:
        control.rollback(DANA, record.rollout_id, "second")
    assert conflict.value.status_code == 409


# --- overrides (two-person rule) ---------------------------------------------------------------


def test_an_override_needs_two_distinct_release_managers(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, store, registry = service
    record = _start(control, registry, spec)
    requested = control.request_override(DANA, record.rollout_id, "force_advance", "launch window")
    assert requested["status"] == "pending"
    assert requested["approvals"] == []
    request_id = str(requested["request_id"])

    first = control.approve_override(RAVI, request_id)
    assert first["status"] == "pending"
    assert first["approvals"] == ["ravi"]
    assert store.rollout(record.rollout_id).stage_index == 0  # nothing happened yet

    with pytest.raises(AppError, match="cannot sign an override twice"):
        control.approve_override(RAVI, request_id)

    second = control.approve_override(RITA, request_id)
    assert second["status"] == "executed"
    assert second["approvals"] == ["ravi", "rita"]
    advanced = store.rollout(record.rollout_id)
    assert advanced.stage_index == 1
    entry = next(e for e in store.audit_entries(record.rollout_id) if e.action == "stage_advanced")
    assert entry.actor == "override:ravi+rita"
    evidence = entry.data["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["override"] is True
    assert evidence["approved_by"] == ["ravi", "rita"]


def test_a_release_manager_requesting_an_override_counts_as_the_first_signature(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, store, registry = service
    record = _start(control, registry, spec)
    requested = control.request_override(RAVI, record.rollout_id, "force_advance", "reason")
    assert requested["approvals"] == ["ravi"]
    executed = control.approve_override(RITA, str(requested["request_id"]))
    assert executed["status"] == "executed"
    assert store.rollout(record.rollout_id).stage_index == 1


def test_an_override_expires_when_the_rollout_moves_on(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, store, registry = service
    record = _start(control, registry, spec)
    requested = control.request_override(DANA, record.rollout_id, "force_advance", "reason")
    with store.transaction() as conn:  # the controller advanced in the meantime
        advance(store, conn, record, actor="controller:c1", evidence={})
    first = control.approve_override(RAVI, str(requested["request_id"]))
    assert first["status"] == "expired"
    stored = store.approval(str(requested["request_id"]))
    assert stored is not None
    assert stored["status"] == "expired"
    with pytest.raises(AppError, match="Override is expired"):
        control.approve_override(RITA, str(requested["request_id"]))


def test_override_requests_are_validated_and_role_checked(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, _, registry = service
    record = _start(control, registry, spec)
    with pytest.raises(AppError) as unknown:
        control.request_override(DANA, record.rollout_id, "delete_everything", "why")
    assert unknown.value.code == "unknown_override"
    with pytest.raises(AppError, match="deployer"):
        control.request_override(VIC, record.rollout_id, "force_advance", "why")
    with pytest.raises(AppError, match="release_manager"):
        control.approve_override(DANA, "does-not-matter")
    with pytest.raises(AppError) as missing:
        control.approve_override(RAVI, "unknown-request")
    assert missing.value.status_code == 404


def test_an_override_cannot_be_requested_for_a_closed_rollout(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, _, registry = service
    record = _start(control, registry, spec)
    control.rollback(DANA, record.rollout_id, "done")
    with pytest.raises(AppError) as conflict:
        control.request_override(DANA, record.rollout_id, "force_advance", "why")
    assert conflict.value.status_code == 409


def test_a_promotion_approval_request_id_is_not_an_override(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, store, registry = service
    record = _start(control, registry, spec)
    with store.transaction() as conn:
        store.put_approval(
            conn,
            request_id="not-an-override",
            rollout_id=record.rollout_id,
            kind="promotion",
            record={},
            status="pending",
        )
    with pytest.raises(AppError) as error:
        control.approve_override(RAVI, "not-an-override")
    assert error.value.status_code == 404


def test_evidence_lookup_is_empty_when_the_controller_never_asked(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, _, registry = service
    record = _start(control, registry, spec)
    assert control._latest_evidence(record.rollout_id) == {}


def test_the_audit_chain_survives_every_rbac_path(
    service: tuple[ControlService, ControlStore, DemoRegistry], spec: RolloutSpec
) -> None:
    control, store, registry = service
    record = _start(control, registry, spec)
    requested = control.request_override(DANA, record.rollout_id, "force_advance", "reason")
    control.approve_override(RAVI, str(requested["request_id"]))
    control.approve_override(RITA, str(requested["request_id"]))
    control.rollback(RAVI, record.rollout_id, "changed our mind")
    verification = store.verify_audit()
    assert verification.ok
    actions: list[Any] = [entry.action for entry in store.audit_entries()]
    assert actions == [
        "champion_bootstrapped",
        "rollout_proposed",
        "override_requested",
        "override_signed",
        "stage_advanced",
        "override_executed",
        "rolled_back",
    ]


# --- policy floors ----------------------------------------------------------------------------


def test_the_bundled_specs_clear_the_shipped_policy_floors() -> None:
    """The floors in ``.env.example`` and ``docker-compose.yml`` accept the policies we ship."""
    limits = PolicyLimits(
        max_alpha_rollback=0.05,
        max_alpha_promotion=0.05,
        required_guardrails=("error_rate", "worst_subgroup_error", "latency_slo", "log_loss"),
        min_label_window_seconds=120.0,
    )
    for name in ("standard", "demo"):
        assert limits.refusal(bundled_spec(name)) is None, name


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        (PolicyLimits(max_alpha_rollback=0.01), "alpha_rollback"),
        (PolicyLimits(max_alpha_promotion=0.01), "alpha_promotion"),
        (PolicyLimits(required_guardrails=("calibration",)), "calibration"),
        (PolicyLimits(min_label_window_seconds=3_600.0), "label_window_seconds"),
    ],
)
def test_a_spec_that_breaches_a_floor_cannot_open_a_rollout(
    service: tuple[ControlService, ControlStore, DemoRegistry],
    spec: RolloutSpec,
    limits: PolicyLimits,
    expected: str,
) -> None:
    """A weak spec is the one thing the two-person rules do not cover, so the floors refuse it."""
    control, store, registry = service
    control.limits = limits
    refusal = limits.refusal(spec)
    assert refusal is not None
    assert expected in refusal
    with pytest.raises(AppError) as raised:
        _start(control, registry, spec)
    assert raised.value.status_code == 422
    assert raised.value.code == "spec_refused"
    assert store.open_rollout() is None
    assert [entry.action for entry in store.audit_entries()] == ["champion_bootstrapped"]
