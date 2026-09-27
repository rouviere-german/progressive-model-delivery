"""The control API: every endpoint behind an API key bound to a role.

Authentication is separate from authorisation here: a valid key identifies a principal, and the
principal's roles decide what it may do. An unknown key never learns which endpoints exist, and
the audit endpoints expose the chain so an operator can verify it without database access.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from progressive_model_delivery.control.api import build_control_app
from progressive_model_delivery.control.controller import Controller, ReconcileOutcome
from progressive_model_delivery.control.operations import advance
from progressive_model_delivery.control.spec import bundled_spec
from progressive_model_delivery.control.store import ControlStore
from progressive_model_delivery.core.security import hash_api_key
from tests.integration.conftest import ControlEnv

pytestmark = pytest.mark.integration

KEYS = {
    "dana": "deployer-key-" + "d" * 24,
    "ravi": "manager-key-" + "r" * 24,
    "rita": "manager2-key-" + "t" * 24,
    "vic": "viewer-key-" + "v" * 24,
}
ROLES = {"dana": "deployer", "ravi": "release_manager", "rita": "release_manager", "vic": "viewer"}


def _auth(who: str) -> dict[str, str]:
    return {"X-API-Key": KEYS[who]}


@pytest.fixture
def api(control_env: ControlEnv) -> Iterator[TestClient]:
    """A control app over an *empty* control plane (no champion yet)."""
    fresh = control_env.tmp_path / "api-control.sqlite"
    settings = control_env.settings(
        control_db=fresh,
        role_bindings=[f"{hash_api_key(KEYS[name])}:{name}:{ROLES[name]}" for name in KEYS],
    )
    app = build_control_app(
        settings, clock=lambda: control_env.clock[0], background=False
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _control(app: FastAPI) -> ControlStore:
    controller = app.state.controller
    assert isinstance(controller, Controller)
    return controller.control


def _bootstrap(api: TestClient, control_env: ControlEnv) -> None:
    response = api.post(
        "/v1/champion", json={"release_id": control_env.release("champion")}, headers=_auth("ravi")
    )
    assert response.status_code == 201, response.text


def _propose(api: TestClient, control_env: ControlEnv, name: str = "better") -> dict[str, Any]:
    response = api.post(
        "/v1/rollouts",
        json={"challenger": control_env.release(name), "spec": "demo"},
        headers=_auth("dana"),
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


# --- authentication ----------------------------------------------------------------------------


def test_every_endpoint_needs_a_known_key(api: TestClient, control_env: ControlEnv) -> None:
    for method, path in [
        ("post", "/v1/champion"),
        ("post", "/v1/rollouts"),
        ("get", "/v1/rollouts"),
        ("get", "/v1/rollouts/current"),
        ("get", "/v1/rollouts/current/evidence"),
        ("post", f"/v1/rollouts/{'a' * 32}/approve"),
        ("post", f"/v1/rollouts/{'a' * 32}/rollback"),
        ("post", f"/v1/rollouts/{'a' * 32}/overrides"),
        ("post", "/v1/overrides/abc/approve"),
        ("get", "/v1/audit"),
        ("get", "/v1/audit/verify"),
    ]:
        kwargs: dict[str, Any] = {"json": {}} if method == "post" else {}
        response = getattr(api, method)(path, **kwargs)
        assert response.status_code == 401, (method, path, response.text)
        assert response.headers["WWW-Authenticate"] == "ApiKey"
        assert response.json()["error"]["code"] == "unauthorized"


def test_a_valid_key_with_the_wrong_role_is_forbidden_not_unauthorized(
    api: TestClient, control_env: ControlEnv
) -> None:
    response = api.post(
        "/v1/champion", json={"release_id": control_env.release("champion")}, headers=_auth("vic")
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_health_and_readiness_track_the_controller_lease(api: TestClient) -> None:
    assert api.get("/healthz").status_code == 200
    ready = api.get("/readyz")
    assert ready.status_code == 503  # the background loop is off, so no lease is held
    assert ready.json()["checks"]["lease_held"] == "fail"
    controller = api.app.state.controller  # type: ignore[attr-defined]
    assert isinstance(controller, Controller)
    assert controller.reconcile().outcome is ReconcileOutcome.IDLE
    assert api.get("/readyz").status_code == 200


# --- the happy path ----------------------------------------------------------------------------


def test_bootstrap_propose_list_and_read_the_current_rollout(
    api: TestClient, control_env: ControlEnv
) -> None:
    assert api.get("/v1/rollouts/current", headers=_auth("vic")).status_code == 404
    _bootstrap(api, control_env)
    record = _propose(api, control_env)
    assert record["status"] == "active"
    assert record["spec"]["name"] == "churn-risk-demo"
    assert record["spec_sha256"] == bundled_spec("demo").sha256()

    listed = api.get("/v1/rollouts", headers=_auth("vic"))
    assert listed.status_code == 200
    assert [item["rollout_id"] for item in listed.json()] == [record["rollout_id"]]
    current = api.get("/v1/rollouts/current", headers=_auth("vic"))
    assert current.json()["rollout_id"] == record["rollout_id"]


def test_a_rollout_can_be_proposed_with_inline_spec_yaml(
    api: TestClient, control_env: ControlEnv
) -> None:
    _bootstrap(api, control_env)
    yaml_spec = """
name: inline-policy
version: 3
label_window_seconds: 60
stages:
  - kind: shadow
    min_samples: 10
    min_duration_seconds: 0
  - kind: canary
    weight_percent: 5
    min_samples: 10
    min_duration_seconds: 0
guardrails:
  - name: error_rate
    metric: error
    non_inferiority_ratio: 1.2
"""
    response = api.post(
        "/v1/rollouts",
        json={"challenger": control_env.release("better"), "spec_yaml": yaml_spec},
        headers=_auth("dana"),
    )
    assert response.status_code == 201, response.text
    assert response.json()["spec"]["name"] == "inline-policy"
    assert response.json()["spec"]["version"] == 3


@pytest.mark.parametrize(
    "body",
    [
        {"challenger": "b" * 64},  # neither spec nor spec_yaml
        {"challenger": "b" * 64, "spec": "demo", "spec_yaml": "name: x"},  # both
        {"challenger": "b" * 64, "spec_yaml": "not: a spec"},
        {"challenger": "b" * 64, "spec_yaml": "- 1\n- 2\n"},
    ],
)
def test_a_bad_spec_is_a_422(api: TestClient, control_env: ControlEnv, body: dict[str, Any]) -> None:
    _bootstrap(api, control_env)
    response = api.post("/v1/rollouts", json=body, headers=_auth("dana"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] in {"validation_error", "invalid_spec"}


def test_an_unknown_release_is_refused_with_a_stable_code(
    api: TestClient, control_env: ControlEnv
) -> None:
    _bootstrap(api, control_env)
    response = api.post(
        "/v1/rollouts", json={"challenger": "f" * 64, "spec": "demo"}, headers=_auth("dana")
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "untrusted_release"


def test_the_evidence_endpoint_reports_the_controllers_latest_pass(
    api: TestClient, control_env: ControlEnv
) -> None:
    assert api.get("/v1/rollouts/current/evidence", headers=_auth("vic")).status_code == 404
    _bootstrap(api, control_env)
    record = _propose(api, control_env)
    controller = api.app.state.controller  # type: ignore[attr-defined]
    assert isinstance(controller, Controller)
    controller.reconcile()
    response = api.get("/v1/rollouts/current/evidence", headers=_auth("vic"))
    assert response.status_code == 200
    body = response.json()
    assert body["rollout_id"] == record["rollout_id"]
    assert body["outcome"] == "hold"
    assert body["evidence"]["spec_sha256"] == record["spec_sha256"]
    assert isinstance(body["evidence"]["tests"], list)


def test_promotion_needs_a_release_manager_who_is_not_the_proposer(
    api: TestClient, control_env: ControlEnv
) -> None:
    _bootstrap(api, control_env)
    record = _propose(api, control_env)
    rollout_id = record["rollout_id"]
    control = _control(api.app)  # type: ignore[arg-type]

    assert api.post(f"/v1/rollouts/{rollout_id}/approve", headers=_auth("ravi")).status_code == 409
    stored = control.rollout(rollout_id)
    for _ in range(len(stored.spec.stages)):
        with control.transaction() as conn:
            stored = advance(control, conn, stored, actor="controller:c1", evidence={"ok": True})
    assert stored.status == "awaiting_approval"

    assert api.post(f"/v1/rollouts/{rollout_id}/approve", headers=_auth("dana")).status_code == 403
    assert api.post(f"/v1/rollouts/{rollout_id}/approve", headers=_auth("vic")).status_code == 403
    promoted = api.post(f"/v1/rollouts/{rollout_id}/approve", headers=_auth("ravi"))
    assert promoted.status_code == 200
    assert promoted.json()["status"] == "promoted"
    assert control.champion() == control_env.release("better")


def test_rollback_is_available_to_both_roles_and_needs_a_reason(
    api: TestClient, control_env: ControlEnv
) -> None:
    _bootstrap(api, control_env)
    record = _propose(api, control_env)
    rollout_id = record["rollout_id"]
    missing = api.post(f"/v1/rollouts/{rollout_id}/rollback", json={}, headers=_auth("dana"))
    assert missing.status_code == 422
    empty = api.post(
        f"/v1/rollouts/{rollout_id}/rollback", json={"reason": ""}, headers=_auth("dana")
    )
    assert empty.status_code == 422
    forbidden = api.post(
        f"/v1/rollouts/{rollout_id}/rollback", json={"reason": "no"}, headers=_auth("vic")
    )
    assert forbidden.status_code == 403
    done = api.post(
        f"/v1/rollouts/{rollout_id}/rollback", json={"reason": "pager"}, headers=_auth("dana")
    )
    assert done.status_code == 200
    assert done.json()["status"] == "rolled_back"
    assert done.json()["decision_reason"] == "manual: pager"


def test_an_override_takes_two_release_managers_over_the_api(
    api: TestClient, control_env: ControlEnv
) -> None:
    _bootstrap(api, control_env)
    record = _propose(api, control_env)
    rollout_id = record["rollout_id"]
    body = {"action": "force_advance", "reason": "launch window closes tonight"}
    requested = api.post(f"/v1/rollouts/{rollout_id}/overrides", json=body, headers=_auth("dana"))
    assert requested.status_code == 201
    request_id = requested.json()["request_id"]
    assert requested.json()["status"] == "pending"

    unknown_action = api.post(
        f"/v1/rollouts/{rollout_id}/overrides",
        json={"action": "delete", "reason": "x"},
        headers=_auth("dana"),
    )
    assert unknown_action.status_code == 422

    first = api.post(f"/v1/overrides/{request_id}/approve", headers=_auth("ravi"))
    assert first.status_code == 200 and first.json()["status"] == "pending"
    again = api.post(f"/v1/overrides/{request_id}/approve", headers=_auth("ravi"))
    assert again.status_code == 403
    assert api.post(f"/v1/overrides/{request_id}/approve", headers=_auth("dana")).status_code == 403
    second = api.post(f"/v1/overrides/{request_id}/approve", headers=_auth("rita"))
    assert second.status_code == 200
    assert second.json()["status"] == "executed"
    assert second.json()["approvals"] == ["ravi", "rita"]
    assert _control(api.app).rollout(rollout_id).stage_index == 1  # type: ignore[arg-type]

    missing = api.post("/v1/overrides/nope/approve", headers=_auth("ravi"))
    assert missing.status_code == 404


# --- audit -------------------------------------------------------------------------------------


def test_the_audit_chain_is_readable_and_verifiable_over_http(
    api: TestClient, control_env: ControlEnv
) -> None:
    _bootstrap(api, control_env)
    record = _propose(api, control_env)
    entries = api.get("/v1/audit", headers=_auth("vic"))
    assert entries.status_code == 200
    actions = [entry["action"] for entry in entries.json()]
    assert actions == ["champion_bootstrapped", "rollout_proposed"]
    scoped = api.get(
        "/v1/audit", params={"rollout_id": record["rollout_id"]}, headers=_auth("vic")
    )
    assert [entry["action"] for entry in scoped.json()] == ["rollout_proposed"]

    verified = api.get("/v1/audit/verify", headers=_auth("vic"))
    assert verified.json() == {
        "ok": True,
        "entries": 2,
        "head": entries.json()[-1]["hash"],
        "error": None,
    }

    control = _control(api.app)  # type: ignore[arg-type]
    with control.transaction() as conn:
        conn.execute("UPDATE audit SET actor = 'mallory' WHERE seq = 1")
    broken = api.get("/v1/audit/verify", headers=_auth("vic")).json()
    assert broken["ok"] is False
    assert broken["error"] == "entry content modified"


def test_unknown_rollout_ids_are_404_and_malformed_ones_are_422(
    api: TestClient, control_env: ControlEnv
) -> None:
    _bootstrap(api, control_env)
    assert (
        api.post(
            f"/v1/rollouts/{'a' * 32}/rollback", json={"reason": "x"}, headers=_auth("dana")
        ).status_code
        == 404
    )
    too_long = api.post("/v1/champion", json={"release_id": "a" * 63}, headers=_auth("ravi"))
    assert too_long.status_code == 422
