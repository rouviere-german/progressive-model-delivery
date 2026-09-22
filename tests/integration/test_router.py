from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from progressive_model_delivery.registry.signing import SigningKey
from progressive_model_delivery.serving.assignment import Arm, assign, subject_hash
from progressive_model_delivery.serving.router import RouterRuntime, build_router_app
from tests.integration.conftest import AUTH, FEATURES, SUBJECT_SECRET, ControlEnv

pytestmark = pytest.mark.integration


def _runtime(app: FastAPI) -> RouterRuntime:
    runtime = app.state.runtime
    assert isinstance(runtime, RouterRuntime)
    return runtime


@pytest.fixture
def router(control_env: ControlEnv) -> Iterator[TestClient]:
    app = build_router_app(
        control_env.settings(), clock=lambda: control_env.clock[0], background=False
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _predict(client: TestClient, subject: str, **features: object) -> dict[str, object]:
    response = client.post(
        "/v1/predict",
        json={"subject_id": subject, "features": {**FEATURES, **features}},
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def _rows(tmp_path: Path, table: str) -> list[dict[str, object]]:
    files = [str(p) for p in (tmp_path / "predlog" / table).glob("*.parquet")]
    if not files:
        return []
    with duckdb.connect() as con:
        frame = con.execute("SELECT * FROM read_parquet($f)", {"f": files}).df()
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]


def test_champion_only_serving_logs_hashed_subjects(
    router: TestClient, control_env: ControlEnv
) -> None:
    body = _predict(router, "cust-0000042")
    champion = control_env.release("champion")
    assert body["model_version"] == champion[:12]
    assert 0.0 <= float(str(body["score"])) <= 1.0
    _runtime(router.app).flush()  # type: ignore[arg-type]
    rows = _rows(control_env.tmp_path, "router")
    assert len(rows) == 1
    row = rows[0]
    assert row["arm"] == "champion"
    assert row["release_id"] == champion
    assert row["subject_hash"] == subject_hash(SUBJECT_SECRET.encode(), "cust-0000042")
    raw = b"".join(p.read_bytes() for p in (control_env.tmp_path / "predlog").rglob("*.parquet"))
    assert b"cust-0000042" not in raw


def test_readiness_needs_a_verified_champion(tmp_path: Path, control_env: ControlEnv) -> None:
    with TestClient(build_router_app(control_env.settings(), background=False)) as client:
        assert client.get("/readyz").status_code == 200
    stranger = SigningKey.generate().public().to_b64()
    untrusting = control_env.settings(trusted_signing_keys=[stranger])
    with TestClient(build_router_app(untrusting, background=False)) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["checks"] == {"champion_verified": "fail"}
        refused = client.post(
            "/v1/predict", json={"subject_id": "c1", "features": FEATURES}, headers=AUTH
        )
        assert refused.status_code == 503
    with pytest.raises(ValueError, match="SUBJECT_HASH_SECRET"):
        build_router_app(control_env.settings(subject_hash_secret=None))


def test_canary_assignment_is_sticky_and_logged_with_its_weight(
    router: TestClient, control_env: ControlEnv
) -> None:
    record = control_env.advance_to(control_env.start("better"), 3)  # canary 25%
    runtime = _runtime(router.app)  # type: ignore[arg-type]
    runtime.refresh()
    challenger = control_env.release("better")
    arms: dict[str, str] = {}
    for i in range(400):
        subject = f"cust-{i:07d}"
        body = _predict(router, subject)
        arms[subject] = "challenger" if body["model_version"] == challenger[:12] else "champion"
        expected = assign(subject_hash(SUBJECT_SECRET.encode(), subject), record.salt, 2_500)
        assert arms[subject] == str(expected)
    share = sum(arm == "challenger" for arm in arms.values()) / len(arms)
    assert 0.18 < share < 0.32
    # Advancing to 50% keeps every challenger subject on the challenger.
    control_env.advance_to(record, 4)
    runtime.refresh()
    for subject, arm in list(arms.items())[:200]:
        version = _predict(router, subject)["model_version"]
        if arm == "challenger":
            assert version == challenger[:12]
    runtime.flush()
    weights = {(row["arm"], row["weight"]) for row in _rows(control_env.tmp_path, "router")}
    assert ("challenger", 0.25) in weights
    assert ("champion", 0.5) in weights


def test_an_unverifiable_challenger_leaves_the_champion_serving(
    control_env: ControlEnv, tmp_path: Path
) -> None:
    control_env.advance_to(control_env.start("better"), 4)
    release = control_env.release("better")
    registry_copy = tmp_path / "registry-copy"
    import shutil

    shutil.copytree(control_env.registry.root / "registry", registry_copy)
    path = registry_copy / "releases" / f"{release}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["signature"] = "unsigned"
    path.write_text(json.dumps(data), encoding="utf-8")
    app = build_router_app(control_env.settings(registry_dir=registry_copy), background=False)
    with TestClient(app) as client:
        champion = control_env.release("champion")[:12]
        versions = {_predict(client, f"cust-{i}")["model_version"] for i in range(60)}
        assert versions == {champion}
        metrics = client.get("/metrics", headers=AUTH).text
    assert "app_challenger_rejected_total 1.0" in metrics


def test_feedback_integrity(router: TestClient, control_env: ControlEnv) -> None:
    request_id = str(_predict(router, "cust-1")["request_id"])
    accepted = router.post(
        "/v1/feedback", json={"request_id": request_id, "label": 1}, headers=AUTH
    )
    assert accepted.status_code == 202
    duplicate = router.post(
        "/v1/feedback", json={"request_id": request_id, "label": 0}, headers=AUTH
    )
    assert duplicate.status_code == 409
    unknown = router.post("/v1/feedback", json={"request_id": "e" * 32, "label": 1}, headers=AUTH)
    assert unknown.status_code == 404
    late_id = str(_predict(router, "cust-2")["request_id"])
    control_env.clock[0] += control_env.settings().label_ttl_seconds + 1
    expired = router.post("/v1/feedback", json={"request_id": late_id, "label": 1}, headers=AUTH)
    assert expired.status_code == 404
    malformed = router.post(
        "/v1/feedback", json={"request_id": request_id, "label": 7}, headers=AUTH
    )
    assert malformed.status_code == 422
    _runtime(router.app).flush()  # type: ignore[arg-type]
    labels = _rows(control_env.tmp_path, "labels")
    assert [(row["request_id"], row["label"]) for row in labels] == [(request_id, 1)]


def test_feedback_is_rate_limited_per_key(control_env: ControlEnv) -> None:
    settings = control_env.settings(feedback_rate_per_second=0.001, feedback_burst=2)
    with TestClient(build_router_app(settings, background=False)) as client:
        ids = [str(_predict(client, f"cust-{i}")["request_id"]) for i in range(3)]
        codes = [
            client.post(
                "/v1/feedback", json={"request_id": rid, "label": 0}, headers=AUTH
            ).status_code
            for rid in ids
        ]
    assert codes == [202, 202, 429]


@pytest.mark.parametrize(
    "features",
    [
        {**FEATURES, "discount_rate": 0.9},
        {**FEATURES, "region": "atlantis"},
        {**FEATURES, "usage_trend": "NaN"},
        {**FEATURES, "email": "a@example.com"},
        {k: v for k, v in FEATURES.items() if k != "tenure_months"},
    ],
)
def test_input_contract_rejects_bad_requests_without_echoing_them(
    router: TestClient, features: dict[str, object]
) -> None:
    response = router.post(
        "/v1/predict", json={"subject_id": "cust-9", "features": features}, headers=AUTH
    )
    assert response.status_code == 422
    for leaked in ("atlantis", "a@example.com", "0.9"):
        assert leaked not in response.text


def test_predict_requires_an_api_key_and_a_safe_subject_id(router: TestClient) -> None:
    body = {"subject_id": "cust-1", "features": FEATURES}
    assert router.post("/v1/predict", json=body).status_code == 401
    bad = {"subject_id": "cust 1; DROP", "features": FEATURES}
    assert router.post("/v1/predict", json=bad, headers=AUTH).status_code == 422
    assert router.get("/healthz").status_code == 200


def test_arm_enum_values_are_the_logged_values() -> None:
    assert {str(arm) for arm in Arm} == {"champion", "challenger"}
