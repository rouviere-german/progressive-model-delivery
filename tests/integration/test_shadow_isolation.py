"""Fault injection on the shadow path: the champion's answers must never change.

The shadow worker is reached through a real HTTP client (``httpx``) and is, per test, a working
shadow app, a crashed process (closed port), a hanging server, or a server that errors. Champion
responses are compared with the champion model's own scores for the same inputs.
"""

from __future__ import annotations

import asyncio
import socket
import statistics
import time
from collections.abc import Callable
from typing import Any

import duckdb
import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.types import Receive, Scope, Send

from progressive_model_delivery.serving.router import RouterRuntime, build_router_app
from progressive_model_delivery.serving.shadow_worker import ShadowRuntime, build_shadow_app
from tests.integration.conftest import AUTH, FEATURES, SHADOW_TOKEN, ControlEnv

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _hanging_app(scope: Scope, receive: Receive, send: Send) -> None:
    await asyncio.sleep(3600)


async def _erroring_app(scope: Scope, receive: Receive, send: Send) -> None:
    raise RuntimeError("challenger crashed")


def _run_traffic(
    client: TestClient, n: int, *, subject_prefix: str = "cust"
) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    latencies: list[float] = []
    for i in range(n):
        started = time.perf_counter()
        response = client.post(
            "/v1/predict",
            json={"subject_id": f"{subject_prefix}-{i}", "features": FEATURES},
            headers=AUTH,
        )
        latencies.append(time.perf_counter() - started)
        assert response.status_code == 200, response.text
        scores.append(float(response.json()["score"]))
    return scores, latencies


def _metric(client: TestClient, name: str) -> float:
    for line in client.get("/metrics", headers=AUTH).text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached in time")


def _shadow_router(
    control_env: ControlEnv, transport: httpx.AsyncBaseTransport | None, **extra: Any
) -> TestClient:
    settings = control_env.settings(
        shadow_url=extra.pop("shadow_url", "http://shadow.internal:8001"),
        shadow_token=SHADOW_TOKEN,
        shadow_timeout_seconds=extra.pop("timeout", 0.2),
        shadow_queue_size=extra.pop("queue", 1_000),
        **extra,
    )
    app = build_router_app(settings, shadow_transport=transport, background=False)
    return TestClient(app, raise_server_exceptions=False)


def test_shadow_predictions_are_paired_with_served_requests(control_env: ControlEnv) -> None:
    control_env.start("better")
    shadow_app = build_shadow_app(control_env.settings(shadow_token=SHADOW_TOKEN), background=False)
    shadow_runtime = shadow_app.state.runtime
    assert isinstance(shadow_runtime, ShadowRuntime)
    with _shadow_router(control_env, httpx.ASGITransport(app=shadow_app)) as client:
        _, _ = _run_traffic(client, 40)
        _wait_until(lambda: _metric(client, "app_shadow_delivered_total") >= 40)
        runtime = client.app.state.runtime  # type: ignore[attr-defined]
        assert isinstance(runtime, RouterRuntime)
        runtime.flush()
    shadow_runtime.writer.flush()
    predlog = control_env.tmp_path / "predlog"
    with duckdb.connect() as con:
        paired = con.execute(
            "SELECT count(*), count(DISTINCT s.release_id) FROM read_parquet($r) r "
            "JOIN read_parquet($s) s USING (request_id) WHERE s.arm = 'shadow'",
            {
                "r": [str(p) for p in (predlog / "router").glob("*.parquet")],
                "s": [str(p) for p in (predlog / "shadow").glob("*.parquet")],
            },
        ).fetchone()
    assert paired == (40, 1)


@pytest.mark.parametrize("fault", ["crashed", "hanging", "erroring"])
def test_shadow_faults_never_affect_champion_responses(control_env: ControlEnv, fault: str) -> None:
    control_env.start("feature_order_bug")
    champion = control_env.registry.store.load_verified(
        control_env.release("champion"), control_env.registry.trust
    )
    expected = champion.scorer.predict_one(FEATURES)
    transport: httpx.AsyncBaseTransport | None = None
    extra: dict[str, Any] = {"queue": 16}
    if fault == "crashed":
        extra["shadow_url"] = f"http://127.0.0.1:{_free_port()}"  # nothing listens here
    elif fault == "hanging":
        transport = httpx.ASGITransport(app=_hanging_app)
    else:
        transport = httpx.ASGITransport(app=_erroring_app)
    with _shadow_router(control_env, transport, **extra) as client:
        scores, _ = _run_traffic(client, 120)
        assert scores == [pytest.approx(expected)] * 120
        _wait_until(
            lambda: (
                _metric(client, "app_shadow_dropped_total")
                + sum(
                    _metric(client, f'app_shadow_failed_total{{reason="{r}"}}')
                    for r in ("timeout", "transport", "status", "error")
                )
                > 0
            )
        )
        assert client.get("/readyz").status_code == 200


def test_a_hanging_shadow_worker_does_not_slow_the_champion(control_env: ControlEnv) -> None:
    control_env.start("better")
    baseline_app = build_router_app(control_env.settings(), background=False)  # no shadow path
    with TestClient(baseline_app) as baseline:
        _run_traffic(baseline, 50, subject_prefix="warm")
        _, fast = _run_traffic(baseline, 300)
    with _shadow_router(control_env, httpx.ASGITransport(app=_hanging_app), queue=32) as client:
        _run_traffic(client, 50, subject_prefix="warm")
        _, slow = _run_traffic(client, 300)
        assert _metric(client, "app_shadow_dropped_total") > 0
    # Loose bound for a noisy CI box; bench/bench_serving.py measures the real overhead.
    assert statistics.median(slow) < statistics.median(fast) + 0.005


def test_shadow_worker_authenticates_before_parsing_and_verifies_releases(
    control_env: ControlEnv,
) -> None:
    settings = control_env.settings(shadow_token=SHADOW_TOKEN)
    job = {
        "request_id": "a" * 32,
        "rollout_id": "b" * 32,
        "stage_index": 0,
        "release_id": control_env.release("better"),
        "subject_hash": "c" * 32,
        "ts": 1.0,
        "features": FEATURES,
    }
    with TestClient(build_shadow_app(settings, background=False)) as client:
        assert client.post("/internal/shadow", json=job).status_code == 401
        garbage = client.post(
            "/internal/shadow", content=b"{not json", headers={"X-Shadow-Token": "wrong" * 5}
        )
        assert garbage.status_code == 401
        token = {"X-Shadow-Token": SHADOW_TOKEN}
        assert client.post("/internal/shadow", json=job, headers=token).status_code == 202
        bad = client.post("/internal/shadow", json={**job, "extra": 1}, headers=token)
        assert bad.status_code == 422
        unknown = client.post(
            "/internal/shadow", json={**job, "release_id": "f" * 64}, headers=token
        )
        assert unknown.status_code == 422
        assert unknown.json()["error"]["code"] == "untrusted_release"
    with TestClient(build_shadow_app(control_env.settings(), background=False)) as client:
        assert client.get("/readyz").status_code == 503  # no token configured: fail closed
        assert client.post("/internal/shadow", json=job, headers=token).status_code == 401
