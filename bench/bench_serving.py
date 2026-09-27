#!/usr/bin/env python3
"""What does the shadow path cost the customer's request?

The router answers from the champion and only then offers the request to a bounded queue that a
pool of tasks drains into the shadow worker. Two things are measured, and one is deliberately not
claimed.

**Measured and enforced**

* ``forwarder_offer`` - the work the *request path* actually does for the shadow stage: build the
  job and ``put_nowait`` it onto the bounded queue, which FastAPI runs as a background task after
  the response is produced. Both the queue-has-room and the queue-is-full paths are timed.
* ``shadow_hanging`` - a shadow worker that never answers. Every job times out, the queue overflows
  and drops, and the champion's own p50/p95/p99 and error count must not move. This is the
  isolation claim from the threat model.

**Measured and reported, not claimed**

* ``shadow_in_process`` - the same traffic with a real shadow worker mounted on the *router's own*
  event loop through an ASGI transport, so its signature verification, LightGBM scoring and Parquet
  writing compete with the request being measured. A single-process benchmark cannot separate the
  forwarder's cost from the worker's compute: the deployed topology gives the worker its own
  container, its own event loop and its own CPU share (see ``docker-compose.yml``), and measuring
  that end to end needs two servers and a concurrent client, which this benchmark is not. The
  column is printed so the difference is visible rather than hidden, and it is *not* the cost of
  shadow forwarding in production.

Absolute latencies include the in-process test client, so only differences between modes mean
anything.

Usage:
  uv run python bench/bench_serving.py [--requests 2000] [--check]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from starlette.types import Receive, Scope, Send

from progressive_model_delivery.control.operations import bootstrap_champion, propose_rollout
from progressive_model_delivery.control.spec import bundled_spec
from progressive_model_delivery.control.store import ControlStore
from progressive_model_delivery.core.config import Settings
from progressive_model_delivery.core.logging import configure_logging
from progressive_model_delivery.core.security import hash_api_key
from progressive_model_delivery.models.pipeline import DemoRegistry
from progressive_model_delivery.serving.forwarder import ShadowForwarder, ShadowMetrics
from progressive_model_delivery.serving.router import RouterRuntime, build_router_app
from progressive_model_delivery.serving.shadow_worker import build_shadow_app

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.harness import (  # noqa: E402 - the repo root has to be importable first
    check_thresholds,
    machine,
    print_table,
    timed,
    write_results,
)

API_KEY = "bench-api-key-" + "b" * 24
SUBJECT_SECRET = "bench-subject-secret-" + "s" * 16
SHADOW_TOKEN = "bench-shadow-token-" + "t" * 16
MODES = ("champion_only", "shadow_in_process", "shadow_hanging")
FEATURES: dict[str, Any] = {
    "tenure_months": 14,
    "monthly_spend": 72.5,
    "support_tickets": 3,
    "num_products": 2,
    "discount_rate": 0.15,
    "usage_trend": -0.2,
    "region": "south",
}


async def _hanging_app(scope: Scope, receive: Receive, send: Send) -> None:
    await asyncio.sleep(3_600)


def _settings(workdir: Path, registry: DemoRegistry, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "api_key_sha256": [hash_api_key(API_KEY)],
        "registry_dir": registry.root / "registry",
        "control_db": workdir / "control.sqlite",
        "predlog_dir": workdir / "predlog",
        "trusted_signing_keys": [registry.key.public().to_b64()],
        "subject_hash_secret": SUBJECT_SECRET,
        "rate_limit_per_second": 1e6,
        "rate_limit_burst": 1_000_000,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _prepare(workdir: Path, registry: DemoRegistry) -> None:
    """A control plane with a champion and an open rollout in the shadow stage."""
    control = ControlStore(workdir / "control.sqlite")
    with control.transaction() as conn:
        bootstrap_champion(control, conn, registry.models.release_id("champion"), actor="bench")
        propose_rollout(
            control,
            conn,
            challenger=registry.models.release_id("better"),
            spec=bundled_spec("demo"),
            actor="bench",
        )


@contextmanager
def _router(
    workdir: Path, registry: DemoRegistry, mode: str
) -> Iterator[tuple[TestClient, RouterRuntime]]:
    extra: dict[str, Any] = {}
    transport: httpx.AsyncBaseTransport | None = None
    shadow_app = None
    if mode != "champion_only":
        extra = {
            "shadow_url": "http://shadow.internal:8001",
            "shadow_token": SHADOW_TOKEN,
            "shadow_timeout_seconds": 0.2,
            "shadow_queue_size": 512,
        }
        if mode == "shadow_in_process":
            shadow_app = build_shadow_app(_settings(workdir, registry, **extra), background=False)
            transport = httpx.ASGITransport(app=shadow_app)
        else:
            transport = httpx.ASGITransport(app=_hanging_app)
    app = build_router_app(
        _settings(workdir, registry, **extra), shadow_transport=transport, background=False
    )
    with TestClient(app) as client:
        runtime = client.app.state.runtime  # type: ignore[attr-defined]
        if not isinstance(runtime, RouterRuntime):  # pragma: no cover - wiring guard
            raise TypeError("router app did not expose its runtime")
        yield client, runtime
    if shadow_app is not None:
        shadow_app.state.runtime.writer.flush()


def _measure(client: TestClient, requests: int, warmup: int = 200) -> dict[str, float]:
    headers = {"X-API-Key": API_KEY}
    body = {"subject_id": "cust-0000001", "features": FEATURES}
    failures = 0
    for index in range(warmup):
        response = client.post(
            "/v1/predict", json={**body, "subject_id": f"warm-{index}"}, headers=headers
        )
        if response.status_code != 200:  # pragma: no cover - configuration error
            raise RuntimeError(f"router answered {response.status_code}: {response.text}")
    samples = np.empty(requests, dtype=np.float64)
    for index in range(requests):
        started = time.perf_counter_ns()
        response = client.post(
            "/v1/predict", json={**body, "subject_id": f"cust-{index:07d}"}, headers=headers
        )
        samples[index] = time.perf_counter_ns() - started
        failures += response.status_code != 200
    micros = samples / 1_000.0
    return {
        "runs": float(requests),
        "failed_requests": float(failures),
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
        "p99_us": round(float(np.percentile(micros, 99)), 1),
    }


def _shadow_counters(client: TestClient) -> dict[str, float]:
    counters: dict[str, float] = {}
    for line in client.get("/metrics", headers={"X-API-Key": API_KEY}).text.splitlines():
        for name in (
            "app_shadow_enqueued_total",
            "app_shadow_delivered_total",
            "app_shadow_dropped_total",
        ):
            if line.startswith(name + " "):
                counters[name] = float(line.rsplit(" ", 1)[1])
        if line.startswith("app_shadow_failed_total{"):
            counters["app_shadow_failed_total"] = counters.get(
                "app_shadow_failed_total", 0.0
            ) + float(line.rsplit(" ", 1)[1])
    return counters


def bench_offer() -> dict[str, dict[str, float]]:
    """The request path's own cost: offer a built job to the bounded queue."""

    async def send(job: dict[str, Any]) -> None:  # pragma: no cover - never awaited here
        del job

    job: dict[str, Any] = {
        "request_id": "a" * 32,
        "rollout_id": "b" * 32,
        "stage_index": 0,
        "release_id": "c" * 64,
        "subject_hash": "d" * 32,
        "ts": 1.0,
        "features": dict(FEATURES),
    }

    async def build() -> tuple[ShadowForwarder, ShadowForwarder]:
        roomy = ShadowForwarder(
            send, capacity=1_000_000, workers=1, metrics=ShadowMetrics.create(CollectorRegistry())
        )
        full = ShadowForwarder(
            send, capacity=1, workers=1, metrics=ShadowMetrics.create(CollectorRegistry())
        )
        full.offer(job)
        return roomy, full

    roomy, full = asyncio.run(build())
    return {
        "forwarder_offer_queued": timed(lambda: roomy.offer(job), runs=20_000, warmup=1_000),
        "forwarder_offer_dropped": timed(lambda: full.offer(job), runs=20_000, warmup=1_000),
    }


def run(requests: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        registry = DemoRegistry.create(root / "models")
        modes: dict[str, Any] = {}
        for mode in MODES:
            workdir = root / mode
            workdir.mkdir()
            _prepare(workdir, registry)
            with _router(workdir, registry, mode) as (client, runtime):
                latency = _measure(client, requests)
                counters = _shadow_counters(client)
                runtime.flush()
            modes[mode] = {**latency, "shadow": counters}

    baseline = modes["champion_only"]
    for payload in modes.values():
        payload["added_p99_ms"] = round((payload["p99_us"] - baseline["p99_us"]) / 1_000.0, 4)
        payload["added_p50_ms"] = round((payload["p50_us"] - baseline["p50_us"]) / 1_000.0, 4)
    return {"requests": requests, "modes": modes, "offer": bench_offer()}


def metrics(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    modes = results["modes"]
    offer = results["offer"]
    return {
        "shadow_isolation": {
            "hanging_added_p99_ms": float(modes["shadow_hanging"]["added_p99_ms"]),
            "hanging_added_p50_ms": float(modes["shadow_hanging"]["added_p50_ms"]),
            "hanging_failed_requests": float(modes["shadow_hanging"]["failed_requests"]),
            "in_process_failed_requests": float(modes["shadow_in_process"]["failed_requests"]),
        },
        "request_path_cost": {
            "offer_queued_p99_us": float(offer["forwarder_offer_queued"]["p99_us"]),
            "offer_dropped_p99_us": float(offer["forwarder_offer_dropped"]["p99_us"]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--requests", type=int, default=2_000)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    # One access log line per timed request would dominate the output and the measurement.
    configure_logging("WARNING", json_output=False, stream=sys.stderr)
    results = run(args.requests)
    write_results("serving", {"machine": machine(), "results": results})
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))

    print(f"POST /v1/predict, {results['requests']} requests per mode (in-process test client)")
    print_table(
        ["mode", "p50 (us)", "p95 (us)", "p99 (us)", "added p99 (ms)", "failed", "shadow jobs"],
        [
            [
                mode,
                f"{payload['p50_us']}",
                f"{payload['p95_us']}",
                f"{payload['p99_us']}",
                f"{payload['added_p99_ms']:+.4f}",
                f"{payload['failed_requests']:.0f}",
                ", ".join(
                    f"{name.removeprefix('app_shadow_').removesuffix('_total')}={value:.0f}"
                    for name, value in sorted(payload["shadow"].items())
                )
                or "-",
            ]
            for mode, payload in results["modes"].items()
        ],
    )
    print()
    print_table(
        ["request-path operation", "runs", "p50 (us)", "p99 (us)"],
        [
            [name, f"{row['runs']:.0f}", f"{row['p50_us']}", f"{row['p99_us']}"]
            for name, row in results["offer"].items()
        ],
    )
    if args.check:
        return check_thresholds("serving", metrics(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
