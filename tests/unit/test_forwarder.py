from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from prometheus_client import CollectorRegistry

from progressive_model_delivery.serving.forwarder import ShadowForwarder, ShadowMetrics, http_sender

pytestmark = pytest.mark.unit


def _value(registry: CollectorRegistry, name: str, labels: dict[str, str] | None = None) -> float:
    return registry.get_sample_value(name, labels or {}) or 0.0


def test_a_full_queue_drops_without_blocking() -> None:
    async def scenario() -> tuple[list[bool], CollectorRegistry]:
        registry = CollectorRegistry()

        async def never(_: dict[str, Any]) -> None:
            await asyncio.sleep(3600)

        forwarder = ShadowForwarder(
            never, capacity=2, workers=1, metrics=ShadowMetrics.create(registry)
        )
        accepted = [forwarder.offer({"i": i}) for i in range(5)]
        return accepted, registry

    accepted, registry = asyncio.run(scenario())
    assert accepted == [True, True, False, False, False]
    assert _value(registry, "app_shadow_dropped_total") == 3
    assert _value(registry, "app_shadow_enqueued_total") == 2


def test_failures_are_counted_by_reason_and_never_raise() -> None:
    async def scenario() -> CollectorRegistry:
        registry = CollectorRegistry()
        calls = {"n": 0}

        async def flaky(job: dict[str, Any]) -> None:
            calls["n"] += 1
            kind = job["kind"]
            if kind == "timeout":
                await asyncio.sleep(10)
            if kind == "transport":
                raise httpx.ConnectError("refused")
            if kind == "error":
                raise RuntimeError("boom")

        forwarder = ShadowForwarder(
            flaky,
            capacity=10,
            workers=2,
            metrics=ShadowMetrics.create(registry),
            deadline_seconds=0.05,
        )
        await forwarder.start()
        running = [forwarder.running]
        for kind in ("ok", "timeout", "transport", "error", "ok"):
            forwarder.offer({"kind": kind})
        await forwarder.stop(drain_seconds=2.0)
        running.append(forwarder.running)
        assert running == [True, False]
        return registry

    registry = asyncio.run(scenario())
    assert _value(registry, "app_shadow_delivered_total") == 2
    for reason in ("timeout", "transport", "error"):
        assert _value(registry, "app_shadow_failed_total", {"reason": reason}) == 1


def test_http_sender_treats_non_202_as_failure() -> None:
    async def scenario() -> list[int]:
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["x-shadow-token"] == "t" * 16
            status = 202 if b'"ok"' in request.content else 500
            seen.append(status)
            return httpx.Response(status)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            send = http_sender(client, "http://shadow/internal/shadow", "t" * 16, timeout_seconds=1)
            await send({"kind": "ok"})
            with pytest.raises(httpx.HTTPStatusError):
                await send({"kind": "bad"})
        return seen

    assert asyncio.run(scenario()) == [202, 500]


def test_invalid_forwarder_configuration_is_rejected() -> None:
    async def noop(_: dict[str, Any]) -> None:
        return None

    with pytest.raises(ValueError, match="positive"):
        ShadowForwarder(
            noop, capacity=0, workers=1, metrics=ShadowMetrics.create(CollectorRegistry())
        )
