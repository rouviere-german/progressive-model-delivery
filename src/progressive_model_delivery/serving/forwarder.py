"""Bounded, drop-on-overflow forwarder from the router to the out-of-process shadow worker.

The champion's response never waits for the shadow path: the router calls :meth:`offer` after the
response is produced, ``offer`` is a non-blocking ``put_nowait`` on a bounded queue, and a full
queue drops the job and counts it. A fixed pool of worker tasks posts jobs to the shadow worker
with a hard timeout; a crashed, hanging or erroring worker only ever costs dropped or failed
shadow jobs, never champion latency or errors (see ``tests/integration/test_shadow_isolation.py``
and ``bench/bench_serving.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from prometheus_client import CollectorRegistry, Counter, Gauge

from progressive_model_delivery.core.logging import get_logger

__all__ = ["ShadowForwarder", "ShadowMetrics", "http_sender"]

Sender = Callable[[dict[str, Any]], Awaitable[None]]
_log = get_logger("progressive_model_delivery.shadow")


@dataclass(frozen=True, slots=True)
class ShadowMetrics:
    enqueued_total: Counter
    dropped_total: Counter
    failed_total: Counter
    delivered_total: Counter
    queue_depth: Gauge

    @classmethod
    def create(cls, registry: CollectorRegistry, namespace: str = "app") -> ShadowMetrics:
        return cls(
            enqueued_total=Counter(
                "shadow_enqueued_total",
                "Shadow jobs accepted.",
                namespace=namespace,
                registry=registry,
            ),
            dropped_total=Counter(
                "shadow_dropped_total",
                "Shadow jobs dropped because the queue was full.",
                namespace=namespace,
                registry=registry,
            ),
            failed_total=Counter(
                "shadow_failed_total",
                "Shadow deliveries that errored or timed out.",
                ["reason"],
                namespace=namespace,
                registry=registry,
            ),
            delivered_total=Counter(
                "shadow_delivered_total",
                "Shadow jobs accepted by the shadow worker.",
                namespace=namespace,
                registry=registry,
            ),
            queue_depth=Gauge(
                "shadow_queue_depth",
                "Jobs waiting in the shadow queue.",
                namespace=namespace,
                registry=registry,
            ),
        )


def http_sender(
    client: httpx.AsyncClient, url: str, token: str, *, timeout_seconds: float
) -> Sender:
    """POST each job to the shadow worker; any non-202 answer counts as a failure."""

    async def send(job: dict[str, Any]) -> None:
        response = await client.post(
            url, json=job, headers={"X-Shadow-Token": token}, timeout=timeout_seconds
        )
        if response.status_code != 202:
            raise httpx.HTTPStatusError(
                f"shadow worker answered {response.status_code}",
                request=response.request,
                response=response,
            )

    return send


class ShadowForwarder:
    def __init__(
        self,
        send: Sender,
        *,
        capacity: int,
        workers: int,
        metrics: ShadowMetrics,
        deadline_seconds: float = 1.0,
    ) -> None:
        if capacity < 1 or workers < 1:
            raise ValueError("capacity and workers must be positive")
        self._send = send
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=capacity)
        self._workers = workers
        self._deadline = deadline_seconds
        self.metrics = metrics
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def running(self) -> bool:
        return bool(self._tasks) and not all(task.done() for task in self._tasks)

    def offer(self, job: dict[str, Any]) -> bool:
        """Enqueue without blocking; returns False (and counts a drop) when the queue is full."""
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            self.metrics.dropped_total.inc()
            return False
        self.metrics.enqueued_total.inc()
        self.metrics.queue_depth.set(self._queue.qsize())
        return True

    async def start(self) -> None:
        if not self._tasks:
            self._tasks = [asyncio.create_task(self._work()) for _ in range(self._workers)]

    async def stop(self, *, drain_seconds: float = 0.0) -> None:
        if drain_seconds > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._queue.join(), timeout=drain_seconds)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    async def _work(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await asyncio.wait_for(self._send(job), timeout=self._deadline)
            except TimeoutError:
                self.metrics.failed_total.labels("timeout").inc()
            except httpx.TransportError:
                self.metrics.failed_total.labels("transport").inc()
            except httpx.HTTPStatusError:
                self.metrics.failed_total.labels("status").inc()
            except Exception as exc:  # noqa: BLE001 - the shadow path must never propagate
                self.metrics.failed_total.labels("error").inc()
                _log.warning("shadow_delivery_failed", error_type=type(exc).__name__)
            else:
                self.metrics.delivered_total.inc()
            finally:
                self._queue.task_done()
                self.metrics.queue_depth.set(self._queue.qsize())
