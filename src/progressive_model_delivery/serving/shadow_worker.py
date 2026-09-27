"""The shadow worker: scores challenger releases on mirrored traffic, off the request path.

It runs as a separate process (its own container on an internal-only network in compose). Its
only inputs are shadow jobs from the router, authenticated with a shared token, and its only
output is shadow prediction records in its own log directory. It holds no API keys and no role
bindings, so it has no control-plane credentials, and it cannot answer a customer or reach the
network beyond its internal peer. It verifies every release's signature before loading it, like
the router.

Two things make "no control-plane access" more than a statement about credentials, and both live in
``docker-compose.yml`` rather than here: the internal-only network, and a read-only mount of the
shared state volume with a separate writable volume for this worker's own segments. Otherwise a
compromised worker could rewrite ``control.sqlite`` or the router's prediction log -- the input to
every guardrail -- with no credential at all. ``scripts/compose_e2e.sh`` asserts both.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry
from pydantic import ValidationError

from progressive_model_delivery.core.app import create_app
from progressive_model_delivery.core.config import Settings
from progressive_model_delivery.core.errors import AppError
from progressive_model_delivery.observability import ServingMetrics
from progressive_model_delivery.registry.release import ReleaseStore, ReleaseVerificationError
from progressive_model_delivery.registry.signing import TrustStore
from progressive_model_delivery.serving.predlog import SegmentWriter
from progressive_model_delivery.serving.router import ModelCache, feature_record, periodic
from progressive_model_delivery.serving.schemas import ShadowJob

__all__ = ["ShadowRuntime", "build_shadow_app"]


class ShadowRuntime:
    def __init__(
        self,
        *,
        models: ModelCache,
        writer: SegmentWriter,
        token: bytes | None,
        metrics: ServingMetrics,
    ) -> None:
        self.models = models
        self.writer = writer
        self._token = token
        self.metrics = metrics

    def authorized(self, presented: str | None) -> bool:
        if self._token is None or not presented:
            return False
        return hmac.compare_digest(presented.encode("utf-8"), self._token)

    def score(self, job: ShadowJob) -> None:
        try:
            model = self.models.get(job.release_id)
        except ReleaseVerificationError:
            raise AppError(422, "untrusted_release", "Release failed verification.") from None
        features = job.features.model_dump()
        short = job.release_id[:12]
        started = time.perf_counter()
        try:
            score = model.scorer.predict_one(features)
        except Exception:
            self.metrics.prediction_errors_total.labels(short).inc()
            raise
        latency = time.perf_counter() - started
        self.metrics.predictions_total.labels(short, "shadow").inc()
        self.metrics.prediction_latency_seconds.labels(short, "shadow").observe(latency)
        self.writer.append(
            {
                "request_id": job.request_id,
                "rollout_id": job.rollout_id,
                "ts": job.ts,
                "stage_index": job.stage_index,
                "arm": "shadow",
                "weight": 0.0,
                "release_id": job.release_id,
                "subject_hash": job.subject_hash,
                "subgroup": str(features["region"]),
                "score": score,
                "latency_ms": latency * 1_000.0,
                **feature_record(features),
            }
        )


def build_shadow_app(
    settings: Settings,
    *,
    registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
    background: bool = True,
) -> FastAPI:
    reg = registry if registry is not None else CollectorRegistry()
    token = (
        settings.shadow_token.get_secret_value().encode("utf-8") if settings.shadow_token else None
    )
    runtime = ShadowRuntime(
        models=ModelCache(
            ReleaseStore(settings.registry_dir),
            TrustStore.from_b64(settings.trusted_signing_keys),
        ),
        writer=SegmentWriter(settings.predlog_dir / "shadow", prefix="shadow"),
        token=token,
        metrics=ServingMetrics.create(reg),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = (
            asyncio.create_task(periodic(runtime.writer.flush, settings.log_flush_seconds))
            if background
            else None
        )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await asyncio.to_thread(runtime.writer.flush)

    checks: dict[str, Callable[[], bool | Awaitable[bool]]] = {
        "token_configured": lambda: token is not None,
    }
    app = create_app(
        settings,
        readiness_checks=checks,
        registry=reg,
        tracer_provider=tracer_provider,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.post("/internal/shadow", status_code=202, include_in_schema=False)
    async def shadow(request: Request) -> dict[str, str]:
        # The token is checked before the body is parsed, so unauthenticated callers learn
        # nothing about the job schema.
        if not runtime.authorized(request.headers.get("x-shadow-token")):
            raise AppError(401, "unauthorized", "A valid shadow token is required.")
        try:
            job = ShadowJob.model_validate_json(await request.body())
        except ValidationError:
            raise AppError(422, "validation_error", "Shadow job validation failed.") from None
        await asyncio.to_thread(runtime.score, job)
        return {"status": "accepted"}

    return app
