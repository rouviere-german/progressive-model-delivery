"""The router: verified champion and challenger, sticky canary assignment, shadow forwarding.

Request path (``POST /v1/predict``):

1. Authenticate (API key), validate the body (pydantic), hash the subject id with HMAC.
2. Pick the arm: champion, unless a canary stage is active and the subject's consistent-hash
   bucket falls under the canary weight.
3. Score with the chosen verified model, measure latency, log the prediction (arm, weight,
   release, subgroup, features, no raw subject id) and remember the request id for feedback.
4. Respond. In the shadow stage the job for the shadow worker is offered to the bounded
   forwarder only after the response is ready (FastAPI background task, ``put_nowait``).

``POST /v1/feedback`` accepts a delayed label only for a request id this router served, within
the label TTL, once, and under a per-key rate limit (metric-poisoning defense).

The serving state (champion, open rollout, stage, weight, salt) is re-read from the control plane
every ``state_refresh_seconds``; models are verified (signature, hashes) before first use. If the
champion cannot be verified the router reports not-ready; if the challenger cannot be verified the
router serves the champion only and counts ``challenger_rejected_total``.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer
from prometheus_client import CollectorRegistry

from progressive_model_delivery.control.spec import StageKind
from progressive_model_delivery.control.store import ControlStore
from progressive_model_delivery.core.app import create_app
from progressive_model_delivery.core.config import Settings
from progressive_model_delivery.core.errors import AppError
from progressive_model_delivery.core.logging import get_logger
from progressive_model_delivery.core.ratelimit import RateLimiter
from progressive_model_delivery.core.security import AuthenticatedPrincipal
from progressive_model_delivery.core.tracing import build_tracer_provider
from progressive_model_delivery.models.features import NUMERIC_FEATURES, SUBGROUP_FEATURE
from progressive_model_delivery.observability import ServingMetrics
from progressive_model_delivery.registry.release import (
    ReleaseStore,
    ReleaseVerificationError,
    VerifiedModel,
)
from progressive_model_delivery.registry.signing import TrustStore
from progressive_model_delivery.serving.assignment import Arm, assign, subject_hash
from progressive_model_delivery.serving.forwarder import (
    ShadowForwarder,
    ShadowMetrics,
    http_sender,
)
from progressive_model_delivery.serving.predlog import (
    LABEL_SCHEMA,
    NO_ROLLOUT,
    FeedbackOutcome,
    RequestIndex,
    SegmentWriter,
)
from progressive_model_delivery.serving.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    PredictRequest,
    PredictResponse,
)

__all__ = [
    "ModelCache",
    "RolloutView",
    "RouterRuntime",
    "ServingState",
    "build_router_app",
    "feature_record",
    "periodic",
]

_log = get_logger("progressive_model_delivery.router")


class ModelCache:
    """Verified models by release id. Verification failures are not cached (retried later)."""

    def __init__(self, store: ReleaseStore, trust: TrustStore) -> None:
        self._store = store
        self._trust = trust
        self._models: dict[str, VerifiedModel] = {}
        self._lock = threading.Lock()

    def get(self, release_id: str) -> VerifiedModel:
        with self._lock:
            cached = self._models.get(release_id)
        if cached is not None:
            return cached
        model = self._store.load_verified(release_id, self._trust)
        with self._lock:
            self._models[release_id] = model
        return model


@dataclass(frozen=True, slots=True)
class RolloutView:
    rollout_id: str
    stage_index: int
    kind: StageKind
    weight_bp: int
    salt: str
    challenger: VerifiedModel | None


@dataclass(frozen=True, slots=True)
class ServingState:
    champion: VerifiedModel
    rollout: RolloutView | None


def feature_record(features: dict[str, Any]) -> dict[str, float]:
    return {f"f_{spec.name}": float(features[spec.name]) for spec in NUMERIC_FEATURES}


class RouterRuntime:
    def __init__(
        self,
        *,
        control: ControlStore,
        models: ModelCache,
        predictions: SegmentWriter,
        labels: SegmentWriter,
        index: RequestIndex,
        secret: bytes,
        metrics: ServingMetrics,
        tracer: Tracer,
        feedback_limiter: RateLimiter,
        clock: Callable[[], float] = time.time,
        forwarder: ShadowForwarder | None = None,
    ) -> None:
        self.control = control
        self.models = models
        self.predictions = predictions
        self.labels = labels
        self.index = index
        self.metrics = metrics
        self.tracer = tracer
        self.feedback_limiter = feedback_limiter
        self.clock = clock
        self.forwarder = forwarder
        self._secret = secret
        self.state: ServingState | None = None

    # --- state -------------------------------------------------------------------------------

    def refresh(self) -> ServingState | None:
        """Re-read the champion and the open rollout; verify any model not seen before."""
        champion_id = self.control.champion()
        if champion_id is None:
            self.state = None
            return None
        try:
            champion = self.models.get(champion_id)
        except ReleaseVerificationError as exc:
            _log.error("champion_rejected", reason=exc.reason)
            self.state = None
            return None
        view: RolloutView | None = None
        rollout = self.control.open_rollout()
        if rollout is not None:
            challenger: VerifiedModel | None
            try:
                challenger = self.models.get(rollout.challenger)
            except ReleaseVerificationError as exc:
                self.metrics.challenger_rejected_total.inc()
                _log.error("challenger_rejected", reason=exc.reason)
                challenger = None
            window = rollout.current
            view = RolloutView(
                rollout_id=rollout.rollout_id,
                stage_index=window.index,
                kind=window.kind,
                weight_bp=window.weight_bp,
                salt=rollout.salt,
                challenger=challenger,
            )
        self.state = ServingState(champion=champion, rollout=view)
        return self.state

    def ready(self) -> bool:
        return self.state is not None

    # --- request handling --------------------------------------------------------------------

    def predict(self, request: PredictRequest) -> tuple[PredictResponse, dict[str, Any] | None]:
        state = self.state
        if state is None:
            raise AppError(503, "not_ready", "No verified champion model is loaded.")
        hashed = subject_hash(self._secret, request.subject_id)
        rollout = state.rollout
        arm = Arm.CHAMPION
        weight = 0.0
        if (
            rollout is not None
            and rollout.challenger is not None
            and rollout.kind is StageKind.CANARY
        ):
            arm = assign(hashed, rollout.salt, rollout.weight_bp)
            weight = rollout.weight_bp / 10_000
        model = state.champion
        if arm is Arm.CHALLENGER and rollout is not None and rollout.challenger is not None:
            model = rollout.challenger
        features = request.features.model_dump()
        short = model.release_id[:12]
        with self.tracer.start_as_current_span(
            "predict",
            attributes={
                "model.version": short,
                "rollout.arm": str(arm),
                "rollout.id": rollout.rollout_id if rollout else NO_ROLLOUT,
            },
        ):
            started = time.perf_counter()
            try:
                score = model.scorer.predict_one(features)
            except Exception:
                if arm is Arm.CHAMPION:
                    raise
                # A failing challenger never fails the request: the champion answers instead.
                self.metrics.prediction_errors_total.labels(short).inc()
                arm, model, short = Arm.CHAMPION, state.champion, state.champion.release_id[:12]
                started = time.perf_counter()
                score = model.scorer.predict_one(features)
            latency = time.perf_counter() - started
        self.metrics.predictions_total.labels(short, str(arm)).inc()
        self.metrics.prediction_latency_seconds.labels(short, str(arm)).observe(latency)
        request_id = uuid.uuid4().hex
        now = self.clock()
        self.predictions.append(
            {
                "request_id": request_id,
                "rollout_id": rollout.rollout_id if rollout else NO_ROLLOUT,
                "ts": now,
                "stage_index": rollout.stage_index if rollout else 0,
                "arm": str(arm),
                "weight": weight,
                "release_id": model.release_id,
                "subject_hash": hashed,
                "subgroup": str(features[SUBGROUP_FEATURE]),
                "score": score,
                "latency_ms": latency * 1_000.0,
                **feature_record(features),
            }
        )
        self.index.remember(request_id, now)
        job: dict[str, Any] | None = None
        if (
            rollout is not None
            and rollout.kind is StageKind.SHADOW
            and rollout.challenger is not None
            and self.forwarder is not None
        ):
            job = {
                "request_id": request_id,
                "rollout_id": rollout.rollout_id,
                "stage_index": rollout.stage_index,
                "release_id": rollout.challenger.release_id,
                "subject_hash": hashed,
                "ts": now,
                "features": features,
            }
        response = PredictResponse(
            request_id=request_id,
            score=score,
            label=int(score >= model.manifest.decision_threshold),
            model_version=short,
        )
        return response, job

    def feedback(self, request: FeedbackRequest, key_id: str) -> FeedbackOutcome:
        if not self.feedback_limiter.check(key_id).allowed:
            self.metrics.feedback_total.labels("rate_limited").inc()
            raise AppError(429, "rate_limited", "Too many label submissions.")
        outcome = self.index.claim(request.request_id)
        self.metrics.feedback_total.labels(str(outcome)).inc()
        if outcome is FeedbackOutcome.ACCEPTED:
            self.labels.append(
                {
                    "request_id": request.request_id,
                    "label": int(request.label),
                    "received_at": self.clock(),
                }
            )
        return outcome

    def flush(self) -> None:
        self.predictions.flush()
        self.labels.flush()


async def periodic(action: Callable[[], object], interval: float) -> None:
    """Run a blocking ``action`` in a worker thread every ``interval`` seconds, forever."""
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(action)
        except Exception as exc:  # noqa: BLE001 - a background failure must not kill the loop
            _log.warning("background_task_failed", error_type=type(exc).__name__)


def build_router_app(
    settings: Settings,
    *,
    clock: Callable[[], float] = time.time,
    shadow_transport: httpx.AsyncBaseTransport | None = None,
    registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
    background: bool = True,
) -> FastAPI:
    """Wire the router from settings. ``shadow_transport`` lets tests and the demo route the
    shadow hop in-process; ``background=False`` disables the refresh and flush loops."""
    if settings.subject_hash_secret is None:
        raise ValueError("APP_SUBJECT_HASH_SECRET is required to serve predictions")
    reg = registry if registry is not None else CollectorRegistry()
    provider = tracer_provider or build_tracer_provider(
        settings.service_name, settings.tracing_exporter
    )
    metrics = ServingMetrics.create(reg)
    client: httpx.AsyncClient | None = None
    forwarder: ShadowForwarder | None = None
    if settings.shadow_url is not None and settings.shadow_token is not None:
        client = httpx.AsyncClient(transport=shadow_transport)
        forwarder = ShadowForwarder(
            http_sender(
                client,
                settings.shadow_url.rstrip("/") + "/internal/shadow",
                settings.shadow_token.get_secret_value(),
                timeout_seconds=settings.shadow_timeout_seconds,
            ),
            capacity=settings.shadow_queue_size,
            workers=settings.shadow_workers,
            metrics=ShadowMetrics.create(reg),
            deadline_seconds=settings.shadow_timeout_seconds * 2,
        )
    runtime = RouterRuntime(
        control=ControlStore(settings.control_db, clock=clock),
        models=ModelCache(
            ReleaseStore(settings.registry_dir),
            TrustStore.from_b64(settings.trusted_signing_keys),
        ),
        predictions=SegmentWriter(settings.predlog_dir / "router", prefix="predictions"),
        labels=SegmentWriter(settings.predlog_dir / "labels", schema=LABEL_SCHEMA, prefix="labels"),
        index=RequestIndex(
            ttl_seconds=settings.label_ttl_seconds,
            capacity=settings.request_index_capacity,
            clock=clock,
        ),
        secret=settings.subject_hash_secret.get_secret_value().encode("utf-8"),
        metrics=metrics,
        tracer=provider.get_tracer("progressive_model_delivery.router"),
        feedback_limiter=RateLimiter(settings.feedback_rate_per_second, settings.feedback_burst),
        clock=clock,
        forwarder=forwarder,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(runtime.refresh)
        if forwarder is not None:
            await forwarder.start()
        tasks = []
        if background:
            tasks = [
                asyncio.create_task(periodic(runtime.refresh, settings.state_refresh_seconds)),
                asyncio.create_task(periodic(runtime.flush, settings.log_flush_seconds)),
            ]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if forwarder is not None:
                await forwarder.stop(drain_seconds=1.0)
            if client is not None:
                await client.aclose()
            await asyncio.to_thread(runtime.flush)

    checks: dict[str, Callable[[], bool | Awaitable[bool]]] = {"champion_verified": runtime.ready}
    app = create_app(
        settings,
        readiness_checks=checks,
        registry=reg,
        tracer_provider=provider,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(
        body: PredictRequest, principal: AuthenticatedPrincipal, background_tasks: BackgroundTasks
    ) -> PredictResponse:
        del principal
        response, job = runtime.predict(body)
        if job is not None and runtime.forwarder is not None:
            background_tasks.add_task(runtime.forwarder.offer, job)
        return response

    @app.post("/v1/feedback", response_model=FeedbackResponse, status_code=202)
    async def feedback(
        body: FeedbackRequest, principal: AuthenticatedPrincipal
    ) -> FeedbackResponse:
        outcome = runtime.feedback(body, principal.key_id)
        if outcome is FeedbackOutcome.UNKNOWN:
            raise AppError(404, "unknown_request", "Unknown or expired request id.")
        if outcome is FeedbackOutcome.DUPLICATE:
            raise AppError(409, "duplicate_label", "A label was already recorded.")
        return FeedbackResponse(status="accepted")

    return app
