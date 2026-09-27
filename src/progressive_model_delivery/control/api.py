"""The controller service: reconciliation loop plus the role-checked control API.

Endpoints (API key bound to a role in ``APP_ROLE_BINDINGS``):

======  =====================================  ==========================================
POST    /v1/champion                           release_manager: bootstrap the first champion
POST    /v1/rollouts                           deployer: propose a rollout for a release
GET     /v1/rollouts, /v1/rollouts/current     any role
GET     /v1/rollouts/current/evidence          any role: the controller's latest evaluation
POST    /v1/rollouts/{id}/approve              release_manager (not the proposer): promote
POST    /v1/rollouts/{id}/rollback             deployer or release_manager
POST    /v1/rollouts/{id}/overrides            deployer or release_manager: request override
POST    /v1/overrides/{request_id}/approve     release_manager: second signature executes it
GET     /v1/audit, /v1/audit/verify            any role
======  =====================================  ==========================================

``/readyz`` reports ready only while this controller holds the lease.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Request
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field

from progressive_model_delivery.control.controller import Controller
from progressive_model_delivery.control.evidence import EvidenceReader
from progressive_model_delivery.control.rbac import ControlService, Principal, RoleDirectory
from progressive_model_delivery.control.spec import MAX_SPEC_BYTES, bundled_spec, load_spec
from progressive_model_delivery.control.store import ControlStore
from progressive_model_delivery.core.app import create_app
from progressive_model_delivery.core.config import Settings
from progressive_model_delivery.core.errors import AppError
from progressive_model_delivery.core.security import extract_api_key
from progressive_model_delivery.observability import RolloutMetrics
from progressive_model_delivery.registry.release import ReleaseStore
from progressive_model_delivery.registry.signing import TrustStore

__all__ = ["build_control_app"]

_HEX64 = r"^[0-9a-f]{64}$"


def principal_of(request: Request) -> Principal:
    """Authenticate the caller against ``APP_ROLE_BINDINGS`` and return its principal.

    The directory lives on ``app.state`` rather than in a closure because FastAPI resolves the
    string annotations of ``Annotated[..., Depends(...)]`` in *module* scope: an alias defined
    inside the app factory is invisible there and the parameter silently becomes a query
    parameter instead of a dependency (``tests/integration/test_control_api.py`` covers this).
    """
    directory = request.app.state.role_directory
    principal = directory.authenticate(extract_api_key(request.headers))
    if principal is None:
        request.app.state.metrics.auth_failures_total.inc()
        raise AppError(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not isinstance(principal, Principal):  # pragma: no cover - directory contract
        raise TypeError("role directory returned a non-principal")
    return principal


Caller = Annotated[Principal, Depends(principal_of)]


class BootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    release_id: str = Field(pattern=_HEX64)


class ProposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    challenger: str = Field(pattern=_HEX64)
    spec: Literal["standard", "demo"] | None = None
    spec_yaml: str | None = Field(default=None, max_length=MAX_SPEC_BYTES)


class ReasonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1, max_length=500)


class OverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["force_advance"]
    reason: str = Field(min_length=1, max_length=500)


def build_control_app(
    settings: Settings,
    *,
    clock: Callable[[], float] = time.time,
    background: bool = True,
    registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    reg = registry if registry is not None else CollectorRegistry()
    directory = RoleDirectory(settings.role_bindings)
    control = ControlStore(settings.control_db, clock=clock)
    releases = ReleaseStore(settings.registry_dir)
    trust = TrustStore.from_b64(settings.trusted_signing_keys)
    controller = Controller(
        control=control,
        evidence=EvidenceReader(settings.predlog_dir),
        releases=releases,
        trust=trust,
        controller_id=settings.controller_id,
        lease_seconds=settings.lease_seconds,
        metrics=RolloutMetrics.create(reg),
    )
    service = ControlService(control, releases, trust)
    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        thread: threading.Thread | None = None
        if background:
            thread = threading.Thread(
                target=controller.run,
                args=(stop, settings.reconcile_interval_seconds),
                name="controller",
                daemon=True,
            )
            thread.start()
        try:
            yield
        finally:
            stop.set()
            if thread is not None:
                await asyncio.to_thread(thread.join, 10.0)

    checks: dict[str, Callable[[], bool | Awaitable[bool]]] = {
        "lease_held": lambda: controller.holds_lease,
    }
    app = create_app(
        settings,
        readiness_checks=checks,
        registry=reg,
        tracer_provider=tracer_provider,
        lifespan=lifespan,
        extra_key_digests=directory.digests,
    )
    app.state.controller = controller
    app.state.service = service
    app.state.role_directory = directory

    _register_rollout_routes(app, control=control, controller=controller, service=service)
    _register_decision_routes(app, control=control, service=service)
    return app


def _register_rollout_routes(
    app: FastAPI,
    *,
    control: ControlStore,
    controller: Controller,
    service: ControlService,
) -> None:
    """Bootstrap, proposal and read-only rollout views."""

    @app.post("/v1/champion", status_code=201)
    async def bootstrap(body: BootstrapRequest, caller: Caller) -> dict[str, str]:
        await asyncio.to_thread(service.bootstrap, caller, body.release_id)
        return {"champion": body.release_id}

    @app.post("/v1/rollouts", status_code=201)
    async def propose(body: ProposeRequest, caller: Caller) -> dict[str, Any]:
        if (body.spec is None) == (body.spec_yaml is None):
            raise AppError(422, "validation_error", "Give exactly one of spec or spec_yaml.")
        try:
            spec = bundled_spec(body.spec) if body.spec else load_spec(body.spec_yaml or "")
        except (ValueError, TypeError):
            raise AppError(422, "invalid_spec", "The rollout spec is invalid.") from None
        record = await asyncio.to_thread(service.propose, caller, body.challenger, spec)
        return record.model_dump(mode="json")

    @app.get("/v1/rollouts")
    async def rollouts(caller: Caller) -> list[dict[str, Any]]:
        del caller
        return [record.model_dump(mode="json") for record in control.rollouts()]

    @app.get("/v1/rollouts/current")
    async def current(caller: Caller) -> dict[str, Any]:
        del caller
        record = control.open_rollout()
        if record is None:
            raise AppError(404, "not_found", "No rollout is open.")
        return record.model_dump(mode="json")

    @app.get("/v1/rollouts/current/evidence")
    async def evidence(caller: Caller) -> dict[str, Any]:
        del caller
        report = controller.last_report
        if report is None:
            raise AppError(404, "not_found", "The controller has not evaluated a rollout yet.")
        return {
            "outcome": str(report.outcome),
            "rollout_id": report.rollout_id,
            "stage_index": report.stage_index,
            "evidence": report.evidence,
        }


def _register_decision_routes(
    app: FastAPI, *, control: ControlStore, service: ControlService
) -> None:
    """The role-checked decisions (approve, rollback, override) and the audit views."""

    @app.post("/v1/rollouts/{rollout_id}/approve")
    async def approve(rollout_id: str, caller: Caller) -> dict[str, Any]:
        record = await asyncio.to_thread(service.approve_promotion, caller, rollout_id)
        return record.model_dump(mode="json")

    @app.post("/v1/rollouts/{rollout_id}/rollback")
    async def rollback(rollout_id: str, body: ReasonRequest, caller: Caller) -> dict[str, Any]:
        record = await asyncio.to_thread(service.rollback, caller, rollout_id, body.reason)
        return record.model_dump(mode="json")

    @app.post("/v1/rollouts/{rollout_id}/overrides", status_code=201)
    async def request_override(
        rollout_id: str, body: OverrideRequest, caller: Caller
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            service.request_override, caller, rollout_id, body.action, body.reason
        )

    @app.post("/v1/overrides/{request_id}/approve")
    async def approve_override(request_id: str, caller: Caller) -> dict[str, Any]:
        return await asyncio.to_thread(service.approve_override, caller, request_id)

    @app.get("/v1/audit")
    async def audit(caller: Caller, rollout_id: str | None = None) -> list[dict[str, Any]]:
        del caller
        return [entry.model_dump(mode="json") for entry in control.audit_entries(rollout_id)]

    @app.get("/v1/audit/verify")
    async def verify(caller: Caller) -> dict[str, Any]:
        del caller
        result = control.verify_audit()
        return {
            "ok": result.ok,
            "entries": result.entries,
            "head": result.head,
            "error": result.error,
        }
