"""Role-based control of rollouts: who may propose, approve, roll back and override.

Roles (bound to API keys through ``APP_ROLE_BINDINGS``):

* ``deployer``: proposes rollouts, rolls back, requests overrides.
* ``release_manager``: bootstraps the first champion, approves final promotion (never their own
  proposal), rolls back, requests and approves overrides.
* ``viewer``: reads rollouts, evidence and the audit chain.

Automatic rollback by the controller needs no approval. Final promotion needs one release
manager other than the proposer. Overrides (forcing a stage advance past its statistical gate)
follow a two-person rule: they execute only once two *distinct* release managers have signed
them, and they expire if the rollout moved on in the meantime. Every request, approval,
refusal-worthy state and execution lands in the hash-chained audit log.
"""

from __future__ import annotations

import hashlib
import hmac
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import JsonValue

from progressive_model_delivery.control.operations import (
    OperationError,
    advance,
    bootstrap_champion,
    close,
    promote,
    propose_rollout,
)
from progressive_model_delivery.control.spec import RolloutSpec
from progressive_model_delivery.control.store import ControlStore, RolloutRecord
from progressive_model_delivery.core.errors import AppError
from progressive_model_delivery.registry.release import ReleaseStore, ReleaseVerificationError
from progressive_model_delivery.registry.signing import TrustStore

__all__ = ["ControlService", "Principal", "Role", "RoleDirectory"]

OVERRIDE_ACTIONS = frozenset({"force_advance"})
REQUIRED_OVERRIDE_APPROVALS = 2


class Role(StrEnum):
    DEPLOYER = "deployer"
    RELEASE_MANAGER = "release_manager"
    VIEWER = "viewer"


@dataclass(frozen=True, slots=True)
class Principal:
    name: str
    roles: frozenset[Role]

    def has(self, *roles: Role) -> bool:
        return bool(self.roles & set(roles))


class RoleDirectory:
    """API-key digest -> principal. Lookups compare against every digest (constant time)."""

    def __init__(self, bindings: Iterable[str]) -> None:
        self._entries: list[tuple[bytes, Principal]] = []
        for binding in bindings:
            digest, name, roles = binding.split(":", 2)
            self._entries.append(
                (
                    bytes.fromhex(digest),
                    Principal(name, frozenset(Role(r) for r in roles.split("+"))),
                )
            )

    @property
    def digests(self) -> list[str]:
        return [digest.hex() for digest, _ in self._entries]

    def authenticate(self, presented: str | None) -> Principal | None:
        if not presented or len(presented) > 512:
            return None
        digest = hashlib.sha256(presented.encode("utf-8")).digest()
        found: Principal | None = None
        for candidate, principal in self._entries:
            if hmac.compare_digest(digest, candidate):
                found = principal
        return found


def _forbid(message: str) -> AppError:
    return AppError(403, "forbidden", message)


def _conflict(message: str) -> AppError:
    return AppError(409, "conflict", message)


class ControlService:
    def __init__(self, control: ControlStore, releases: ReleaseStore, trust: TrustStore) -> None:
        self.control = control
        self.releases = releases
        self.trust = trust

    # --- helpers -----------------------------------------------------------------------------

    @staticmethod
    def require(principal: Principal, *roles: Role) -> None:
        if not principal.has(*roles):
            raise _forbid("This action needs one of the roles: " + ", ".join(roles))

    def _verified(self, release_id: str) -> None:
        try:
            self.releases.load_verified(release_id, self.trust)
        except ReleaseVerificationError as exc:
            raise AppError(422, "untrusted_release", f"Release refused ({exc.reason}).") from None

    def _rollout(self, rollout_id: str) -> RolloutRecord:
        try:
            return self.control.rollout(rollout_id)
        except KeyError:
            raise AppError(404, "not_found", "Unknown rollout.") from None

    # --- actions -----------------------------------------------------------------------------

    def bootstrap(self, principal: Principal, release_id: str) -> None:
        self.require(principal, Role.RELEASE_MANAGER)
        self._verified(release_id)
        try:
            with self.control.transaction() as conn:
                bootstrap_champion(self.control, conn, release_id, actor=principal.name)
        except OperationError as exc:
            raise _conflict(str(exc)) from None

    def propose(self, principal: Principal, challenger: str, spec: RolloutSpec) -> RolloutRecord:
        self.require(principal, Role.DEPLOYER)
        self._verified(challenger)
        try:
            with self.control.transaction() as conn:
                return propose_rollout(
                    self.control, conn, challenger=challenger, spec=spec, actor=principal.name
                )
        except OperationError as exc:
            raise _conflict(str(exc)) from None

    def approve_promotion(self, principal: Principal, rollout_id: str) -> RolloutRecord:
        self.require(principal, Role.RELEASE_MANAGER)
        record = self._rollout(rollout_id)
        if principal.name == record.proposed_by:
            raise _forbid("The proposer of a rollout cannot approve its promotion.")
        if record.status != "awaiting_approval":
            raise _conflict(f"Rollout is {record.status}, not awaiting approval.")
        evidence = self._latest_evidence(rollout_id)
        try:
            with self.control.transaction() as conn:
                current = self.control.rollout(rollout_id, conn)
                if current.revision != record.revision:
                    raise _conflict("Rollout changed; review the new evidence first.")
                return promote(
                    self.control, conn, current, approved_by=(principal.name,), evidence=evidence
                )
        except OperationError as exc:
            raise _conflict(str(exc)) from None

    def rollback(self, principal: Principal, rollout_id: str, reason: str) -> RolloutRecord:
        """Manual rollback: always allowed for deployers and release managers."""
        self.require(principal, Role.DEPLOYER, Role.RELEASE_MANAGER)
        record = self._rollout(rollout_id)
        try:
            with self.control.transaction() as conn:
                current = self.control.rollout(rollout_id, conn)
                return close(
                    self.control,
                    conn,
                    current,
                    actor=principal.name,
                    status="rolled_back",
                    reason=f"manual: {reason}",
                    evidence={"requested_by": principal.name, "revision": record.revision},
                )
        except OperationError as exc:
            raise _conflict(str(exc)) from None

    def request_override(
        self, principal: Principal, rollout_id: str, action: str, reason: str
    ) -> dict[str, Any]:
        self.require(principal, Role.DEPLOYER, Role.RELEASE_MANAGER)
        if action not in OVERRIDE_ACTIONS:
            raise AppError(422, "unknown_override", "Unsupported override action.")
        record = self._rollout(rollout_id)
        if record.status != "active":
            raise _conflict(f"Rollout is {record.status}.")
        request_id = uuid.uuid4().hex
        approvals: list[JsonValue] = [principal.name] if principal.has(Role.RELEASE_MANAGER) else []
        body: dict[str, JsonValue] = {
            "action": action,
            "reason": reason,
            "requested_by": principal.name,
            "approvals": approvals,
            "rollout_revision": record.revision,
        }
        with self.control.transaction() as conn:
            self.control.put_approval(
                conn,
                request_id=request_id,
                rollout_id=rollout_id,
                kind="override",
                record=body,
                status="pending",
            )
            self.control.append_audit(
                conn,
                actor=principal.name,
                action="override_requested",
                rollout_id=rollout_id,
                data={"request_id": request_id, **body},
            )
        return {"request_id": request_id, "status": "pending", **body}

    def approve_override(self, principal: Principal, request_id: str) -> dict[str, Any]:
        """Add a release manager's signature; execute once two distinct managers signed."""
        self.require(principal, Role.RELEASE_MANAGER)
        with self.control.transaction() as conn:
            stored = self.control.approval(request_id, conn)
            if stored is None or stored["kind"] != "override":
                raise AppError(404, "not_found", "Unknown override request.")
            if stored["status"] != "pending":
                raise _conflict(f"Override is {stored['status']}.")
            body: dict[str, Any] = dict(stored["record"])
            approvals = [str(name) for name in body["approvals"]]
            if principal.name in approvals:
                raise _forbid("The same person cannot sign an override twice.")
            approvals.append(principal.name)
            body["approvals"] = list(approvals)
            rollout = self.control.rollout(str(stored["rollout_id"]), conn)
            status = "pending"
            if rollout.revision != body["rollout_revision"] or rollout.status != "active":
                status = "expired"
            elif len(set(approvals)) >= REQUIRED_OVERRIDE_APPROVALS:
                advance(
                    self.control,
                    conn,
                    rollout,
                    actor="override:" + "+".join(approvals),
                    evidence={
                        "override": True,
                        "request_id": request_id,
                        "reason": str(body["reason"]),
                        "approved_by": list(approvals),
                    },
                )
                status = "executed"
            self.control.put_approval(
                conn,
                request_id=request_id,
                rollout_id=str(stored["rollout_id"]),
                kind="override",
                record=body,
                status=status,
            )
            self.control.append_audit(
                conn,
                actor=principal.name,
                action=f"override_{status}" if status != "pending" else "override_signed",
                rollout_id=str(stored["rollout_id"]),
                data={"request_id": request_id, "approvals": list(approvals)},
            )
        return {"request_id": request_id, "status": status, **body}

    def _latest_evidence(self, rollout_id: str) -> dict[str, JsonValue]:
        """The evidence of the controller's promotion request, carried into the approval."""
        for entry in reversed(self.control.audit_entries(rollout_id)):
            if entry.action == "promotion_requested":
                evidence = entry.data.get("evidence")
                return {"promotion_request_seq": entry.seq, "evidence": evidence}
        return {}
