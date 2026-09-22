"""Signed, content-addressed model releases with verify-before-load.

Layout under ``root``::

    store/...                         content-addressed blobs and version records (ml.registry)
    releases/<release_id>.json        {"manifest": ReleaseManifest, "key_id", "signature"}

A release id is the SHA-256 of the canonical JSON of its manifest, and the Ed25519 signature is
over exactly those bytes. The manifest pins the version record digest, the model text digest, the
model card digest, the training run manifest digest and the dataset digests (lineage), plus the
feature order and the PSI references used by the controller.

:meth:`ReleaseStore.load_verified` checks, in order: the file name matches the manifest hash, the
signature is by a trusted key, the version record and blobs match their content addresses and the
manifest, the artifact looks like LightGBM text, and only then parses it; the parsed booster's
feature names must equal the manifest's feature order. Anything else raises
:class:`ReleaseVerificationError` with a stable reason code. Model files are LightGBM text; there
is no pickle path.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import lightgbm as lgb
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from progressive_model_delivery.core.canonical import canonical_json, sha256_hex
from progressive_model_delivery.core.fileio import atomic_write_text
from progressive_model_delivery.ml.manifest import RunManifest
from progressive_model_delivery.ml.registry import ModelRegistry, RegistryError
from progressive_model_delivery.models.scoring import ModelScorer
from progressive_model_delivery.registry.signing import SigningKey, TrustStore

__all__ = [
    "MAX_MODEL_BYTES",
    "ReleaseManifest",
    "ReleaseStore",
    "ReleaseVerificationError",
    "SignedRelease",
    "VerifiedModel",
]

MAX_MODEL_BYTES = 32 * 1024 * 1024
_HEX64 = r"^[0-9a-f]{64}$"
_RELEASE_ID = re.compile(_HEX64)
_MODEL_FILE = "model.txt"
_CARD_FILE = "model_card.md"


class ReleaseVerificationError(RegistryError):
    """A release failed verification; ``reason`` is a stable machine-readable code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    model_name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    release: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    model_format: Literal["lightgbm-text"] = "lightgbm-text"
    version_digest: str = Field(pattern=_HEX64)
    artifact_sha256: str = Field(pattern=_HEX64)
    artifact_bytes: int = Field(ge=1, le=MAX_MODEL_BYTES)
    model_card_sha256: str = Field(pattern=_HEX64)
    feature_order: tuple[str, ...] = Field(min_length=1)
    categorical_levels: dict[str, tuple[str, ...]]
    subgroup_feature: str
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    run_manifest_sha256: str = Field(pattern=_HEX64)
    dataset_sha256: tuple[str, ...]
    parent_release: str | None = Field(default=None, pattern=_HEX64)
    metrics: dict[str, float]
    references: dict[str, dict[str, list[float]]]
    created_at: str

    @model_validator(mode="after")
    def _consistent_features(self) -> ReleaseManifest:
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature_order has duplicates")
        if not set(self.categorical_levels) <= set(self.feature_order):
            raise ValueError("categorical features must be part of feature_order")
        if self.subgroup_feature not in self.categorical_levels:
            raise ValueError("the subgroup feature must be categorical")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.model_dump(mode="json")).encode("utf-8")

    def release_id(self) -> str:
        return sha256_hex(self.canonical_bytes())


class SignedRelease(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: ReleaseManifest
    key_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    signature: str = Field(min_length=1, max_length=200)

    @property
    def release_id(self) -> str:
        return self.manifest.release_id()


@dataclass(frozen=True, slots=True)
class VerifiedModel:
    release_id: str
    manifest: ReleaseManifest
    scorer: ModelScorer


def _looks_like_lightgbm_text(text: str) -> bool:
    head = text[:4096]
    return head.startswith("tree\n") and "version=v4" in head


class ReleaseStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry = ModelRegistry(root / "store")
        self._releases = root / "releases"

    def _release_path(self, release_id: str) -> Path:
        if not _RELEASE_ID.fullmatch(release_id):
            raise ReleaseVerificationError("bad_release_id", "release ids are SHA-256 hex digests")
        return self._releases / f"{release_id}.json"

    def publish(
        self,
        *,
        model_name: str,
        release: str,
        model_text: str,
        model_card: str,
        run_manifest: RunManifest,
        feature_order: tuple[str, ...],
        categorical_levels: Mapping[str, tuple[str, ...]],
        subgroup_feature: str,
        metrics: Mapping[str, float],
        references: Mapping[str, dict[str, list[float]]],
        signing_key: SigningKey,
        parent_release: str | None = None,
        created_at: datetime | None = None,
    ) -> SignedRelease:
        """Store the artifacts content-addressed, sign the manifest and write the release."""
        if not _looks_like_lightgbm_text(model_text):
            raise ReleaseVerificationError("not_lightgbm_text", "only LightGBM text models ship")
        model_bytes = model_text.encode("utf-8")
        card_bytes = model_card.encode("utf-8")
        version = self.registry.register(
            model_name,
            {_MODEL_FILE: model_bytes, _CARD_FILE: card_bytes},
            run_manifest,
            metrics=dict(metrics),
        )
        manifest = ReleaseManifest(
            model_name=model_name,
            release=release,
            version_digest=version.digest,
            artifact_sha256=version.files[_MODEL_FILE],
            artifact_bytes=len(model_bytes),
            model_card_sha256=version.files[_CARD_FILE],
            feature_order=feature_order,
            categorical_levels=dict(categorical_levels),
            subgroup_feature=subgroup_feature,
            run_manifest_sha256=run_manifest.sha256(),
            dataset_sha256=tuple(item.sha256 for item in run_manifest.datasets),
            parent_release=parent_release,
            metrics=dict(metrics),
            references=dict(references),
            created_at=(created_at or datetime.now(UTC)).isoformat(),
        )
        signature = base64.b64encode(signing_key.sign(manifest.canonical_bytes())).decode()
        signed = SignedRelease(manifest=manifest, key_id=signing_key.key_id, signature=signature)
        atomic_write_text(
            self._release_path(signed.release_id), signed.model_dump_json(indent=2) + "\n"
        )
        return signed

    def release_ids(self) -> list[str]:
        if not self._releases.exists():
            return []
        return sorted(
            p.stem for p in self._releases.glob("*.json") if _RELEASE_ID.fullmatch(p.stem)
        )

    def read(self, release_id: str) -> SignedRelease:
        """Parse a release file without trusting it (see :meth:`load_verified`)."""
        path = self._release_path(release_id)
        if not path.exists():
            raise ReleaseVerificationError("unknown_release", f"no release {release_id}")
        try:
            return SignedRelease.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReleaseVerificationError("malformed_release", "release file is invalid") from exc

    def verify(self, release_id: str, trust: TrustStore) -> SignedRelease:
        """Check the content address and the signature (no model bytes are parsed)."""
        signed = self.read(release_id)
        if signed.release_id != release_id:
            raise ReleaseVerificationError(
                "manifest_tampered", "manifest does not hash to its release id"
            )
        if not signed.signature.strip() or signed.signature == "unsigned":
            raise ReleaseVerificationError("unsigned", "release carries no signature")
        if signed.key_id not in trust.key_ids:
            raise ReleaseVerificationError("untrusted_key", "release signed by an unknown key")
        if not trust.verify(signed.key_id, signed.manifest.canonical_bytes(), signed.signature):
            raise ReleaseVerificationError("bad_signature", "signature does not verify")
        return signed

    def load_verified(self, release_id: str, trust: TrustStore) -> VerifiedModel:
        """Verify everything, then parse the LightGBM text and check its feature names."""
        signed = self.verify(release_id, trust)
        manifest = signed.manifest
        try:
            version = self.registry.get(manifest.model_name, manifest.version_digest)
        except RegistryError as exc:
            raise ReleaseVerificationError("artifact_tampered", str(exc)) from exc
        if (
            version.files.get(_MODEL_FILE) != manifest.artifact_sha256
            or version.files.get(_CARD_FILE) != manifest.model_card_sha256
        ):
            raise ReleaseVerificationError(
                "artifact_tampered", "version record does not match the manifest"
            )
        try:
            model_bytes = self.registry.read_blob(manifest.artifact_sha256)
            self.registry.read_blob(manifest.model_card_sha256)
        except RegistryError as exc:
            raise ReleaseVerificationError("artifact_tampered", str(exc)) from exc
        if len(model_bytes) != manifest.artifact_bytes or len(model_bytes) > MAX_MODEL_BYTES:
            raise ReleaseVerificationError("artifact_tampered", "artifact size mismatch")
        text = model_bytes.decode("utf-8")
        if not _looks_like_lightgbm_text(text):
            raise ReleaseVerificationError("not_lightgbm_text", "artifact is not LightGBM text")
        booster = lgb.Booster(model_str=text)
        if tuple(booster.feature_name()) != manifest.feature_order:
            raise ReleaseVerificationError(
                "feature_mismatch", "model feature names differ from the manifest feature order"
            )
        scorer = ModelScorer(
            booster,
            feature_order=manifest.feature_order,
            categorical_levels=manifest.categorical_levels,
            threshold=manifest.decision_threshold,
        )
        return VerifiedModel(release_id=release_id, manifest=manifest, scorer=scorer)

    def model_card(self, release_id: str, trust: TrustStore) -> str:
        signed = self.verify(release_id, trust)
        return self.registry.read_blob(signed.manifest.model_card_sha256).decode("utf-8")
