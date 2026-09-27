"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. Secrets are ``SecretStr`` so they never appear in ``repr``
or logs. API keys are configured as SHA-256 digests, so the process never holds raw keys.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")
_ROLE_BINDING = re.compile(
    r"^[0-9a-f]{64}:[a-z][a-z0-9_.-]{0,31}:(deployer|release_manager|viewer)"
    r"(\+(deployer|release_manager|viewer))*$"
)
_URL = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?(/[A-Za-z0-9._/-]*)?$")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(
        default="progressive-model-delivery", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$"
    )
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"

    # --- storage ---
    data_dir: Path = Path("var")
    registry_dir: Path = Path("var/registry")
    control_db: Path = Path("var/control.sqlite")
    predlog_dir: Path = Path("var/predlog")

    # --- model supply chain ---
    # Base64 raw Ed25519 public keys; releases signed by any other key are refused.
    trusted_signing_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- router ---
    subject_hash_secret: SecretStr | None = None
    label_ttl_seconds: float = Field(default=3_600.0, gt=0)
    request_index_capacity: int = Field(default=200_000, ge=1)
    feedback_rate_per_second: float = Field(default=50.0, gt=0)
    feedback_burst: int = Field(default=200, ge=1)
    state_refresh_seconds: float = Field(default=1.0, gt=0)
    log_flush_seconds: float = Field(default=1.0, gt=0)

    # --- shadow path ---
    shadow_url: str | None = None
    shadow_token: SecretStr | None = None
    shadow_queue_size: int = Field(default=1_000, ge=1)
    shadow_workers: int = Field(default=4, ge=1, le=64)
    shadow_timeout_seconds: float = Field(default=0.5, gt=0)

    # --- controller and control API ---
    # Comma-separated "<sha256 of API key>:<principal>:<role>[+<role>]" entries.
    role_bindings: Annotated[list[str], NoDecode] = Field(default_factory=list)
    controller_id: str = Field(default="controller-1", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    lease_seconds: float = Field(default=15.0, gt=0)
    reconcile_interval_seconds: float = Field(default=5.0, gt=0)

    @field_validator("api_key_sha256", mode="before")
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            if not _SHA256_HEX.fullmatch(digest):
                raise ValueError("api_key_sha256 entries must be lowercase hex SHA-256 digests")
        return value

    @field_validator("trusted_signing_keys", "role_bindings", mode="before")
    @classmethod
    def _parse_lists(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("role_bindings")
    @classmethod
    def _check_role_bindings(cls, value: list[str]) -> list[str]:
        for binding in value:
            if not _ROLE_BINDING.fullmatch(binding):
                raise ValueError(
                    "role_bindings entries look like <sha256>:<principal>:<role>[+<role>] with "
                    "roles deployer, release_manager or viewer"
                )
        return value

    @field_validator("shadow_url")
    @classmethod
    def _check_shadow_url(cls, value: str | None) -> str | None:
        if value is not None and not _URL.fullmatch(value):
            raise ValueError("shadow_url must be an http(s) URL without credentials or query")
        return value

    @field_validator("subject_hash_secret", "shadow_token")
    @classmethod
    def _check_secret_length(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 16:
            raise ValueError("secrets must be at least 16 characters")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
