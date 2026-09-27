"""Fixtures for service-level tests: settings, a control plane with a champion, rollout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from progressive_model_delivery.control.operations import (
    advance,
    bootstrap_champion,
    propose_rollout,
)
from progressive_model_delivery.control.spec import RolloutSpec, bundled_spec
from progressive_model_delivery.control.store import ControlStore, RolloutRecord
from progressive_model_delivery.core.config import Settings
from progressive_model_delivery.core.security import hash_api_key
from progressive_model_delivery.models.pipeline import DemoRegistry

API_KEY = "router-test-key-" + "k" * 24
AUTH = {"X-API-Key": API_KEY}
SUBJECT_SECRET = "subject-secret-" + "s" * 20
SHADOW_TOKEN = "shadow-token-" + "t" * 20
FEATURES: dict[str, Any] = {
    "tenure_months": 14,
    "monthly_spend": 72.5,
    "support_tickets": 3,
    "num_products": 2,
    "discount_rate": 0.15,
    "usage_trend": -0.2,
    "region": "south",
}


def sub(payload: object, *keys: str) -> dict[str, Any]:
    """Walk a JSON payload down ``keys`` and assert every hop is an object.

    Evidence and audit payloads are typed ``JsonValue``, so tests have to narrow before they
    index. Doing it here keeps the assertions readable and keeps `mypy --strict` honest instead
    of scattering ``type: ignore`` comments over the suite.
    """
    node = payload
    for key in keys:
        assert isinstance(node, dict), f"{key}: expected an object, got {type(node).__name__}"
        assert key in node, f"missing key {key!r}"
        node = node[key]
    assert isinstance(node, dict), f"expected an object at {'.'.join(keys)}"
    return dict(node)


def number(payload: object, *keys: str) -> float:
    """The number at ``keys`` inside a JSON payload (the last key may address a scalar)."""
    node = sub(payload, *keys[:-1]) if len(keys) > 1 else payload
    if keys:
        assert isinstance(node, dict), "expected an object"
        node = node[keys[-1]]
    assert not isinstance(node, bool), f"expected a number at {'.'.join(keys)}, got a bool"
    assert isinstance(node, int | float), f"expected a number at {'.'.join(keys)}, got {node!r}"
    return float(node)


def service_settings(tmp_path: Path, registry: DemoRegistry, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "api_key_sha256": [hash_api_key(API_KEY)],
        "registry_dir": registry.root / "registry",
        "control_db": tmp_path / "control.sqlite",
        "predlog_dir": tmp_path / "predlog",
        "trusted_signing_keys": [registry.key.public().to_b64()],
        "subject_hash_secret": SUBJECT_SECRET,
        "rate_limit_per_second": 100_000,
        "rate_limit_burst": 100_000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@dataclass
class ControlEnv:
    tmp_path: Path
    registry: DemoRegistry
    control: ControlStore
    clock: list[float]

    def settings(self, **overrides: Any) -> Settings:
        return service_settings(self.tmp_path, self.registry, **overrides)

    def release(self, name: str) -> str:
        return self.registry.models.release_id(name)

    def start(self, challenger: str, spec: RolloutSpec | None = None) -> RolloutRecord:
        with self.control.transaction() as conn:
            return propose_rollout(
                self.control,
                conn,
                challenger=self.release(challenger),
                spec=spec or bundled_spec("demo"),
                actor="dana",
            )

    def advance_to(self, record: RolloutRecord, stage_index: int) -> RolloutRecord:
        while record.stage_index < stage_index:
            with self.control.transaction() as conn:
                record = advance(self.control, conn, record, actor="test", evidence={})
        return record


@pytest.fixture
def control_env(tmp_path: Path, trained_registry: DemoRegistry) -> ControlEnv:
    clock = [1_000_000.0]
    control = ControlStore(tmp_path / "control.sqlite", clock=lambda: clock[0])
    with control.transaction() as conn:
        bootstrap_champion(
            control, conn, trained_registry.models.release_id("champion"), actor="ops"
        )
    return ControlEnv(tmp_path, trained_registry, control, clock)
