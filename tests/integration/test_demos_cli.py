"""The command line: key and binding generation, serving each role, audit tooling, bootstrap."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
from opentelemetry import trace as otel_trace

from progressive_model_delivery import cli
from progressive_model_delivery.control.store import ControlStore
from progressive_model_delivery.core import tracing
from progressive_model_delivery.core.config import get_settings
from progressive_model_delivery.core.security import hash_api_key
from progressive_model_delivery.models.pipeline import DemoRegistry
from progressive_model_delivery.registry.release import ReleaseStore
from progressive_model_delivery.registry.signing import SigningKey, TrustStore

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Any:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_cli_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


def test_cli_role_binding_prints_a_settings_entry(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["role-binding", "dana", "deployer"]) == 0
    payload = json.loads(capsys.readouterr().out)
    digest, principal, roles = payload["binding"].split(":", 2)
    assert digest == hash_api_key(payload["api_key"])
    assert (principal, roles) == ("dana", "deployer")


@pytest.mark.parametrize("role", ["router", "controller", "shadow"])
@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_configures_uvicorn_safely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, trained_registry: DemoRegistry, role: str
) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    monkeypatch.setenv("APP_SUBJECT_HASH_SECRET", "subject-secret-" + "s" * 20)
    monkeypatch.setenv("APP_SHADOW_TOKEN", "shadow-token-" + "t" * 20)
    monkeypatch.setenv("APP_REGISTRY_DIR", str(trained_registry.root / "registry"))
    monkeypatch.setenv("APP_CONTROL_DB", str(tmp_path / "control.sqlite"))
    monkeypatch.setenv("APP_PREDLOG_DIR", str(tmp_path / "predlog"))
    monkeypatch.setenv("APP_TRUSTED_SIGNING_KEYS", trained_registry.key.public().to_b64())
    get_settings.cache_clear()

    assert cli.main(["serve", "--role", role, "--port", "8123"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_cli_audit_verifies_and_exports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    database = tmp_path / "control.sqlite"
    assert cli.main(["audit", "verify", "--database", str(database)]) == 2
    assert "no control database" in capsys.readouterr().err

    store = ControlStore(database)
    with store.transaction() as conn:
        store.append_audit(conn, actor="ravi", action="promoted", rollout_id=None, data={"a": 1})
    assert cli.main(["audit", "verify", "--database", str(database)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["ok"] is True
    assert verified["entries"] == 1

    target = tmp_path / "audit.jsonl"
    assert cli.main(["audit", "export", "--database", str(database), "--output", str(target)]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["exported"] == 1
    assert json.loads(target.read_text(encoding="utf-8").splitlines()[0])["actor"] == "ravi"

    with store.transaction() as conn:
        conn.execute("UPDATE audit SET actor = 'mallory' WHERE seq = 0")
    assert cli.main(["audit", "verify", "--database", str(database)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


@pytest.mark.slow
def test_cli_bootstrap_publishes_verifiable_releases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "models"
    key_out = tmp_path / "keys" / "signing.pem"
    assert cli.main(["bootstrap", "--root", str(root), "--key-out", str(key_out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["releases"]) == {
        "champion",
        "better",
        "identical",
        "worse",
        "subgroup_worse",
        "feature_order_bug",
    }
    # Both quality views are printed, so MODEL_CARD.md numbers are reproducible from one command.
    for view in ("pipeline_metrics", "live_metrics"):
        assert set(payload[view]) == set(payload["releases"])
        assert payload[view]["champion"]["error_rate"] > 0.0
    # The pipeline's own view of subgroup_worse looks fine; serving traffic tells another story.
    assert (
        payload["live_metrics"]["subgroup_worse"]["error_rate_south"]
        > payload["pipeline_metrics"]["subgroup_worse"]["error_rate_south"] * 1.5
    )
    assert len(payload["dataset_sha256"]) == 2
    key = SigningKey.load(key_out)
    assert key.public().to_b64() == payload["public_key_b64"]
    trust = TrustStore([key.public()])
    store = ReleaseStore(root / "registry")
    loaded = store.load_verified(payload["releases"]["champion"], trust)
    assert loaded.manifest.model_name == "churn-risk"


def test_cli_bootstrap_warns_when_the_key_is_not_saved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from progressive_model_delivery import cli as cli_module

    def fake_build(root: Path, key: object, **kwargs: Any) -> Any:
        class Models:
            releases: ClassVar[dict[str, object]] = {"champion": object()}
            metrics: ClassVar[dict[str, dict[str, float]]] = {"champion": {"error_rate": 0.1}}
            live_metrics: ClassVar[dict[str, dict[str, float]]] = {"champion": {"error_rate": 0.1}}
            dataset_sha256: ClassVar[tuple[str, ...]] = ("b" * 64,)

            @staticmethod
            def release_id(name: str) -> str:
                return "a" * 64

        return Models()

    monkeypatch.setattr("progressive_model_delivery.models.pipeline.build_demo_models", fake_build)
    assert cli_module.main(["bootstrap", "--root", str(tmp_path / "m")]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["private_key_file"] is None
    assert "not saved" in out.err


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["progressive_model_delivery", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("progressive_model_delivery", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out


def test_an_unknown_subcommand_exits_with_usage() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["nonsense"])
    assert exit_info.value.code == 2
