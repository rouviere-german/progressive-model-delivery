from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from progressive_model_delivery import cli
from progressive_model_delivery.core import tracing
from progressive_model_delivery.core.security import hash_api_key
from progressive_model_delivery.ml.demo import run_ml_demo

pytestmark = pytest.mark.integration


def test_ml_demo_checks_contract_registry_and_drift(tmp_path: Path) -> None:
    summary = run_ml_demo(tmp_path)
    assert summary["rows"] == 2_000
    assert len(summary["contract_failures"]) == 2  # type: ignore[arg-type]
    assert summary["restored_threshold_matches"] is True
    assert summary["stable_drifted"] is False
    assert summary["shifted_drifted"] is True
    assert isinstance(summary["model_version"], str)
    assert len(summary["model_version"]) == 64


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_prints_one_json_summary_per_kit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--workdir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    decoder = json.JSONDecoder()
    demos: list[str] = []
    index = 0
    while index < len(output):
        if output[index].isspace():
            index += 1
            continue
        obj, index = decoder.raw_decode(output, index)
        demos.append(obj["demo"])
    assert demos == ["ml"]


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_skips_removed_kits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert cli.main(["demo", "--kit", "ml", "--workdir", str(tmp_path)]) == 1


def test_cli_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_configures_uvicorn_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["progressive_model_delivery", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("progressive_model_delivery", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out
