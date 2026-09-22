from __future__ import annotations

import json
import runpy
import sys
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from progressive_model_delivery import cli
from progressive_model_delivery.core import tracing
from progressive_model_delivery.core.security import hash_api_key

pytestmark = pytest.mark.integration


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
