from __future__ import annotations

import io

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from progressive_model_delivery.core import tracing
from progressive_model_delivery.core.metrics import create_http_metrics, render_metrics
from progressive_model_delivery.core.tracing import (
    build_tracer_provider,
    install_global_tracer_provider,
)

pytestmark = pytest.mark.unit


def test_provider_records_spans_without_global_state() -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("svc", extra_exporter=exporter)
    with provider.get_tracer("t").start_as_current_span("work"):
        pass
    assert [span.name for span in exporter.get_finished_spans()] == ["work"]
    resource = exporter.get_finished_spans()[0].resource
    assert resource.attributes["service.name"] == "svc"


def test_console_exporter_can_be_selected() -> None:
    """The console exporter must actually format and write the span, not just be configured."""
    provider = build_tracer_provider("svc", "console")
    processors = provider._active_span_processor._span_processors
    assert len(processors) == 1
    exporter = processors[0].span_exporter  # type: ignore[attr-defined]
    assert isinstance(exporter, ConsoleSpanExporter)
    # ConsoleSpanExporter binds sys.stdout as a default argument at import time, so redirect the
    # exporter's own stream rather than the process's.
    stream = io.StringIO()
    exporter.out = stream
    with provider.get_tracer("t").start_as_current_span("console-work"):
        pass
    provider.shutdown()  # flushes the batch processor
    printed = stream.getvalue()
    assert '"name": "console-work"' in printed
    assert '"service.name": "svc"' in printed


def test_global_provider_is_installed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: list[TracerProvider] = []
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", installed.append)
    first, second = build_tracer_provider("a"), build_tracer_provider("b")
    assert install_global_tracer_provider(first) is True
    assert install_global_tracer_provider(second) is False
    assert installed == [first]


def test_metrics_render_in_prometheus_format() -> None:
    metrics = create_http_metrics()
    metrics.requests_total.labels("GET", "/x", "200").inc()
    metrics.request_duration_seconds.labels("GET", "/x").observe(0.01)
    payload, content_type = render_metrics(metrics.registry)
    assert b'app_http_requests_total{method="GET",route="/x",status="200"} 1.0' in payload
    assert content_type.startswith("text/plain")
