"""The metric catalogue the runbook and the dashboards depend on.

A renamed or dropped metric is a silent outage of the rollback runbook: the page tells someone to
look at ``app_e_value{guardrail=...}`` and there is nothing there. These tests touch every metric
once and then assert against the *rendered exposition text* -- the bytes a scraper actually sees --
so a change to a name, a label set or the namespace fails here.
"""

from __future__ import annotations

import math

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from progressive_model_delivery.observability import STAGE_CODES, RolloutMetrics, ServingMetrics
from progressive_model_delivery.registry.lifecycle import Stage
from progressive_model_delivery.serving.forwarder import ShadowMetrics

pytestmark = pytest.mark.unit

# Exported sample name -> the label pairs the README and docs/RUNBOOK_ROLLBACK.md name.
SERVING_SAMPLES = {
    'app_predictions_total{arm="challenger",release="abc123"}': None,
    'app_prediction_latency_seconds_count{arm="challenger",release="abc123"}': None,
    'app_prediction_errors_total{release="abc123"}': None,
    'app_feedback_total{outcome="accepted"}': None,
    "app_challenger_rejected_total": None,
}
ROLLOUT_SAMPLES = {
    'app_rollout_stage{release="abc123"}': 2.0,
    "app_canary_weight": 0.25,
    'app_e_value{guardrail="error_rate",test="harm"}': 7.0,
    'app_guardrail_breaches_total{guardrail="error_rate"}': 1.0,
    'app_feature_psi{feature="usage_trend"}': 0.03,
    "app_score_psi": 0.04,
    'app_reconcile_total{outcome="advanced"}': 1.0,
    "app_lease_held": 1.0,
}
SHADOW_SAMPLES = {
    "app_shadow_enqueued_total": 1.0,
    "app_shadow_dropped_total": 1.0,
    'app_shadow_failed_total{reason="timeout"}': 1.0,
    "app_shadow_delivered_total": 1.0,
    "app_shadow_queue_depth": 3.0,
}


def _touch_serving(metrics: ServingMetrics) -> None:
    metrics.predictions_total.labels("abc123", "challenger").inc()
    metrics.prediction_latency_seconds.labels("abc123", "challenger").observe(0.002)
    metrics.prediction_errors_total.labels("abc123").inc()
    metrics.feedback_total.labels("accepted").inc()
    metrics.challenger_rejected_total.inc()


def _touch_rollout(metrics: RolloutMetrics) -> None:
    metrics.rollout_stage.labels("abc123").set(2.0)
    metrics.canary_weight.set(0.25)
    metrics.set_e_value("error_rate", "harm", math.log(7.0))
    metrics.guardrail_breaches_total.labels("error_rate").inc()
    metrics.feature_psi.labels("usage_trend").set(0.03)
    metrics.score_psi.set(0.04)
    metrics.reconcile_total.labels("advanced").inc()
    metrics.lease_held.set(1.0)


def _touch_shadow(metrics: ShadowMetrics) -> None:
    metrics.enqueued_total.inc()
    metrics.dropped_total.inc()
    metrics.failed_total.labels("timeout").inc()
    metrics.delivered_total.inc()
    metrics.queue_depth.set(3.0)


def _samples(registry: CollectorRegistry) -> dict[str, float]:
    """Parse the exposition text into {sample name with labels: value}."""
    parsed: dict[str, float] = {}
    for line in generate_latest(registry).decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.rpartition(" ")
        parsed[key] = float(value)
    return parsed


def test_every_documented_serving_metric_is_exported_with_its_labels() -> None:
    registry = CollectorRegistry()
    _touch_serving(ServingMetrics.create(registry))
    exported = _samples(registry)
    for name in SERVING_SAMPLES:
        assert name in exported, f"{name} is not exported any more"


def test_every_documented_rollout_metric_is_exported_with_its_value() -> None:
    registry = CollectorRegistry()
    _touch_rollout(RolloutMetrics.create(registry))
    exported = _samples(registry)
    for name, value in ROLLOUT_SAMPLES.items():
        assert name in exported, f"{name} is not exported any more"
        assert value is not None
        assert exported[name] == pytest.approx(value)


def test_every_documented_shadow_metric_is_exported_with_its_value() -> None:
    """docs/RUNBOOK_ROLLBACK.md section 6 reads shadow coverage off exactly these."""
    registry = CollectorRegistry()
    _touch_shadow(ShadowMetrics.create(registry))
    exported = _samples(registry)
    for name, value in SHADOW_SAMPLES.items():
        assert name in exported, f"{name} is not exported any more"
        assert exported[name] == pytest.approx(value)


def test_all_three_families_live_in_one_registry_without_colliding() -> None:
    """The router exports serving and shadow metrics, the controller rollout ones."""
    registry = CollectorRegistry()
    _touch_serving(ServingMetrics.create(registry))
    _touch_rollout(RolloutMetrics.create(registry))
    _touch_shadow(ShadowMetrics.create(registry))
    exported = _samples(registry)
    documented = set(SERVING_SAMPLES) | set(ROLLOUT_SAMPLES) | set(SHADOW_SAMPLES)
    assert documented <= set(exported)


def test_registering_the_same_family_twice_is_refused() -> None:
    registry = CollectorRegistry()
    ServingMetrics.create(registry)
    with pytest.raises(ValueError, match=r"[Dd]uplicat"):
        ServingMetrics.create(registry)


def test_stage_codes_cover_every_lifecycle_stage_and_are_distinct() -> None:
    assert set(STAGE_CODES) == set(Stage)
    assert sorted(STAGE_CODES.values()) == list(range(len(Stage)))


def test_a_huge_e_value_is_capped_instead_of_exported_as_infinity() -> None:
    """log_e grows without bound during a clear regression; exp() of it must stay finite."""
    registry = CollectorRegistry()
    metrics = RolloutMetrics.create(registry)
    metrics.set_e_value("error_rate", "harm", 10_000.0)
    capped = _samples(registry)['app_e_value{guardrail="error_rate",test="harm"}']
    assert math.isfinite(capped)
    assert capped == pytest.approx(1e12)


def test_a_guardrail_that_was_never_evaluated_is_absent_rather_than_zero() -> None:
    """Absent means "no verdict"; a zero e-value would read as a confident verdict on a chart."""
    registry = CollectorRegistry()
    metrics = RolloutMetrics.create(registry)
    metrics.set_e_value("error_rate", "harm", 0.0)
    exported = _samples(registry)
    assert 'app_e_value{guardrail="error_rate",test="harm"}' in exported
    assert 'app_e_value{guardrail="log_loss",test="harm"}' not in exported
