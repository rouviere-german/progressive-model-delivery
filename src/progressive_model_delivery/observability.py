"""Prometheus metrics for serving and rollouts (per-app registries, no global state).

Serving: ``predictions_total{release,arm}``, ``prediction_latency_seconds{release,arm}``,
``prediction_errors_total{release}``, ``feedback_total{outcome}``, ``challenger_rejected_total``.
Rollouts: ``rollout_stage{release}`` (0 staged, 1 shadow, 2 canary, 3 production, 4 retired,
5 rolled back), ``canary_weight``, ``e_value{guardrail,test}``, ``guardrail_breaches_total``,
``feature_psi{feature}``, ``score_psi``, ``reconcile_total{outcome}`` and ``lease_held``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from progressive_model_delivery.registry.lifecycle import Stage

__all__ = ["STAGE_CODES", "RolloutMetrics", "ServingMetrics"]

STAGE_CODES = {
    Stage.STAGED: 0,
    Stage.SHADOW: 1,
    Stage.CANARY: 2,
    Stage.PRODUCTION: 3,
    Stage.RETIRED: 4,
    Stage.ROLLED_BACK: 5,
}
_LATENCY_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0)


@dataclass(frozen=True, slots=True)
class ServingMetrics:
    predictions_total: Counter
    prediction_latency_seconds: Histogram
    prediction_errors_total: Counter
    feedback_total: Counter
    challenger_rejected_total: Counter

    @classmethod
    def create(cls, registry: CollectorRegistry, namespace: str = "app") -> ServingMetrics:
        return cls(
            predictions_total=Counter(
                "predictions_total",
                "Predictions served, by model release (short id) and arm.",
                ["release", "arm"],
                namespace=namespace,
                registry=registry,
            ),
            prediction_latency_seconds=Histogram(
                "prediction_latency_seconds",
                "Model scoring latency by release and arm.",
                ["release", "arm"],
                namespace=namespace,
                registry=registry,
                buckets=_LATENCY_BUCKETS,
            ),
            prediction_errors_total=Counter(
                "prediction_errors_total",
                "Scoring failures by release (the champion answered instead).",
                ["release"],
                namespace=namespace,
                registry=registry,
            ),
            feedback_total=Counter(
                "feedback_total",
                "Label submissions by outcome.",
                ["outcome"],
                namespace=namespace,
                registry=registry,
            ),
            challenger_rejected_total=Counter(
                "challenger_rejected_total",
                "Challenger releases that failed verification (served champion-only).",
                namespace=namespace,
                registry=registry,
            ),
        )


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    rollout_stage: Gauge
    canary_weight: Gauge
    e_value: Gauge
    guardrail_breaches_total: Counter
    feature_psi: Gauge
    score_psi: Gauge
    reconcile_total: Counter
    lease_held: Gauge

    @classmethod
    def create(cls, registry: CollectorRegistry, namespace: str = "app") -> RolloutMetrics:
        return cls(
            rollout_stage=Gauge(
                "rollout_stage",
                "Lifecycle stage code of a release (see observability.STAGE_CODES).",
                ["release"],
                namespace=namespace,
                registry=registry,
            ),
            canary_weight=Gauge(
                "canary_weight",
                "Share of traffic on the challenger.",
                namespace=namespace,
                registry=registry,
            ),
            e_value=Gauge(
                "e_value",
                "Current e-value per guardrail test (capped at 1e12).",
                ["guardrail", "test"],
                namespace=namespace,
                registry=registry,
            ),
            guardrail_breaches_total=Counter(
                "guardrail_breaches_total",
                "Harm e-values that crossed their rollback threshold.",
                ["guardrail"],
                namespace=namespace,
                registry=registry,
            ),
            feature_psi=Gauge(
                "feature_psi",
                "PSI of served traffic against the champion's training reference.",
                ["feature"],
                namespace=namespace,
                registry=registry,
            ),
            score_psi=Gauge(
                "score_psi",
                "PSI of challenger scores against the challenger's own holdout reference.",
                namespace=namespace,
                registry=registry,
            ),
            reconcile_total=Counter(
                "reconcile_total",
                "Reconciliation passes by outcome.",
                ["outcome"],
                namespace=namespace,
                registry=registry,
            ),
            lease_held=Gauge(
                "lease_held",
                "1 while this controller holds the lease.",
                namespace=namespace,
                registry=registry,
            ),
        )

    def set_e_value(self, guardrail: str, test: str, log_e: float) -> None:
        self.e_value.labels(guardrail, test).set(math.exp(min(log_e, math.log(1e12))))
