"""RolloutSpec: the versioned YAML contract that drives the controller.

A spec lists the stages (one shadow stage, then strictly increasing canary weights from
1/5/25/50%), per-stage minimum samples and durations, the guardrails and the error budget. The
parsed spec is hashed (canonical JSON), and the hash is stored with every rollout and every audit
entry, so a decision can always be traced to the exact spec that produced it.

Error control
-------------
* Rollback: every harm test (one per guardrail, one per subgroup for subgroup guardrails) runs at
  ``alpha_rollback / K`` (Bonferroni over e-values), so the probability that a challenger that is
  no worse on every guardrail is ever rolled back is at most ``alpha_rollback``.
* Promotion: every non-inferiority test (and the optional superiority test) must reject at
  ``alpha_promotion`` (intersection-union: no correction needed), so a challenger that is
  inferior on any tested guardrail is promoted with probability at most ``alpha_promotion``.

``prior_strength`` is the total mass of the beta mixing prior on the Bernoulli guardrails. It
never affects validity (every prior yields a valid e-value) but it does decide *which* effect
sizes the test has power against: a large strength concentrates the prior near the null and buys
power for very small regressions at the cost of moderate ones. The default of 10 is tuned for the
regressions this system is meant to stop (see ``docs/adr/0004-mixture-prior-strength.md``).
"""

from __future__ import annotations

from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from progressive_model_delivery.core.canonical import canonical_json, sha256_hex
from progressive_model_delivery.registry.lifecycle import CanaryWeight

__all__ = [
    "MAX_SPEC_BYTES",
    "DriftSpec",
    "GuardrailSpec",
    "Metric",
    "PromotionSpec",
    "RolloutSpec",
    "StageKind",
    "StageSpec",
    "bundled_spec",
    "load_spec",
]

MAX_SPEC_BYTES = 64 * 1024


class StageKind(StrEnum):
    SHADOW = "shadow"
    CANARY = "canary"


class Metric(StrEnum):
    ERROR = "error"
    LATENCY_SLO = "latency_slo"
    LOG_LOSS = "log_loss"


class StageSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: StageKind
    weight_percent: Literal[1, 5, 25, 50] | None = None
    min_samples: int = Field(ge=0, le=10_000_000)
    min_duration_seconds: float = Field(ge=0)
    max_duration_seconds: float = Field(default=7 * 86_400, gt=0)
    require_non_inferiority: bool = False

    @model_validator(mode="after")
    def _weight_matches_kind(self) -> Self:
        if self.kind is StageKind.CANARY and self.weight_percent is None:
            raise ValueError("canary stages need weight_percent")
        if self.kind is StageKind.SHADOW and self.weight_percent is not None:
            raise ValueError("the shadow stage serves no traffic and has no weight")
        if self.max_duration_seconds < self.min_duration_seconds:
            raise ValueError("max_duration_seconds must be >= min_duration_seconds")
        return self

    @property
    def weight_bp(self) -> int:
        return 0 if self.weight_percent is None else int(CanaryWeight(self.weight_percent * 100))

    @property
    def weight(self) -> float:
        return self.weight_bp / 10_000


class GuardrailSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    metric: Metric
    subgroups: bool = False
    slo_ms: float | None = Field(default=None, gt=0)
    non_inferiority_ratio: float | None = Field(default=None, gt=1.0, le=3.0)
    non_inferiority_margin: float | None = Field(default=None, gt=0.0, le=1.0)
    mixture_scale: float = Field(default=0.05, gt=0.0)
    min_samples: int = Field(default=200, ge=2)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.metric is Metric.LATENCY_SLO and self.slo_ms is None:
            raise ValueError("latency_slo guardrails need slo_ms")
        if self.metric is not Metric.LATENCY_SLO and self.slo_ms is not None:
            raise ValueError("slo_ms only applies to latency_slo guardrails")
        if self.metric is Metric.LOG_LOSS and self.non_inferiority_ratio is not None:
            raise ValueError("log_loss uses an additive non_inferiority_margin")
        if self.metric is not Metric.LOG_LOSS and self.non_inferiority_margin is not None:
            raise ValueError("Bernoulli guardrails use a relative non_inferiority_ratio")
        if self.subgroups and self.metric is not Metric.ERROR:
            raise ValueError("subgroup guardrails are defined for the error metric")
        if self.subgroups and (self.non_inferiority_ratio or self.non_inferiority_margin):
            raise ValueError("subgroup guardrails are harm-only (worst-subgroup rollback)")
        return self

    @property
    def has_non_inferiority(self) -> bool:
        return self.non_inferiority_ratio is not None or self.non_inferiority_margin is not None


class DriftSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_psi_threshold: float = Field(default=0.2, gt=0)
    score_psi_threshold: float = Field(default=0.25, gt=0)
    min_samples: int = Field(default=500, ge=50)


class PromotionSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    require_superiority_on: str | None = None


class RolloutSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    version: int = Field(ge=1)
    alpha_rollback: float = Field(default=0.025, gt=0, lt=0.5)
    alpha_promotion: float = Field(default=0.025, gt=0, lt=0.5)
    label_window_seconds: float = Field(gt=0, le=30 * 86_400)
    prior_strength: float = Field(default=10.0, gt=0)
    stages: tuple[StageSpec, ...] = Field(min_length=1, max_length=8)
    guardrails: tuple[GuardrailSpec, ...] = Field(min_length=1, max_length=16)
    drift: DriftSpec = DriftSpec()
    promotion: PromotionSpec = PromotionSpec()

    @model_validator(mode="after")
    def _valid_plan(self) -> Self:
        kinds = [stage.kind for stage in self.stages]
        if kinds[0] is not StageKind.SHADOW or StageKind.SHADOW in kinds[1:]:
            raise ValueError(
                "a rollout starts with exactly one shadow stage (min_samples may be 0)"
            )
        weights = [stage.weight_bp for stage in self.stages if stage.kind is StageKind.CANARY]
        if not weights:
            raise ValueError("a rollout needs at least one canary stage before promotion")
        if weights != sorted(set(weights)):
            raise ValueError("canary weights must strictly increase")
        names = [guardrail.name for guardrail in self.guardrails]
        if len(names) != len(set(names)):
            raise ValueError("guardrail names must be unique")
        target = self.promotion.require_superiority_on
        if target is not None and target not in names:
            raise ValueError("require_superiority_on must name a guardrail")
        if target is not None and next(g for g in self.guardrails if g.name == target).subgroups:
            raise ValueError("superiority cannot be required on a subgroup guardrail")
        return self

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def guardrail(self, name: str) -> GuardrailSpec:
        for guardrail in self.guardrails:
            if guardrail.name == name:
                return guardrail
        raise KeyError(name)


def load_spec(source: Path | str) -> RolloutSpec:
    """Parse a spec from a YAML file path or YAML text (``yaml.safe_load`` only)."""
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    if len(text.encode("utf-8")) > MAX_SPEC_BYTES:
        raise ValueError("rollout spec is too large")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError("a rollout spec must be a YAML mapping")
    return RolloutSpec.model_validate(data)


def bundled_spec(name: str) -> RolloutSpec:
    """A spec shipped with the package (``standard`` or ``demo``)."""
    if name not in {"standard", "demo"}:
        raise ValueError(f"unknown bundled spec {name!r}")
    text = resources.files("progressive_model_delivery.specs").joinpath(f"{name}.yaml")
    return load_spec(text.read_text(encoding="utf-8"))
