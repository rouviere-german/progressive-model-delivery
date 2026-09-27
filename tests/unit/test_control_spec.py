"""RolloutSpec: the plan and guardrail combinations a policy may and may not express."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from progressive_model_delivery.control.spec import (
    MAX_SPEC_BYTES,
    GuardrailSpec,
    Metric,
    RolloutSpec,
    StageKind,
    StageSpec,
    bundled_spec,
    load_spec,
)

pytestmark = pytest.mark.unit


def _spec(**overrides: object) -> RolloutSpec:
    base: dict[str, object] = {
        "name": "unit-policy",
        "version": 1,
        "label_window_seconds": 60.0,
        "stages": (
            StageSpec(kind=StageKind.SHADOW, min_samples=10, min_duration_seconds=0.0),
            StageSpec(
                kind=StageKind.CANARY,
                weight_percent=5,
                min_samples=10,
                min_duration_seconds=0.0,
            ),
        ),
        "guardrails": (GuardrailSpec(name="error_rate", metric=Metric.ERROR),),
    }
    base.update(overrides)
    return RolloutSpec.model_validate(base)


def test_bundled_specs_parse_and_hash_stably() -> None:
    standard, demo = bundled_spec("standard"), bundled_spec("demo")
    assert standard.name == "churn-risk-standard"
    assert [stage.weight_percent for stage in standard.stages] == [None, 1, 5, 25, 50]
    assert standard.sha256() == bundled_spec("standard").sha256()
    assert standard.sha256() != demo.sha256()
    assert {g.name for g in standard.guardrails} == {
        "error_rate",
        "worst_subgroup_error",
        "latency_slo",
        "log_loss",
    }
    assert standard.guardrail("latency_slo").slo_ms == 25
    with pytest.raises(KeyError):
        standard.guardrail("nope")
    with pytest.raises(ValueError, match="unknown bundled spec"):
        bundled_spec("production")


def test_stage_weights_must_be_canary_weights_and_strictly_increase() -> None:
    with pytest.raises(ValidationError, match="canary stages need weight_percent"):
        StageSpec(kind=StageKind.CANARY, min_samples=1, min_duration_seconds=0.0)
    with pytest.raises(ValidationError, match="serves no traffic"):
        StageSpec(
            kind=StageKind.SHADOW, weight_percent=5, min_samples=1, min_duration_seconds=0.0
        )
    with pytest.raises(ValidationError, match="max_duration_seconds"):
        StageSpec(
            kind=StageKind.CANARY,
            weight_percent=5,
            min_samples=1,
            min_duration_seconds=100.0,
            max_duration_seconds=10.0,
        )
    with pytest.raises(ValidationError):  # 10% is not one of 1/5/25/50
        StageSpec(
            kind=StageKind.CANARY, weight_percent=10, min_samples=1, min_duration_seconds=0.0
        )
    canary = StageSpec(
        kind=StageKind.CANARY, weight_percent=25, min_samples=1, min_duration_seconds=0.0
    )
    assert (canary.weight_bp, canary.weight) == (2_500, 0.25)
    shadow = StageSpec(kind=StageKind.SHADOW, min_samples=1, min_duration_seconds=0.0)
    assert (shadow.weight_bp, shadow.weight) == (0, 0.0)

    decreasing = (
        StageSpec(kind=StageKind.SHADOW, min_samples=1, min_duration_seconds=0.0),
        StageSpec(kind=StageKind.CANARY, weight_percent=25, min_samples=1, min_duration_seconds=0),
        StageSpec(kind=StageKind.CANARY, weight_percent=5, min_samples=1, min_duration_seconds=0),
    )
    with pytest.raises(ValidationError, match="strictly increase"):
        _spec(stages=decreasing)


def test_a_plan_needs_one_leading_shadow_stage_and_a_canary_stage() -> None:
    only_canary = (
        StageSpec(kind=StageKind.CANARY, weight_percent=1, min_samples=1, min_duration_seconds=0),
    )
    with pytest.raises(ValidationError, match="starts with exactly one shadow stage"):
        _spec(stages=only_canary)
    two_shadows = (
        StageSpec(kind=StageKind.SHADOW, min_samples=1, min_duration_seconds=0.0),
        StageSpec(kind=StageKind.SHADOW, min_samples=1, min_duration_seconds=0.0),
        StageSpec(kind=StageKind.CANARY, weight_percent=1, min_samples=1, min_duration_seconds=0),
    )
    with pytest.raises(ValidationError, match="starts with exactly one shadow stage"):
        _spec(stages=two_shadows)
    shadow_only = (StageSpec(kind=StageKind.SHADOW, min_samples=1, min_duration_seconds=0.0),)
    with pytest.raises(ValidationError, match="at least one canary stage"):
        _spec(stages=shadow_only)


def test_guardrail_metric_and_margin_combinations_are_checked() -> None:
    with pytest.raises(ValidationError, match="need slo_ms"):
        GuardrailSpec(name="latency", metric=Metric.LATENCY_SLO)
    with pytest.raises(ValidationError, match="only applies"):
        GuardrailSpec(name="err", metric=Metric.ERROR, slo_ms=10)
    with pytest.raises(ValidationError, match="additive"):
        GuardrailSpec(name="ll", metric=Metric.LOG_LOSS, non_inferiority_ratio=1.1)
    with pytest.raises(ValidationError, match="relative"):
        GuardrailSpec(name="err", metric=Metric.ERROR, non_inferiority_margin=0.1)
    with pytest.raises(ValidationError, match="defined for the error metric"):
        GuardrailSpec(name="ll", metric=Metric.LOG_LOSS, subgroups=True)
    with pytest.raises(ValidationError, match="harm-only"):
        GuardrailSpec(
            name="sub", metric=Metric.ERROR, subgroups=True, non_inferiority_ratio=1.1
        )
    assert GuardrailSpec(name="a", metric=Metric.ERROR).has_non_inferiority is False
    assert GuardrailSpec(
        name="a", metric=Metric.ERROR, non_inferiority_ratio=1.1
    ).has_non_inferiority


def test_duplicate_guardrails_and_bad_superiority_targets_are_refused() -> None:
    twice = (
        GuardrailSpec(name="error_rate", metric=Metric.ERROR),
        GuardrailSpec(name="error_rate", metric=Metric.LOG_LOSS),
    )
    with pytest.raises(ValidationError, match="unique"):
        _spec(guardrails=twice)
    with pytest.raises(ValidationError, match="must name a guardrail"):
        _spec(promotion={"require_superiority_on": "missing"})
    subgroup = (
        GuardrailSpec(name="error_rate", metric=Metric.ERROR),
        GuardrailSpec(name="worst", metric=Metric.ERROR, subgroups=True),
    )
    with pytest.raises(ValidationError, match="superiority cannot be required"):
        _spec(guardrails=subgroup, promotion={"require_superiority_on": "worst"})
    ok = _spec(promotion={"require_superiority_on": "error_rate"})
    assert ok.promotion.require_superiority_on == "error_rate"


def test_the_spec_hash_changes_with_every_field_that_affects_decisions() -> None:
    base = _spec()
    assert base.sha256() != _spec(alpha_rollback=0.01).sha256()
    assert base.sha256() != _spec(prior_strength=25.0).sha256()
    assert base.sha256() != _spec(version=2).sha256()
    assert base.sha256() == _spec().sha256()


def test_load_spec_accepts_yaml_text_and_refuses_junk(tmp_path: object) -> None:
    text = yaml.safe_dump(bundled_spec("demo").model_dump(mode="json"))
    assert load_spec(text).sha256() == bundled_spec("demo").sha256()
    with pytest.raises(TypeError, match="YAML mapping"):
        load_spec("- 1\n- 2\n")
    with pytest.raises(ValueError, match="too large"):
        load_spec("name: x\n" + "#" * (MAX_SPEC_BYTES + 1))
    with pytest.raises(ValidationError):
        load_spec("name: x\nversion: 1\n")


def test_extra_fields_are_refused_so_typos_never_pass_silently() -> None:
    with pytest.raises(ValidationError):
        load_spec(
            yaml.safe_dump({**bundled_spec("demo").model_dump(mode="json"), "alpha": 0.05})
        )
