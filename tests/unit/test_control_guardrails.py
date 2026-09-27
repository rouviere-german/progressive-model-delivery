"""Guardrail evaluation and the stage rule, including the invariants that must always hold.

The property tests state the safety properties of the decision layer directly:

* harm dominates: whenever a harm e-value has crossed, the decision is ROLLBACK, whatever else
  the stage rule would have concluded;
* a rollout never advances before its stage minimums are met;
* the last stage is never left without promotion evidence;
* evidence is monotone in the right direction (more challenger-only failures can only raise the
  harm e-value, never lower it);
* thresholds follow the spec's error budget (Bonferroni over harm tests, intersection-union for
  promotion).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from progressive_model_delivery.control.guardrails import (
    ArmCounts,
    ArmMoments,
    Decision,
    PairedCounts,
    PairedMoments,
    StageEvidence,
    decide,
    evaluate,
)
from progressive_model_delivery.control.spec import (
    GuardrailSpec,
    Metric,
    RolloutSpec,
    StageKind,
    StageSpec,
    bundled_spec,
)
from progressive_model_delivery.models.features import REGIONS
from progressive_model_delivery.stats.msprt import Moments

pytestmark = pytest.mark.unit

GROUPS = len(REGIONS)
SPEC = bundled_spec("standard")


def _cell(value: float) -> np.ndarray:
    return np.full((GROUPS,), value, dtype=np.float64)


def _moments(count: float, mean: float, sd: float) -> Moments:
    n = _cell(count)
    total = n * mean
    return Moments(n, total, n * (mean * mean + sd * sd))


def _shadow_stage(
    *,
    n01: float = 40.0,
    n10: float = 40.0,
    n00: float = 400.0,
    n11: float = 20.0,
    loss_diff_mean: float = 0.0,
    latency_names: tuple[str, ...] = ("latency_slo",),
    samples: float | None = None,
) -> StageEvidence:
    error = PairedCounts(_cell(n00), _cell(n01), _cell(n10), _cell(n11))
    latency = {
        name: PairedCounts(_cell(500.0), _cell(5.0), _cell(5.0), _cell(2.0))
        for name in latency_names
    }
    total = float((n00 + n01 + n10 + n11) * GROUPS)
    return StageEvidence(
        kind=StageKind.SHADOW,
        weight=0.0,
        error=error,
        latency=latency,
        log_loss=PairedMoments(_moments(500.0, loss_diff_mean, 0.4)),
        samples=np.asarray(total if samples is None else samples),
        exposure=np.asarray(0.0),
    )


def _canary_stage(
    *,
    weight: float = 0.25,
    n_champion: float = 900.0,
    k_champion: float = 180.0,
    n_challenger: float = 300.0,
    k_challenger: float = 60.0,
    champion_loss: float = 0.45,
    challenger_loss: float = 0.45,
    latency_names: tuple[str, ...] = ("latency_slo",),
) -> StageEvidence:
    error = ArmCounts(
        _cell(n_champion), _cell(k_champion), _cell(n_challenger), _cell(k_challenger)
    )
    latency = {
        name: ArmCounts(_cell(n_champion), _cell(9.0), _cell(n_challenger), _cell(3.0))
        for name in latency_names
    }
    return StageEvidence(
        kind=StageKind.CANARY,
        weight=weight,
        error=error,
        latency=latency,
        log_loss=ArmMoments(
            _moments(n_champion, champion_loss, 0.5), _moments(n_challenger, challenger_loss, 0.5)
        ),
        samples=np.asarray(n_challenger * GROUPS),
        exposure=np.asarray(n_challenger * GROUPS),
    )


def _harm(results: list[object], label: str) -> float:
    for result in results:  # pragma: no branch - the label always exists
        if result.guardrail == label or result.label == label:  # type: ignore[attr-defined]
            if result.test == "harm":  # type: ignore[attr-defined]
                return float(result.log_e)  # type: ignore[attr-defined]
    raise AssertionError(label)


# --- thresholds and plan ---------------------------------------------------------------------


def test_the_test_plan_and_thresholds_follow_the_spec_error_budget() -> None:
    results = evaluate(SPEC, [_shadow_stage(), _canary_stage()], REGIONS)
    harm = [r for r in results if r.test == "harm"]
    promotion = [r for r in results if r.test != "harm"]
    # 4 guardrails: error, worst-subgroup (one test per subgroup), latency, log-loss.
    assert len(harm) == 3 + GROUPS
    assert {r.label for r in harm} == {
        "error_rate",
        "latency_slo",
        "log_loss",
        *(f"worst_subgroup_error[{name}]" for name in REGIONS),
    }
    assert {r.threshold for r in harm} == {math.log(len(harm) / SPEC.alpha_rollback)}
    assert {r.label for r in promotion} == {"error_rate", "log_loss"}
    assert {r.threshold for r in promotion} == {math.log(1.0 / SPEC.alpha_promotion)}


def test_a_superiority_requirement_adds_exactly_one_promotion_test() -> None:
    spec = RolloutSpec.model_validate(
        {**SPEC.model_dump(mode="json"), "promotion": {"require_superiority_on": "log_loss"}}
    )
    results = evaluate(spec, [_shadow_stage(), _canary_stage()], REGIONS)
    assert [r.test for r in results].count("superiority") == 1
    superiority = next(r for r in results if r.test == "superiority")
    assert superiority.guardrail == "log_loss"
    assert superiority.subgroup is None


def test_evaluate_needs_evidence() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        evaluate(SPEC, [], REGIONS)


# --- direction and monotonicity --------------------------------------------------------------


def test_a_balanced_shadow_stage_gives_no_harm_evidence_in_either_direction() -> None:
    results = evaluate(SPEC, [_shadow_stage(n01=40, n10=40)], REGIONS)
    for result in results:
        if result.test == "harm":
            assert float(result.log_e) < result.threshold
    assert _harm(list(results), "error_rate") < 0.0


def test_challenger_only_failures_drive_the_harm_e_value_up_and_theirs_down() -> None:
    worse = evaluate(SPEC, [_shadow_stage(n01=90, n10=20)], REGIONS)
    better = evaluate(SPEC, [_shadow_stage(n01=20, n10=90)], REGIONS)
    assert _harm(list(worse), "error_rate") > math.log(1 / 0.025)
    assert _harm(list(better), "error_rate") < 0.0


@given(
    n01=st.integers(min_value=0, max_value=400),
    extra=st.integers(min_value=1, max_value=200),
    n10=st.integers(min_value=0, max_value=400),
)
def test_harm_evidence_is_monotone_in_challenger_only_failures(
    n01: int, extra: int, n10: int
) -> None:
    low = _harm(list(evaluate(SPEC, [_shadow_stage(n01=n01, n10=n10)], REGIONS)), "error_rate")
    high = _harm(
        list(evaluate(SPEC, [_shadow_stage(n01=n01 + extra, n10=n10)], REGIONS)), "error_rate"
    )
    assert high >= low - 1e-9


@given(
    k_challenger=st.integers(min_value=0, max_value=200),
    extra=st.integers(min_value=1, max_value=100),
)
def test_canary_harm_evidence_is_monotone_in_challenger_errors(
    k_challenger: int, extra: int
) -> None:
    assume(k_challenger + extra <= 300)
    low = _harm(
        list(evaluate(SPEC, [_shadow_stage(), _canary_stage(k_challenger=k_challenger)], REGIONS)),
        "error_rate",
    )
    high = _harm(
        list(
            evaluate(
                SPEC, [_shadow_stage(), _canary_stage(k_challenger=k_challenger + extra)], REGIONS
            )
        ),
        "error_rate",
    )
    assert high >= low - 1e-9


def test_only_the_named_subgroup_reacts_to_a_subgroup_regression() -> None:
    """South is clearly worse while the pooled table stays balanced, as a real subgroup bug is."""
    south = REGIONS.index("south")
    n01 = np.array([16.0, 160.0, 16.0, 16.0])
    n10 = np.array([64.0, 16.0, 64.0, 64.0])
    n01[south], n10[south] = 160.0, 16.0
    assert n01.sum() == pytest.approx(n10.sum(), abs=1.0)  # no pooled regression
    stage = _shadow_stage()
    regressed = StageEvidence(
        kind=stage.kind,
        weight=stage.weight,
        error=PairedCounts(_cell(400.0), n01, n10, _cell(20.0)),
        latency=stage.latency,
        log_loss=stage.log_loss,
        samples=np.asarray(float((n01 + n10).sum() + 420.0 * GROUPS)),
    )
    results = evaluate(SPEC, [regressed], REGIONS)
    crossed = [r.label for r in results if r.test == "harm" and bool(r.crossed)]
    assert crossed == ["worst_subgroup_error[south]"]
    assert _harm(list(results), "error_rate") < 0.0


def test_shadow_evidence_never_contributes_promotion_evidence() -> None:
    results = evaluate(SPEC, [_shadow_stage(n01=5, n10=200, loss_diff_mean=-0.2)], REGIONS)
    for result in results:
        if result.test != "harm":
            assert float(result.log_e) == 0.0


def test_a_clearly_better_challenger_proves_non_inferiority_on_served_traffic() -> None:
    stages = [
        _shadow_stage(),
        _canary_stage(k_challenger=30.0, challenger_loss=0.40, n_challenger=1_500.0),
    ]
    results = evaluate(SPEC, stages, REGIONS)
    for result in results:
        if result.test == "non_inferiority":
            assert bool(result.crossed), result.label


# --- the stage rule --------------------------------------------------------------------------


def _decide(**overrides: object) -> Decision:
    kwargs: dict[str, object] = {
        "stage_index": 0,
        "elapsed_seconds": 100_000.0,
        "stage_samples": 100_000.0,
        "results": [],
    }
    kwargs.update(overrides)
    outcome = decide(SPEC, **kwargs)  # type: ignore[arg-type]
    return Decision(int(outcome.decision))


def test_advancing_needs_minimum_samples_duration_and_no_drift() -> None:
    assert _decide() is Decision.ADVANCE
    assert _decide(stage_samples=1.0) is Decision.HOLD
    assert _decide(elapsed_seconds=1.0) is Decision.HOLD
    assert _decide(drift_ok=False) is Decision.HOLD


def test_the_last_stage_requests_promotion_only_with_promotion_evidence() -> None:
    last = len(SPEC.stages) - 1
    results = evaluate(SPEC, [_shadow_stage(), _canary_stage()], REGIONS)
    assert _decide(stage_index=last, results=results) is Decision.HOLD
    strong = evaluate(
        SPEC,
        [_shadow_stage(), _canary_stage(k_challenger=30.0, challenger_loss=0.40, n_challenger=1500)],
        REGIONS,
    )
    assert _decide(stage_index=last, results=strong) is Decision.REQUEST_PROMOTION


def test_a_stage_that_requires_non_inferiority_holds_without_it() -> None:
    index = next(i for i, stage in enumerate(SPEC.stages) if stage.require_non_inferiority)
    assert index < len(SPEC.stages) - 1
    weak = evaluate(SPEC, [_shadow_stage(), _canary_stage()], REGIONS)
    assert _decide(stage_index=index, results=weak) is Decision.HOLD


def test_a_stage_that_never_meets_its_gate_aborts_after_its_maximum_duration() -> None:
    long = SPEC.stages[0].max_duration_seconds + 1.0
    assert _decide(stage_samples=1.0, elapsed_seconds=long) is Decision.ABORT


def test_awaiting_approval_holds_but_never_blocks_a_rollback() -> None:
    last = len(SPEC.stages) - 1
    strong = evaluate(
        SPEC,
        [_shadow_stage(), _canary_stage(k_challenger=30.0, challenger_loss=0.40, n_challenger=1500)],
        REGIONS,
    )
    assert _decide(stage_index=last, results=strong, awaiting_approval=True) is Decision.HOLD
    harmful = evaluate(SPEC, [_shadow_stage(n01=200, n10=10)], REGIONS)
    assert (
        _decide(stage_index=last, results=harmful, awaiting_approval=True) is Decision.ROLLBACK
    )


def test_the_label_free_score_guard_rolls_back_on_its_own() -> None:
    assert _decide(score_breach=True) is Decision.ROLLBACK
    assert _decide(stage_samples=1.0, score_breach=True) is Decision.ROLLBACK


@given(
    samples=st.floats(min_value=0, max_value=1e6),
    elapsed=st.floats(min_value=0, max_value=1e7),
    stage_index=st.integers(min_value=0, max_value=4),
    awaiting=st.booleans(),
    drift_ok=st.booleans(),
    harmful=st.booleans(),
)
def test_a_crossed_harm_e_value_always_wins(
    samples: float,
    elapsed: float,
    stage_index: int,
    awaiting: bool,
    drift_ok: bool,
    harmful: bool,
) -> None:
    stages = [_shadow_stage(n01=200 if harmful else 10, n10=10 if harmful else 200)]
    results = evaluate(SPEC, stages, REGIONS)
    crossed = any(r.test == "harm" and bool(r.crossed) for r in results)
    assume(crossed == harmful)
    decision = _decide(
        stage_index=stage_index,
        elapsed_seconds=elapsed,
        stage_samples=samples,
        results=results,
        awaiting_approval=awaiting,
        drift_ok=drift_ok,
    )
    if harmful:
        assert decision is Decision.ROLLBACK
    else:
        assert decision is not Decision.ROLLBACK


@given(
    samples=st.floats(min_value=0, max_value=1e5),
    elapsed=st.floats(min_value=0, max_value=1e6),
    stage_index=st.integers(min_value=0, max_value=4),
)
def test_advancing_implies_the_stage_minimums_were_met(
    samples: float, elapsed: float, stage_index: int
) -> None:
    decision = _decide(
        stage_index=stage_index, elapsed_seconds=elapsed, stage_samples=samples, results=[]
    )
    stage = SPEC.stages[stage_index]
    if decision in (Decision.ADVANCE, Decision.REQUEST_PROMOTION):
        assert samples >= stage.min_samples
        assert elapsed >= stage.min_duration_seconds


def test_decide_is_vectorized_over_rollouts() -> None:
    spec = RolloutSpec(
        name="vector",
        version=1,
        label_window_seconds=60.0,
        stages=(
            StageSpec(kind=StageKind.SHADOW, min_samples=100, min_duration_seconds=0.0),
            StageSpec(
                kind=StageKind.CANARY, weight_percent=5, min_samples=10, min_duration_seconds=0.0
            ),
        ),
        guardrails=(GuardrailSpec(name="error_rate", metric=Metric.ERROR),),
    )
    outcome = decide(
        spec,
        stage_index=np.array([0, 0, 1]),
        elapsed_seconds=np.array([10.0, 10.0, 10.0]),
        stage_samples=np.array([1.0, 500.0, 500.0]),
        results=[],
    )
    assert outcome.decision.tolist() == [
        int(Decision.HOLD),
        int(Decision.ADVANCE),
        int(Decision.REQUEST_PROMOTION),
    ]
    assert outcome.minimums_met.tolist() == [False, True, True]
