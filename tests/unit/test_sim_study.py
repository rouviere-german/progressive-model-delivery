"""The rollout study and its data-generating process.

The properties that make the study trustworthy:

* the coupling construction has the marginals it claims for every coupling value (so an A/A
  scenario really is a null),
* the study reaches a terminal state for every rollout it decides, never two at once,
* it is deterministic in its seed,
* a large harmful effect is detected and a null effect is rarely acted on.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from progressive_model_delivery.control.spec import Metric, StageKind
from progressive_model_delivery.models.features import REGIONS
from progressive_model_delivery.sim import scenarios
from progressive_model_delivery.sim.naive import PeekingStudy
from progressive_model_delivery.sim.study import RolloutStudy, binomial_ci, study_spec

pytestmark = pytest.mark.unit


# --- the coupling ------------------------------------------------------------------------------


@given(
    p_a=st.floats(min_value=0.01, max_value=0.99),
    p_b=st.floats(min_value=0.01, max_value=0.99),
    coupling=st.floats(min_value=0.0, max_value=1.0),
)
def test_the_paired_coupling_keeps_both_marginals_for_any_dependence(
    p_a: float, p_b: float, coupling: float
) -> None:
    cells = scenarios.paired_cell_probabilities(p_a, p_b, coupling)
    _p00, p01, p10, p11 = cells
    assert float(cells.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(np.min(cells)) >= -1e-15
    assert float(p10 + p11) == pytest.approx(p_a, abs=1e-12)  # champion marginal
    assert float(p01 + p11) == pytest.approx(p_b, abs=1e-12)  # challenger marginal


def test_identical_models_give_symmetric_discordant_cells() -> None:
    cells = scenarios.paired_cell_probabilities(0.2, 0.2, 0.85)
    assert float(cells[1]) == pytest.approx(float(cells[2]))
    assert float(cells[1]) > 0.0  # a fully monotone coupling would leave nothing to test


def test_a_worse_challenger_has_more_challenger_only_failures() -> None:
    cells = scenarios.paired_cell_probabilities(0.2, 0.26, 0.85)
    assert float(cells[1]) > float(cells[2])


def test_the_coupling_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="coupling must be"):
        scenarios.paired_cell_probabilities(0.2, 0.2, 1.5)
    with pytest.raises(ValueError, match="must be in"):
        scenarios.paired_cell_probabilities(0.0, 0.2, 0.5)


# --- latency and profiles ----------------------------------------------------------------------


def test_the_latency_model_turns_a_millisecond_shift_into_an_exceedance_rate() -> None:
    model = scenarios.LatencyModel(median_ms=8.0, sigma=0.5)
    base = model.exceed_probability(25.0)
    shifted = model.exceed_probability(25.0, shift_ms=20.0)
    assert 0.0 < base < 0.05
    assert shifted > 0.5
    assert model.exceed_probability(25.0, shift_ms=30.0) == 1.0
    with pytest.raises(ValueError, match="slo_ms"):
        model.exceed_probability(0.0)
    with pytest.raises(ValueError, match="positive"):
        scenarios.LatencyModel(median_ms=0.0, sigma=1.0)


def test_arm_profiles_validate_their_rates() -> None:
    with pytest.raises(ValueError, match="one rate per subgroup"):
        scenarios.ArmProfile(error=(0.1,), latency_exceed=(0.01,) * 4, log_loss_mean=0.4)
    with pytest.raises(ValueError, match="rates must be in"):
        scenarios.ArmProfile(error=(0.0,) * 4, latency_exceed=(0.01,) * 4, log_loss_mean=0.4)
    with pytest.raises(ValueError, match="log_loss_mean"):
        scenarios.ArmProfile(error=(0.1,) * 4, latency_exceed=(0.01,) * 4, log_loss_mean=0.0)


def test_scenario_constructors_describe_what_they_change() -> None:
    aa = scenarios.identical()
    assert aa.champion == aa.challenger
    assert aa.worst_subgroup_gap() == 0.0
    assert aa.paired_log_loss_mean == 0.0
    assert aa.paired_log_loss_sd > 0.0

    regression = scenarios.error_regression(2.0)
    assert regression.worst_subgroup_gap() == pytest.approx(0.02)
    assert regression.paired_log_loss_mean > 0.0

    subgroup = scenarios.subgroup_regression("south", 6.0)
    gaps = subgroup.challenger.error_array - subgroup.champion.error_array
    assert gaps[REGIONS.index("south")] == pytest.approx(0.06)
    assert float(np.delete(gaps, REGIONS.index("south")).max()) == pytest.approx(0.0)

    latency = scenarios.latency_regression(20.0)
    assert latency.challenger.latency_exceed[0] > latency.champion.latency_exceed[0]
    assert latency.worst_subgroup_gap() == 0.0

    good = scenarios.better(delta_pp=3.0)
    assert good.worst_subgroup_gap() < 0.0
    assert good.paired_log_loss_mean < 0.0

    inferior = scenarios.inferior_beyond_margin(1.25)
    assert inferior.challenger.error[0] == pytest.approx(inferior.champion.error[0] * 1.25)


def test_scenario_constructors_reject_nonsense() -> None:
    with pytest.raises(ValueError, match="delta_pp"):
        scenarios.error_regression(0.0)
    with pytest.raises(ValueError, match="unknown region"):
        scenarios.subgroup_regression("mars", 1.0)
    with pytest.raises(ValueError, match="shift_ms"):
        scenarios.latency_regression(0.0)
    with pytest.raises(ValueError, match="ratio"):
        scenarios.inferior_beyond_margin(1.0)
    with pytest.raises(ValueError, match="delta_pp"):
        scenarios.better(delta_pp=40.0)
    with pytest.raises(ValueError, match="coupling"):
        scenarios.Scenario("x", scenarios.identical().champion, scenarios.identical().champion, 2.0)
    with pytest.raises(ValueError, match="correlation"):
        scenarios.Scenario(
            "x",
            scenarios.identical().champion,
            scenarios.identical().champion,
            log_loss_correlation=1.0,
        )


def test_a_measured_scenario_reads_per_region_error_rates() -> None:
    champion = {"log_loss": 0.29, **{f"error_rate_{r}": 0.12 for r in REGIONS}}
    challenger = {"log_loss": 0.31, **{f"error_rate_{r}": 0.12 for r in REGIONS}}
    challenger["error_rate_south"] = 0.24
    built = scenarios.measured_scenario("measured", champion, challenger)
    assert built.worst_subgroup_gap() == pytest.approx(0.12)
    assert built.challenger.log_loss_mean == pytest.approx(0.31)


# --- the study spec ----------------------------------------------------------------------------


def test_the_study_spec_is_the_production_policy_on_a_shorter_schedule() -> None:
    spec = study_spec()
    assert [stage.kind for stage in spec.stages] == [
        StageKind.SHADOW,
        *[StageKind.CANARY] * 4,
    ]
    assert [stage.weight_percent for stage in spec.stages] == [None, 1, 5, 25, 50]
    assert {g.metric for g in spec.guardrails} == {
        Metric.ERROR,
        Metric.LATENCY_SLO,
        Metric.LOG_LOSS,
    }
    assert any(g.subgroups for g in spec.guardrails)
    assert spec.guardrail("error_rate").non_inferiority_ratio == 1.10
    assert study_spec(shadow_samples=0).stages[0].min_samples == 0


# --- the study loop ----------------------------------------------------------------------------


def _study(scenario: scenarios.Scenario, **overrides: object) -> RolloutStudy:
    kwargs: dict[str, object] = {
        "rollouts": 40,
        "look_size": 2_000,
        "max_looks": 40,
        "seed": 5,
    }
    kwargs.update(overrides)
    return RolloutStudy(study_spec(), scenario, **kwargs)  # type: ignore[arg-type]


def test_every_rollout_ends_in_exactly_one_terminal_state() -> None:
    result = _study(scenarios.error_regression(5.0)).run()
    total = result.rolled_back + result.promotion_requested + result.aborted + result.open_at_end
    assert total == result.rollouts
    assert result.rolled_back == result.rollouts  # a 5pp regression is always caught
    assert result.median_looks_to_rollback is not None
    assert result.mean_exposure_to_rollback is not None
    assert "error_rate" in result.breaches or "log_loss" in result.breaches


def test_the_study_is_deterministic_in_its_seed() -> None:
    first = _study(scenarios.identical(), seed=17).run()
    again = _study(scenarios.identical(), seed=17).run()
    other = _study(scenarios.identical(), seed=18).run()
    assert first.to_json() == again.to_json()
    assert first.rollouts == other.rollouts


def test_an_aa_scenario_is_rarely_rolled_back_and_usually_promoted() -> None:
    result = _study(scenarios.identical(), rollouts=200, max_looks=90).run()
    assert result.rollback_rate <= 0.05
    assert result.promotion_rate > 0.8
    low, high = result.rollback_ci()
    assert low <= result.rollback_rate <= high


def test_a_challenger_beyond_the_non_inferiority_margin_is_not_promoted() -> None:
    result = _study(scenarios.inferior_beyond_margin(1.5), rollouts=60, max_looks=60).run()
    assert result.promotion_requested == 0
    assert result.rolled_back > 0


def test_a_subgroup_only_regression_is_caught_by_the_subgroup_guardrail() -> None:
    result = _study(scenarios.subgroup_regression("south", 8.0), rollouts=60, max_looks=60).run()
    assert result.rolled_back > 0
    assert any(label.startswith("worst_subgroup_error[south]") for label in result.breaches)


def test_a_latency_regression_is_caught_by_the_latency_guardrail() -> None:
    result = _study(scenarios.latency_regression(20.0), rollouts=40, max_looks=40).run()
    assert result.rolled_back == result.rollouts
    assert "latency_slo" in result.breaches


def test_without_shadow_traffic_a_regression_costs_real_exposure() -> None:
    spec = study_spec(shadow_samples=0)
    shadowed = RolloutStudy(
        study_spec(),
        scenarios.error_regression(2.0),
        rollouts=40,
        look_size=2_000,
        max_looks=40,
        seed=9,
    ).run()
    canary_only = RolloutStudy(
        spec,
        scenarios.error_regression(2.0),
        rollouts=40,
        look_size=2_000,
        max_looks=40,
        seed=9,
        shadow_traffic=False,
    ).run()
    assert shadowed.median_exposure_to_rollback == 0.0
    assert canary_only.rolled_back > 0
    assert canary_only.median_exposure_to_rollback is not None
    assert canary_only.median_exposure_to_rollback > 0.0


def test_an_aborting_stage_is_reported() -> None:
    """A shadow stage whose sample gate cannot be met within its window aborts the rollout."""
    from progressive_model_delivery.control.spec import RolloutSpec

    spec = study_spec(shadow_samples=10_000_000)  # unreachable at this look size
    plan = spec.model_dump(mode="json")
    plan["stages"][0]["max_duration_seconds"] = 1.0
    bounded = RolloutSpec.model_validate(plan)
    result = RolloutStudy(
        bounded, scenarios.identical(), rollouts=10, look_size=500, max_looks=3, seed=1
    ).run()
    assert result.aborted == result.rollouts


def test_the_study_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _study(scenarios.identical(), rollouts=0)
    with pytest.raises(ValueError, match="label_rate"):
        _study(scenarios.identical(), label_rate=0.0)
    with pytest.raises(ValueError, match="subgroup_shares"):
        _study(scenarios.identical(), subgroup_shares=(0.5, 0.5))


def test_binomial_ci_brackets_the_rate() -> None:
    low, high = binomial_ci(5, 100)
    assert low < 0.05 < high
    assert binomial_ci(0, 100)[0] == 0.0
    assert binomial_ci(100, 100)[1] == 1.0
    with pytest.raises(ValueError, match="trials"):
        binomial_ci(5, 0)
    with pytest.raises(ValueError, match="successes"):
        binomial_ci(11, 10)


# --- the naive baseline ------------------------------------------------------------------------


def test_peeking_inflates_the_decision_rate_far_beyond_a_single_look() -> None:
    study = PeekingStudy(
        scenarios.identical(), rollouts=400, look_size=2_000, looks=40, weight=0.5, seed=3
    )
    result = study.run()
    assert result.final_look_rate < 0.12  # close to the nominal alpha of 0.05
    assert result.peeking_rate > 0.20  # repeated looks at the same nominal alpha
    assert result.peeking_rate > 3 * result.final_look_rate
    assert result.median_look_of_first_decision is not None
    payload = result.to_json()
    assert payload["alpha"] == 0.05
    assert payload["looks"] == 40


def test_the_peeking_study_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        PeekingStudy(scenarios.identical(), rollouts=0, look_size=1, looks=1, weight=0.5, seed=1)
    with pytest.raises(ValueError, match="weight"):
        PeekingStudy(scenarios.identical(), rollouts=1, look_size=1, looks=1, weight=1.0, seed=1)
    with pytest.raises(ValueError, match="alpha"):
        PeekingStudy(
            scenarios.identical(), rollouts=1, look_size=1, looks=1, weight=0.5, alpha=0.0, seed=1
        )
