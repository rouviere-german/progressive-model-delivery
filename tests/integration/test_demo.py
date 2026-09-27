"""The offline demo: the claims the README makes about it, checked against a real run.

This runs the bundled demo policy exactly as ``uv run python tasks.py demo`` does, once per
module (about a minute and a half of real HTTP traffic against a real router, a real shadow worker
and the real controller), and then asserts every statement the README and CLAIMS.md make about it.
Nothing here is a shortcut: a faster policy cannot reach the subgroup verdict, because detecting a
5 pp regression in a 12% subgroup at the Bonferroni-corrected threshold genuinely needs a few
thousand paired requests, and that cost is part of the claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from progressive_model_delivery import demo as demo_module
from progressive_model_delivery.control.controller import ReconcileOutcome
from progressive_model_delivery.control.rbac import Principal
from progressive_model_delivery.core.errors import AppError
from progressive_model_delivery.demo import DemoExpectationError, build_environment, run_demo

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """One run of the bundled demo policy, shared by the assertions below."""
    return run_demo(tmp_path_factory.mktemp("demo") / "run")


def test_the_subgroup_regressed_challenger_is_rolled_back_before_it_serves_anyone(
    demo_run: dict[str, object],
) -> None:
    scenario = demo_run["rollback_scenario"]
    assert isinstance(scenario, dict)
    assert scenario["challenger"] == "subgroup_worse"
    assert scenario["outcome"] == str(ReconcileOutcome.ROLLED_BACK)
    assert scenario["status"] == "rolled_back"
    assert scenario["lifecycle"] == "rolled_back"
    assert scenario["stages"] == ["shadow"]  # never reached a canary stage
    assert scenario["challenger_served_requests"] == 0
    crossed = scenario["harm_tests_crossed"]
    assert isinstance(crossed, list)
    assert crossed
    labels = {str(item["guardrail"]) for item in crossed}
    assert "worst_subgroup_error[south]" in labels
    for item in crossed:
        assert float(item["log_e"]) >= float(item["threshold"])
    assert str(scenario["decision_reason"]).startswith("guardrail breach:")


def test_delayed_labels_really_were_delayed_and_some_were_lost(
    demo_run: dict[str, object],
) -> None:
    scenario = demo_run["rollback_scenario"]
    assert isinstance(scenario, dict)
    accepted = int(scenario["labels_accepted"])
    requests = int(scenario["requests"])
    assert 0 < accepted < requests  # not every request ever gets a label


def test_the_better_challenger_walks_every_stage_and_waits_for_a_human(
    demo_run: dict[str, object],
) -> None:
    scenario = demo_run["promotion_scenario"]
    assert isinstance(scenario, dict)
    assert scenario["challenger"] == "better"
    assert scenario["outcome"] == str(ReconcileOutcome.PROMOTION_REQUESTED)
    assert scenario["status"] == "promoted"
    assert scenario["lifecycle"] == "production"
    assert scenario["canary_weights"] == [1.0, 5.0, 25.0, 50.0]
    assert int(scenario["challenger_served_requests"]) > 0
    proven = scenario["non_inferiority_crossed"]
    assert isinstance(proven, list)
    assert {str(item["guardrail"]) for item in proven} == {"error_rate", "log_loss"}
    refusals = scenario["refused_approvals"]
    assert refusals == ["forbidden: the proposer cannot approve their own rollout"]
    assert scenario["approved_by"] == "ravi"


def test_the_champion_changes_only_after_the_approval(demo_run: dict[str, object]) -> None:
    assert demo_run["champion_changed"] is True
    assert demo_run["champion_before"] != demo_run["champion_after"]
    promotion = demo_run["promotion_scenario"]
    assert isinstance(promotion, dict)
    assert promotion["champion_after"] == demo_run["champion_after"]


def test_the_audit_chain_records_and_verifies_the_whole_run(
    demo_run: dict[str, object],
) -> None:
    audit = demo_run["audit"]
    assert isinstance(audit, dict)
    assert audit["verified"] is True
    assert audit["actions"] == [
        "champion_bootstrapped",
        "rollout_proposed",
        "rolled_back",
        "rollout_proposed",
        "stage_advanced",
        "stage_advanced",
        "stage_advanced",
        "stage_advanced",
        "promotion_requested",
        "promoted",
    ]
    exported = Path(str(audit["path"]))
    lines = exported.read_text(encoding="utf-8").splitlines()
    assert len(lines) == int(audit["entries"])
    entries = [json.loads(line) for line in lines]
    assert entries[-1]["hash"] == audit["head"]
    assert [entry["seq"] for entry in entries] == list(range(len(entries)))
    # Every controller decision carries the spec hash it was made under. The approval entry
    # nests the controller's promotion-request evidence one level deeper.
    spec_hash = demo_run["spec"]["sha256"]  # type: ignore[index]
    controller_evidence = 0
    for entry in entries:
        evidence = entry["data"].get("evidence")
        if not isinstance(evidence, dict):
            continue
        if "evidence" in evidence:  # the "promoted" entry
            assert isinstance(evidence["promotion_request_seq"], int)
            evidence = evidence["evidence"]
        if not evidence:
            continue
        assert evidence["spec_sha256"] == spec_hash
        controller_evidence += 1
    assert controller_evidence >= 6


def test_the_summary_is_json_serializable_and_names_its_policy(
    demo_run: dict[str, object],
) -> None:
    assert json.loads(json.dumps(demo_run, sort_keys=True)) == demo_run
    spec = demo_run["spec"]
    assert isinstance(spec, dict)
    assert spec["name"] == "churn-risk-demo"
    assert len(str(spec["sha256"])) == 64
    assert float(demo_run["simulated_hours"]) > 0.0  # type: ignore[arg-type]


def test_the_demo_fails_loudly_when_a_scenario_does_not_behave(tmp_path: Path) -> None:
    """The demo asserts the behaviour this repository claims rather than printing whatever
    happened."""
    env = build_environment(tmp_path / "unexpected", spec_name="demo")
    env.service.bootstrap(demo_module.RELEASE_MANAGER, env.release("champion"))
    with pytest.raises(DemoExpectationError, match="expected the controller to report"):
        demo_module._expect(ReconcileOutcome.HOLD, ReconcileOutcome.ROLLED_BACK, "scenario")
    with pytest.raises(AppError):  # a viewer cannot propose
        env.service.propose(Principal("vic", frozenset()), env.release("better"), env.spec)
