#!/usr/bin/env python3
"""How reliably, how fast and at what exposure does the controller catch a bad challenger?

For each regression the spec names, this reports:

* **detection** - the share of rollouts rolled back within the horizon;
* **looks** - the median number of reconciliation passes before the rollback;
* **exposure** - the median number of requests the challenger served before being rolled back.
  Zero means the shadow stage caught it, so no customer ever saw it;
* **canary-only exposure** - the same rollout for a challenger that cannot be mirrored (a model
  with side effects), which is what shadow testing buys;
* **fixed horizon** - the total requests a one-sided fixed-horizon z-test would need at 80% power
  for the same effect, from the closed-form formula in ``stats.power``. The sequential test pays
  for validity under continuous monitoring; this is the size of that bill.

The last two scenarios are not invented effect sizes: they are the per-region error rates and
log-loss the actually trained ``subgroup_worse`` and ``feature_order_bug`` challengers reach on the
clean holdout.

Usage:
  uv run python bench/bench_power.py                 # CI size
  uv run python bench/bench_power.py --check         # also enforce bench/thresholds.json
  uv run python bench/bench_power.py --rollouts 2000 # the full study (`make study`)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from progressive_model_delivery.control.spec import Metric
from progressive_model_delivery.models.features import REGIONS
from progressive_model_delivery.models.pipeline import build_demo_models
from progressive_model_delivery.registry.signing import SigningKey
from progressive_model_delivery.sim import scenarios
from progressive_model_delivery.sim.scenarios import Scenario
from progressive_model_delivery.sim.study import RolloutStudy, StudyResult, study_spec
from progressive_model_delivery.stats.power import fixed_horizon_sample_size

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.harness import (  # noqa: E402 - the repo root has to be importable first
    check_thresholds,
    machine,
    print_table,
    write_results,
)

LOOK_SIZE = 2_000
MAX_LOOKS = 60
SEED = 20_260_928
MEASURED_MODELS = ("champion", "subgroup_worse", "feature_order_bug", "better")


def measured_scenarios() -> tuple[dict[str, Scenario], Scenario]:
    """Train the demo catalogue and read the real effect sizes off the clean holdout.

    Returns the two pipeline-bug challengers as regressions, plus the genuinely better model as
    the promotion scenario.
    """
    with tempfile.TemporaryDirectory() as tmp:
        models = build_demo_models(
            Path(tmp),
            SigningKey.generate(),
            n_train=20_000,
            n_holdout=20_000,
            names=MEASURED_MODELS,
        )
        live = {name: dict(models.live_metrics[name]) for name in MEASURED_MODELS}
    regressions = {
        name: scenarios.measured_scenario(f"measured_{name}", live["champion"], live[name])
        for name in ("subgroup_worse", "feature_order_bug")
    }
    return regressions, scenarios.measured_scenario(
        "measured_better", live["champion"], live["better"]
    )


def _study(
    scenario: Scenario, rollouts: int, *, seed: int, canary_only: bool = False
) -> StudyResult:
    spec = study_spec(shadow_samples=0) if canary_only else study_spec()
    return RolloutStudy(
        spec,
        scenario,
        rollouts=rollouts,
        look_size=LOOK_SIZE,
        max_looks=MAX_LOOKS,
        seed=seed,
        shadow_traffic=not canary_only,
    ).run()


def _fixed_horizon(scenario: Scenario) -> int | None:
    """Total requests a fixed-horizon z-test needs at 80% power for the worst subgroup."""
    gap = scenario.worst_subgroup_gap()
    if gap <= 0:
        return None
    index = max(
        range(len(REGIONS)),
        key=lambda i: scenario.challenger.error[i] - scenario.champion.error[i],
    )
    base = scenario.champion.error[index]
    share = 0.12 if REGIONS[index] == "south" else 1.0
    try:
        per_subgroup = fixed_horizon_sample_size(base, gap, alpha=0.025, power=0.8, allocation=0.5)
    except ValueError:
        return None
    return int(per_subgroup / share)


def run(rollouts: int, *, with_models: bool) -> dict[str, Any]:
    spec = study_spec()
    slo_ms = next(g.slo_ms for g in spec.guardrails if g.metric is Metric.LATENCY_SLO and g.slo_ms)
    catalogue: dict[str, Scenario] = {
        "error_+1pp": scenarios.error_regression(1.0, slo_ms=slo_ms),
        "error_+2pp": scenarios.error_regression(2.0, slo_ms=slo_ms),
        "error_+5pp": scenarios.error_regression(5.0, slo_ms=slo_ms),
        "latency_+20ms": scenarios.latency_regression(20.0, slo_ms=slo_ms),
        "subgroup_south_+5pp": scenarios.subgroup_regression("south", 5.0, slo_ms=slo_ms),
    }
    promotion = scenarios.better(delta_pp=2.0, slo_ms=slo_ms)
    if with_models:
        regressions, promotion = measured_scenarios()
        catalogue.update(regressions)

    started = time.perf_counter()
    rows: dict[str, Any] = {}
    for index, (name, scenario) in enumerate(catalogue.items()):
        shadowed = _study(scenario, rollouts, seed=SEED + index)
        canary_only = _study(scenario, rollouts, seed=SEED + index, canary_only=True)
        rows[name] = {
            "scenario": scenario.name,
            "worst_subgroup_gap_pp": round(scenario.worst_subgroup_gap() * 100, 3),
            "detection_rate": round(shadowed.rollback_rate, 4),
            "promotion_rate": round(shadowed.promotion_rate, 4),
            "median_looks_to_rollback": shadowed.median_looks_to_rollback,
            "median_exposure": shadowed.median_exposure_to_rollback,
            "canary_only_detection_rate": round(canary_only.rollback_rate, 4),
            "canary_only_median_exposure": canary_only.median_exposure_to_rollback,
            "fixed_horizon_requests_80pct": _fixed_horizon(scenario),
            "breaches": shadowed.breaches,
            "full": shadowed.to_json(),
        }
    good = _study(promotion, rollouts, seed=SEED + 100)
    elapsed = time.perf_counter() - started
    return {
        "rollouts": rollouts,
        "requests_per_look": LOOK_SIZE,
        "max_looks": MAX_LOOKS,
        "seconds": round(elapsed, 1),
        "scenarios": rows,
        "better_challenger": {
            "scenario": promotion.name,
            "promotion_rate": round(good.promotion_rate, 4),
            "rollback_rate": round(good.rollback_rate, 4),
            "median_looks_to_promotion": good.median_looks_to_promotion,
            "full": good.to_json(),
        },
    }


def metrics(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = results["scenarios"]
    return {
        "detection": {f"minus_{name}": -float(row["detection_rate"]) for name, row in rows.items()},
        "exposure": {
            f"{name}_median": float(row["median_exposure"] or 0.0) for name, row in rows.items()
        },
        "better_challenger": {
            "minus_promotion_rate": -float(results["better_challenger"]["promotion_rate"]),
            "rollback_rate": float(results["better_challenger"]["rollback_rate"]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--rollouts", type=int, default=400)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="skip the scenarios whose effect sizes come from trained models",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = run(args.rollouts, with_models=not args.no_models)
    write_results("power", {"machine": machine(), "results": results})
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))

    print(
        f"{results['rollouts']} rollouts per scenario, up to {results['max_looks']} looks of "
        f"{results['requests_per_look']} requests, in {results['seconds']} s"
    )
    print_table(
        [
            "scenario",
            "detected",
            "median looks",
            "median exposure",
            "canary-only exposure",
            "fixed horizon (80%)",
        ],
        [
            [
                name,
                f"{row['detection_rate']:.3f}",
                "-"
                if row["median_looks_to_rollback"] is None
                else f"{row['median_looks_to_rollback']:.0f}",
                "-" if row["median_exposure"] is None else f"{row['median_exposure']:.0f}",
                "-"
                if row["canary_only_median_exposure"] is None
                else f"{row['canary_only_median_exposure']:.0f}",
                "-"
                if row["fixed_horizon_requests_80pct"] is None
                else f"{row['fixed_horizon_requests_80pct']:,}",
            ]
            for name, row in results["scenarios"].items()
        ],
    )
    good = results["better_challenger"]
    print(
        f"{good['scenario']}: promoted {good['promotion_rate']:.3f}, rolled back "
        f"{good['rollback_rate']:.3f}, median looks to promotion "
        f"{good['median_looks_to_promotion']}"
    )
    if args.check:
        return check_thresholds("power", metrics(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
