#!/usr/bin/env python3
"""Does the controller keep its error budget while someone watches every reconciliation?

Four measurements, all on the shipping ``evaluate``/``decide`` code path:

1. **A/A false rollback.** Identical champion and challenger, monitored at every look for the whole
   rollout. A rollback is a false decision, and the rate must sit at or below
   ``alpha_rollback``. This is the claim anytime-valid inference buys.
2. **Peeking.** The same arrival process judged by a fixed-horizon two-proportion z-test that is
   re-read at every look and acted on as soon as it is significant. Its decision rate is what the
   "refresh the dashboard" habit costs.
3. **A single look.** The same z-test evaluated once, at the horizon, to show the inflation comes
   from peeking rather than from the test.
4. **False promotion.** A challenger whose error rate is beyond the non-inferiority margin must be
   promoted with probability at most ``alpha_promotion``.
5. **Confidence-sequence coverage.** The intervals the operator reads next to the e-values are
   time-uniform: over simulated paths, the share that ever leaves its sequence must stay at or
   below ``alpha``, checked at every step rather than at a chosen horizon.
6. **Sticky assignment.** The router draws the canary arm once per *subject* (a consistent hash), so
   a repeat visitor's requests are one draw, not several. The same A/A traffic is measured three
   ways: one request per subject, five requests per subject scored per request, and five scored per
   subject -- which is what ``control/evidence.py`` records. The middle number is what ignoring the
   clustering costs; the third is the shipped system's rate.

Usage:
  uv run python bench/bench_validity.py                  # CI size: 2000 rollouts
  uv run python bench/bench_validity.py --check          # also enforce bench/thresholds.json
  uv run python bench/bench_validity.py --rollouts 10000 # the full study (`make study`)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from progressive_model_delivery.sim import scenarios
from progressive_model_delivery.sim.naive import PeekingStudy
from progressive_model_delivery.sim.study import RolloutStudy, study_spec
from progressive_model_delivery.stats.confseq import beta_binomial_cs, normal_mixture_cs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.harness import (  # noqa: E402 - the repo root has to be importable first
    check_thresholds,
    machine,
    print_table,
    write_results,
)

LOOK_SIZE = 2_000
MAX_LOOKS = 90
SEED = 20_260_927
REQUESTS_PER_SUBJECT = 5
"""Requests per subject in the sticky-assignment arms (a modest repeat-visit rate)."""


def coverage(paths: int, horizon: int, *, alpha: float = 0.05, seed: int = 11) -> dict[str, Any]:
    """Share of simulated paths that ever leave their confidence sequence (target: <= alpha)."""
    rng = np.random.default_rng(seed)
    steps = np.arange(1, horizon + 1)
    result: dict[str, Any] = {"paths": paths, "horizon": horizon, "alpha": alpha}
    for rate in (0.05, 0.2, 0.5):
        successes = np.cumsum(rng.random((paths, horizon)) < rate, axis=1)
        lower, upper = beta_binomial_cs(successes, steps, alpha=alpha)
        miss = float(((lower > rate) | (upper < rate)).any(axis=1).mean())
        result[f"beta_binomial_miss_rate_p{rate:g}"] = round(miss, 5)

    burn_in = 200
    data = rng.exponential(0.4, size=(paths, horizon)) - 0.4  # skewed, log-loss-like
    total = np.cumsum(data, axis=1)
    total_sq = np.cumsum(data * data, axis=1)
    mean = total / steps
    variance = np.maximum(total_sq - total * total / steps, 1e-12) / np.maximum(steps - 1, 1)
    lower_n, upper_n = normal_mixture_cs(mean, variance / steps, tau=0.05, alpha=alpha)
    miss_normal = float(
        ((lower_n[:, burn_in:] > 0) | (upper_n[:, burn_in:] < 0)).any(axis=1).mean()
    )
    result["normal_mixture_miss_rate"] = round(miss_normal, 5)
    result["normal_mixture_burn_in"] = burn_in
    return result


def run(rollouts: int) -> dict[str, Any]:
    spec = study_spec()
    started = time.perf_counter()

    aa = RolloutStudy(
        spec,
        scenarios.identical(),
        rollouts=rollouts,
        look_size=LOOK_SIZE,
        max_looks=MAX_LOOKS,
        seed=SEED,
    ).run()
    peeking = PeekingStudy(
        scenarios.identical(),
        rollouts=rollouts,
        look_size=LOOK_SIZE,
        looks=MAX_LOOKS,
        weight=0.5,
        alpha=0.05,
        seed=SEED + 1,
    ).run()
    inferior = RolloutStudy(
        spec,
        scenarios.inferior_beyond_margin(1.25),
        rollouts=rollouts,
        look_size=LOOK_SIZE,
        max_looks=MAX_LOOKS,
        seed=SEED + 2,
    ).run()
    # The shipped assignment: one arm draw per subject. `sticky_per_request` scores every request
    # as if it were an independent draw (what the reader did before it aggregated by subject);
    # `sticky_per_subject` scores one observation per subject, which is what it records now.
    clustered = max(rollouts // 2, 250)
    sticky_per_request = RolloutStudy(
        spec,
        scenarios.identical(),
        rollouts=clustered,
        look_size=LOOK_SIZE,
        max_looks=MAX_LOOKS,
        seed=SEED + 5,
        requests_per_subject=REQUESTS_PER_SUBJECT,
    ).run()
    sticky_per_subject = RolloutStudy(
        spec,
        scenarios.identical(),
        rollouts=clustered,
        look_size=LOOK_SIZE,
        max_looks=MAX_LOOKS,
        seed=SEED + 6,
        requests_per_subject=REQUESTS_PER_SUBJECT,
        aggregate_by_subject=True,
    ).run()
    # Sensitivity of the A/A claim to how correlated the two models' mistakes are.
    sensitivity = {
        f"coupling_{coupling:g}": RolloutStudy(
            spec,
            scenarios.identical(coupling=coupling),
            rollouts=max(rollouts // 4, 100),
            look_size=LOOK_SIZE,
            max_looks=MAX_LOOKS,
            seed=SEED + 3,
        )
        .run()
        .to_json()
        for coupling in (0.5, 0.95)
    }
    elapsed = time.perf_counter() - started

    low, high = aa.rollback_ci()
    promotion_low, promotion_high = inferior.promotion_ci()
    request_low, request_high = sticky_per_request.rollback_ci()
    subject_low, subject_high = sticky_per_subject.rollback_ci()
    return {
        "alpha_rollback": spec.alpha_rollback,
        "alpha_promotion": spec.alpha_promotion,
        "rollouts": rollouts,
        "looks_per_rollout": MAX_LOOKS,
        "requests_per_look": LOOK_SIZE,
        "seconds": round(elapsed, 1),
        "aa_false_rollback_rate": round(aa.rollback_rate, 5),
        "aa_false_rollback_ci95": [round(low, 5), round(high, 5)],
        "aa_promotion_rate": round(aa.promotion_rate, 5),
        "naive_peeking_decision_rate": round(peeking.peeking_rate, 5),
        "naive_single_look_decision_rate": round(peeking.final_look_rate, 5),
        "false_promotion_rate": round(inferior.promotion_rate, 5),
        "false_promotion_ci95": [round(promotion_low, 5), round(promotion_high, 5)],
        "inferior_rollback_rate": round(inferior.rollback_rate, 5),
        "requests_per_subject": REQUESTS_PER_SUBJECT,
        "sticky_per_request_rollback_rate": round(sticky_per_request.rollback_rate, 5),
        "sticky_per_request_rollback_ci95": [round(request_low, 5), round(request_high, 5)],
        "sticky_per_subject_rollback_rate": round(sticky_per_subject.rollback_rate, 5),
        "sticky_per_subject_rollback_ci95": [round(subject_low, 5), round(subject_high, 5)],
        "sticky_per_subject_promotion_rate": round(sticky_per_subject.promotion_rate, 5),
        "sticky_per_subject_open_at_end": sticky_per_subject.open_at_end,
        "aa": aa.to_json(),
        "sticky_per_request": sticky_per_request.to_json(),
        "sticky_per_subject": sticky_per_subject.to_json(),
        "peeking": peeking.to_json(),
        "inferior": inferior.to_json(),
        "coupling_sensitivity": sensitivity,
        "confidence_sequences": coverage(max(rollouts // 2, 200), 1_500),
    }


def metrics(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The numbers ``bench/thresholds.json`` constrains."""
    return {
        "aa_false_rollback": {
            "rate": float(results["aa_false_rollback_rate"]),
            "ci95_upper": float(results["aa_false_rollback_ci95"][1]),
        },
        "false_promotion": {
            "rate": float(results["false_promotion_rate"]),
            "ci95_upper": float(results["false_promotion_ci95"][1]),
        },
        "aa_sticky_per_subject": {
            "rate": float(results["sticky_per_subject_rollback_rate"]),
            "ci95_upper": float(results["sticky_per_subject_rollback_ci95"][1]),
        },
        # The demonstration has to keep demonstrating: per-request evidence over clustered arms
        # must stay far above the budget, or this benchmark has stopped measuring anything.
        "aa_sticky_per_request": {
            "minus_rate": -float(results["sticky_per_request_rollback_rate"])
        },
        "naive_peeking": {"minus_rate": -float(results["naive_peeking_decision_rate"])},
        "inferior_caught": {"minus_rate": -float(results["inferior_rollback_rate"])},
        "confidence_sequences": {
            name: float(value)
            for name, value in results["confidence_sequences"].items()
            if name.endswith("miss_rate") or "miss_rate_" in name
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--rollouts", type=int, default=2_000)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    parser.add_argument("--json", action="store_true", help="print the full result document")
    args = parser.parse_args(argv)

    results = run(args.rollouts)
    write_results("validity", {"machine": machine(), "results": results})
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))

    print(
        f"{results['rollouts']} rollouts x up to {results['looks_per_rollout']} looks "
        f"({results['requests_per_look']} requests each) in {results['seconds']} s"
    )
    print_table(
        ["measurement", "rate", "95% CI", "budget"],
        [
            [
                "A/A false rollback (anytime-valid)",
                f"{results['aa_false_rollback_rate']:.4f}",
                f"[{results['aa_false_rollback_ci95'][0]:.4f}, "
                f"{results['aa_false_rollback_ci95'][1]:.4f}]",
                f"alpha_rollback = {results['alpha_rollback']}",
            ],
            [
                "A/A decision by repeated z-test (peeking)",
                f"{results['naive_peeking_decision_rate']:.4f}",
                "-",
                "nominal alpha = 0.05",
            ],
            [
                "A/A decision by the same z-test, one look",
                f"{results['naive_single_look_decision_rate']:.4f}",
                "-",
                "nominal alpha = 0.05",
            ],
            [
                "promotion of a challenger past the margin",
                f"{results['false_promotion_rate']:.4f}",
                f"[{results['false_promotion_ci95'][0]:.4f}, "
                f"{results['false_promotion_ci95'][1]:.4f}]",
                f"alpha_promotion = {results['alpha_promotion']}",
            ],
            [
                "rollback of a challenger past the margin",
                f"{results['inferior_rollback_rate']:.4f}",
                "-",
                "power, not error",
            ],
            [
                f"A/A false rollback, {results['requests_per_subject']} requests per subject, "
                "scored per request",
                f"{results['sticky_per_request_rollback_rate']:.4f}",
                f"[{results['sticky_per_request_rollback_ci95'][0]:.4f}, "
                f"{results['sticky_per_request_rollback_ci95'][1]:.4f}]",
                f"alpha_rollback = {results['alpha_rollback']} (breached)",
            ],
            [
                f"A/A false rollback, {results['requests_per_subject']} requests per subject, "
                "scored per subject",
                f"{results['sticky_per_subject_rollback_rate']:.4f}",
                f"[{results['sticky_per_subject_rollback_ci95'][0]:.4f}, "
                f"{results['sticky_per_subject_rollback_ci95'][1]:.4f}]",
                f"alpha_rollback = {results['alpha_rollback']}",
            ],
        ],
    )
    print(
        f"  sticky assignment: the arm is drawn once per subject, so a customer's "
        f"{results['requests_per_subject']} requests are one observation. Scoring them as "
        f"{results['requests_per_subject']} inflates the rate past the budget; scoring one per "
        "subject holds the budget and costs evidence -- "
        f"{results['sticky_per_subject_promotion_rate']:.3f} of those rollouts reached promotion "
        f"within {results['looks_per_rollout']} looks "
        f"({results['sticky_per_subject_open_at_end']} were still open)."
    )
    for name, payload in results["coupling_sensitivity"].items():
        print(
            f"  {name}: A/A false rollback {payload['rollback_rate']:.4f} "
            f"over {payload['rollouts']} rollouts"
        )
    sequences = results["confidence_sequences"]
    print()
    print(
        f"time-uniform coverage over {sequences['paths']} paths x {sequences['horizon']} steps "
        f"(alpha = {sequences['alpha']}):"
    )
    print_table(
        ["confidence sequence", "miss rate"],
        [
            [name.replace("_miss_rate", "").replace("_", " "), f"{value:.4f}"]
            for name, value in sequences.items()
            if isinstance(value, float) and "miss_rate" in name
        ],
    )
    if args.check:
        return check_thresholds("validity", metrics(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
