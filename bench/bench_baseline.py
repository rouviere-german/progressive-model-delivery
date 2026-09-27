#!/usr/bin/env python3
"""Micro-benchmarks of the hot paths, with CI regression thresholds.

Three groups:

* **request path** - what a single prediction pays outside the model: PII redaction of a log line,
  the token-bucket rate limiter, the HMAC subject hash and the consistent-hash assignment.
* **decision path** - what one reconciliation pass costs: a beta-mixture e-value, the paired
  mSPRT, and a full ``evaluate`` + ``decide`` over five stages and every guardrail of the
  production policy. This is the cost of looking at a rollout, and it is why looking continuously
  is affordable.
* **control plane** - appending and verifying the hash-chained audit log, and PSI against a stored
  reference.

Thresholds in ``bench/thresholds.json`` are several times the numbers measured on a laptop,
because shared CI runners are noisy: they catch order-of-magnitude regressions, not small drifts.

Usage:
  uv run python bench/bench_baseline.py [--check]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from progressive_model_delivery.control.guardrails import (
    ArmCounts,
    ArmMoments,
    PairedCounts,
    PairedMoments,
    StageEvidence,
    decide,
    evaluate,
)
from progressive_model_delivery.control.spec import StageKind, bundled_spec
from progressive_model_delivery.control.store import ControlStore
from progressive_model_delivery.core.ratelimit import RateLimiter
from progressive_model_delivery.core.redaction import redact_text
from progressive_model_delivery.ml.drift import BinnedReference, DriftMonitor
from progressive_model_delivery.models.features import REGIONS
from progressive_model_delivery.serving.assignment import assign, subject_hash
from progressive_model_delivery.stats.evalues import beta_mixture_log_e, discordant_null
from progressive_model_delivery.stats.msprt import Moments, paired_mean_log_e

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.harness import (  # noqa: E402 - the repo root has to be importable first
    check_thresholds,
    machine,
    print_table,
    timed,
    write_results,
)

GROUPS = len(REGIONS)
SPEC = bundled_spec("standard")
SECRET = b"bench-subject-secret-key-material"


def _cell(value: float) -> np.ndarray:
    return np.full((GROUPS,), value, dtype=np.float64)


def _moments(count: float, mean: float, sd: float) -> Moments:
    n = _cell(count)
    return Moments(n, n * mean, n * (mean * mean + sd * sd))


def _stages() -> list[StageEvidence]:
    """One shadow stage plus four canary stages, as the production policy defines them."""
    shadow = StageEvidence(
        kind=StageKind.SHADOW,
        weight=0.0,
        error=PairedCounts(_cell(4_000), _cell(300), _cell(320), _cell(180)),
        latency={"latency_slo": PairedCounts(_cell(4_500), _cell(40), _cell(45), _cell(15))},
        log_loss=PairedMoments(_moments(4_800, 0.001, 0.4)),
        samples=np.asarray(4_800.0 * GROUPS),
    )
    canaries = [
        StageEvidence(
            kind=StageKind.CANARY,
            weight=weight,
            error=ArmCounts(
                _cell(1_000 * (1 - weight)),
                _cell(200 * (1 - weight)),
                _cell(1_000 * weight),
                _cell(200 * weight),
            ),
            latency={
                "latency_slo": ArmCounts(
                    _cell(1_000 * (1 - weight)),
                    _cell(12),
                    _cell(1_000 * weight),
                    _cell(4),
                )
            },
            log_loss=ArmMoments(
                _moments(1_000 * (1 - weight), 0.45, 0.5), _moments(1_000 * weight, 0.44, 0.5)
            ),
            samples=np.asarray(1_000 * weight * GROUPS),
            exposure=np.asarray(1_000 * weight * GROUPS),
        )
        for weight in (0.01, 0.05, 0.25, 0.50)
    ]
    return [shadow, *canaries]


# --- request path -------------------------------------------------------------------------------


def bench_redaction() -> dict[str, float]:
    line = (
        "user jane.doe@example.com paid with 4111 1111 1111 1111 from +49 30 1234 5678; "
        + "lorem ipsum dolor sit amet " * 35
    )
    return timed(lambda: redact_text(line), runs=2_000)


def bench_rate_limiter() -> dict[str, float]:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return timed(lambda: limiter.check("client"), runs=20_000)


def bench_subject_hash() -> dict[str, float]:
    return timed(lambda: subject_hash(SECRET, "cust-0123456"), runs=50_000)


def bench_assignment() -> dict[str, float]:
    hashed = subject_hash(SECRET, "cust-0123456")
    return timed(lambda: assign(hashed, "a" * 32, 2_500), runs=50_000)


# --- decision path ------------------------------------------------------------------------------


def bench_beta_mixture_e_value() -> dict[str, float]:
    n01, n10 = np.asarray(320.0), np.asarray(290.0)
    return timed(
        lambda: beta_mixture_log_e(
            n01, n10, null_p=discordant_null(), alternative="greater", strength=10.0
        ),
        runs=5_000,
    )


def bench_paired_msprt() -> dict[str, float]:
    moments = _moments(5_000, 0.002, 0.4)
    pooled = Moments.of(moments.count.sum(), moments.total.sum(), moments.total_sq.sum())
    return timed(
        lambda: paired_mean_log_e(pooled, null_value=0.0, tau=0.05, sided="greater", min_n=200),
        runs=5_000,
    )


def bench_reconcile_decision() -> dict[str, float]:
    stages = _stages()

    def once() -> None:
        results = evaluate(SPEC, stages, REGIONS)
        decide(
            SPEC,
            stage_index=3,
            elapsed_seconds=20_000.0,
            stage_samples=float(stages[3].samples),
            results=results,
        )

    return timed(once, runs=300, warmup=10)


# --- control plane ------------------------------------------------------------------------------


def bench_audit_append() -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        store = ControlStore(Path(tmp) / "control.sqlite")

        def once() -> None:
            with store.transaction() as conn:
                store.append_audit(
                    conn,
                    actor="controller:c1",
                    action="stage_advanced",
                    rollout_id="a" * 32,
                    data={"from_stage": 1, "to_stage": 2},
                )

        return timed(once, runs=200, warmup=10)


def bench_audit_verify() -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        store = ControlStore(Path(tmp) / "control.sqlite")
        with store.transaction() as conn:
            for index in range(1_000):
                store.append_audit(
                    conn, actor="c1", action="stage_advanced", rollout_id=None, data={"i": index}
                )
        return timed(store.verify_audit, runs=10, warmup=2)


def bench_psi() -> dict[str, float]:
    rng = np.random.default_rng(0)
    reference = BinnedReference.from_sample(rng.normal(size=10_000))
    current = rng.normal(loc=0.1, size=10_000)
    return timed(lambda: reference.psi(current), runs=200, warmup=5)


def bench_drift_monitor() -> dict[str, float]:
    rng = np.random.default_rng(0)
    monitor = DriftMonitor({"x": rng.normal(size=10_000)})
    current = {"x": rng.normal(loc=0.1, size=10_000)}
    return timed(lambda: monitor.check(current), runs=30, warmup=2)


BENCHMARKS = {
    "redact_text_1kb": bench_redaction,
    "rate_limiter_check": bench_rate_limiter,
    "subject_hash": bench_subject_hash,
    "assignment": bench_assignment,
    "beta_mixture_e_value": bench_beta_mixture_e_value,
    "paired_msprt": bench_paired_msprt,
    "reconcile_decision_5_stages": bench_reconcile_decision,
    "audit_append_one_entry": bench_audit_append,
    "audit_verify_1000_entries": bench_audit_verify,
    "psi_10k_vs_10k": bench_psi,
    "drift_check_10k_vs_10k": bench_drift_monitor,
}


def run() -> dict[str, dict[str, float]]:
    return {name: function() for name, function in BENCHMARKS.items()}


def metrics(results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {name: {"p95_us": row["p95_us"]} for name, row in results.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    document: dict[str, Any] = {"machine": machine(), "results": results}
    write_results("baseline", document)
    print_table(
        ["benchmark", "runs", "p50 (us)", "p95 (us)"],
        [
            [name, f"{row['runs']:.0f}", f"{row['p50_us']}", f"{row['p95_us']}"]
            for name, row in results.items()
        ],
    )
    if args.check:
        return check_thresholds("baseline", metrics(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
