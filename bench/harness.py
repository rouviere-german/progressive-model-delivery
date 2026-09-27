"""Shared plumbing for the bench scripts: machine facts, result files and thresholds.

``bench/thresholds.json`` maps ``<bench>.<group>.<metric>`` to an upper bound. Rates that must
stay *high* (power, not error) are recorded as their negation (``minus_rate``), so every threshold
is an upper bound and one comparison covers both directions.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "bench" / "results"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

__all__ = [
    "check_thresholds",
    "machine",
    "print_table",
    "timed",
    "write_results",
]


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
    }


def timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    """Wall-clock percentiles of ``function`` in microseconds."""
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = np.asarray(samples, dtype=np.float64) / 1_000.0
    return {
        "runs": float(runs),
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
        "p99_us": round(float(np.percentile(micros, 99)), 1),
    }


def write_results(name: str, document: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def print_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def check_thresholds(bench: str, measured: dict[str, dict[str, float]]) -> int:
    """Compare ``measured`` against this bench's section of ``bench/thresholds.json``."""
    document: dict[str, dict[str, dict[str, float]]] = json.loads(
        THRESHOLDS.read_text(encoding="utf-8")
    )
    limits = document.get(bench)
    if limits is None:
        print(f"REGRESSION: bench/thresholds.json has no section for {bench!r}")
        return 1
    problems: list[str] = []
    for group, group_limits in limits.items():
        if group not in measured:
            problems.append(f"{bench}.{group}: measurement missing")
            continue
        for metric, limit in group_limits.items():
            if metric not in measured[group]:
                problems.append(f"{bench}.{group}.{metric}: measurement missing")
                continue
            value = measured[group][metric]
            if value > limit:
                problems.append(f"{bench}.{group}.{metric} = {value} exceeds threshold {limit}")
    for problem in problems:
        print(f"REGRESSION: {problem}")
    if problems:
        return 1
    print(f"all {bench} measurements within thresholds")
    return 0
