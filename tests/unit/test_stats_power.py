from __future__ import annotations

import numpy as np
import pytest

from progressive_model_delivery.stats.power import (
    fixed_horizon_power,
    fixed_horizon_sample_size,
    simulate_sequential_detection,
    two_proportion_z,
)

pytestmark = pytest.mark.unit


def test_fixed_horizon_sample_size_achieves_the_requested_power() -> None:
    n = fixed_horizon_sample_size(0.10, 0.02, alpha=0.05, power=0.8, allocation=0.5)
    assert 5_000 < n < 7_000
    assert fixed_horizon_power(0.10, 0.12, n, alpha=0.05, allocation=0.5) == pytest.approx(
        0.8, abs=0.01
    )
    # A 5% canary needs far more total traffic than a 50/50 split.
    assert fixed_horizon_sample_size(0.10, 0.02, allocation=0.05) > 4 * n


def test_fixed_horizon_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="delta"):
        fixed_horizon_sample_size(0.1, -0.01)
    with pytest.raises(ValueError, match="alpha"):
        fixed_horizon_sample_size(0.1, 0.01, alpha=1.0)


def test_two_proportion_z_is_zero_when_undefined() -> None:
    z = two_proportion_z([0, 10], [0, 100], [0, 20], [0, 100])
    assert z[0] == 0.0
    # pooled p = 0.15, se = sqrt(0.15 * 0.85 * 0.02)
    assert z[1] == pytest.approx(0.1 / np.sqrt(0.15 * 0.85 * 0.02))
    assert float(two_proportion_z(0, 10, 0, 10)) == 0.0


def test_sequential_plan_detects_large_regressions_and_rarely_fires_under_the_null() -> None:
    harm = simulate_sequential_detection(0.10, 0.15, max_requests=8_000, reps=300, seed=1)
    assert harm.detection_probability > 0.95
    assert harm.median_requests_to_detection is not None
    assert harm.fixed_horizon_requests_80 is not None
    assert harm.unconditional_median_requests is not None
    # The conditional test needs no more data than the unconditional one.
    assert harm.median_requests_to_detection <= harm.unconditional_median_requests
    null = simulate_sequential_detection(0.10, 0.10, max_requests=8_000, reps=300, seed=2)
    assert null.detection_probability <= 0.05
    assert null.fixed_horizon_requests_80 is None
    assert np.isfinite(null.alpha)
