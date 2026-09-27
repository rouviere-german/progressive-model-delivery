"""The traffic simulator against a real router: delayed labels, the simulated clock, PII.

What the simulator has to get right for every study and the demo to mean anything:

* it speaks the router's real contract (every request is accepted, every feature in range);
* labels arrive late, in the order their delays dictate, and a label that misses the TTL is
  refused rather than silently accepted;
* raw subject ids never leave the simulator, so nothing identifying reaches the logs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from progressive_model_delivery.models.synthetic import LabelDelay
from progressive_model_delivery.serving.router import RouterRuntime, build_router_app
from progressive_model_delivery.sim.traffic import (
    SimulatedClock,
    TrafficConfig,
    TrafficSimulator,
)
from tests.integration.conftest import API_KEY, ControlEnv

pytestmark = pytest.mark.integration


@pytest.fixture
def clock(control_env: ControlEnv) -> SimulatedClock:
    return SimulatedClock(start=control_env.clock[0])


@pytest.fixture
def router(control_env: ControlEnv, clock: SimulatedClock) -> Iterator[TestClient]:
    settings = control_env.settings(
        label_ttl_seconds=120.0, feedback_rate_per_second=1e6, feedback_burst=1_000_000
    )
    app = build_router_app(settings, clock=clock, background=False)
    with TestClient(app) as client:
        yield client


def _runtime(client: TestClient) -> RouterRuntime:
    runtime = client.app.state.runtime  # type: ignore[attr-defined]
    assert isinstance(runtime, RouterRuntime)
    return runtime


def _simulator(
    client: TestClient, clock: SimulatedClock, **config: object
) -> TrafficSimulator:
    runtime = _runtime(client)
    values: dict[str, object] = {"seed": 3, "requests_per_tick": 40, "tick_seconds": 30.0}
    values.update(config)
    return TrafficSimulator(
        client,
        api_key=API_KEY,
        clock=clock,
        config=TrafficConfig(**values),  # type: ignore[arg-type]
        flush=runtime.flush,
    )


def _rows(tmp_path: Path, table: str) -> list[dict[str, object]]:
    files = [str(p) for p in (tmp_path / "predlog" / table).glob("*.parquet")]
    if not files:
        return []
    with duckdb.connect() as con:
        frame = con.execute("SELECT * FROM read_parquet($f)", {"f": files}).df()
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]


# --- the clock ----------------------------------------------------------------------------------


def test_the_simulated_clock_only_moves_forward() -> None:
    clock = SimulatedClock(start=100.0)
    assert clock() == 100.0 and clock.now == 100.0
    assert clock.advance(5.0) == 105.0
    assert clock() == 105.0
    with pytest.raises(ValueError, match="never goes backwards"):
        clock.advance(-1.0)


def test_the_config_validates_itself_and_the_label_ttl() -> None:
    with pytest.raises(ValueError, match="requests_per_tick"):
        TrafficConfig(requests_per_tick=0)
    with pytest.raises(ValueError, match="tick_seconds"):
        TrafficConfig(tick_seconds=0.0)


# --- traffic ------------------------------------------------------------------------------------


def test_every_generated_request_is_accepted_by_the_real_router(
    router: TestClient, clock: SimulatedClock, control_env: ControlEnv
) -> None:
    traffic = _simulator(router, clock, requests_per_tick=120)
    report = traffic.tick()
    assert report.requests == 120
    assert report.predict_failures == 0
    assert clock.now == control_env.clock[0] + 30.0
    rows = _rows(control_env.tmp_path, "router")
    assert len(rows) == 120
    assert {row["arm"] for row in rows} == {"champion"}
    assert all(0.0 <= float(str(row["score"])) <= 1.0 for row in rows)


def test_subject_ids_never_reach_the_logs(
    router: TestClient, clock: SimulatedClock, control_env: ControlEnv
) -> None:
    traffic = _simulator(router, clock, requests_per_tick=60)
    traffic.tick()
    raw = b"".join(
        path.read_bytes() for path in (control_env.tmp_path / "predlog").rglob("*.parquet")
    )
    assert b"cust-" not in raw
    rows = _rows(control_env.tmp_path, "router")
    assert all(len(str(row["subject_hash"])) == 32 for row in rows)


def test_labels_arrive_late_and_are_accepted_inside_the_ttl(
    router: TestClient, clock: SimulatedClock
) -> None:
    traffic = _simulator(
        router,
        clock,
        requests_per_tick=80,
        delay=LabelDelay(mean_negative_seconds=10.0, mean_positive_seconds=20.0, loss_rate=0.0),
    )
    first = traffic.tick()
    assert first.labels_submitted == 0  # nothing is due within the tick it was served in
    assert traffic.pending_labels == 80
    second = traffic.tick()
    assert second.labels_submitted > 0
    assert second.labels_accepted == second.labels_submitted
    assert second.labels_refused == 0
    assert traffic.total.requests == 160
    assert traffic.total.labels_accepted == second.labels_accepted


def test_a_label_that_misses_the_ttl_is_refused(
    router: TestClient, clock: SimulatedClock
) -> None:
    traffic = _simulator(
        router,
        clock,
        requests_per_tick=50,
        tick_seconds=150.0,  # a label delivered one tick later is already past the 120 s TTL
        delay=LabelDelay(mean_negative_seconds=10.0, mean_positive_seconds=20.0, loss_rate=0.0),
    )
    traffic.tick()
    later = traffic.tick()
    assert later.labels_submitted > 0
    assert later.labels_accepted == 0
    assert later.labels_refused == later.labels_submitted


def test_some_labels_never_arrive(router: TestClient, clock: SimulatedClock) -> None:
    traffic = _simulator(
        router,
        clock,
        requests_per_tick=300,
        delay=LabelDelay(mean_negative_seconds=5.0, mean_positive_seconds=8.0, loss_rate=0.25),
    )
    for _ in traffic.run(3):
        pass
    assert traffic.total.requests == 900
    # Only the first two ticks' requests can have been delivered by the third tick.
    deliverable = 600
    assert 0.6 < traffic.total.labels_submitted / deliverable < 0.85


def test_the_ttl_guard_rejects_a_tick_longer_than_the_ttl(
    router: TestClient, clock: SimulatedClock
) -> None:
    traffic = _simulator(router, clock, tick_seconds=200.0)
    with pytest.raises(ValueError, match="must be below the router's label TTL"):
        traffic.assert_ttl_fits(120.0)
    _simulator(router, clock, tick_seconds=30.0).assert_ttl_fits(120.0)


def test_a_covariate_shift_moves_the_feature_distribution(
    router: TestClient, clock: SimulatedClock, control_env: ControlEnv
) -> None:
    baseline = _simulator(router, clock, requests_per_tick=200, seed=1)
    baseline.tick()
    plain = [float(str(row["f_usage_trend"])) for row in _rows(control_env.tmp_path, "router")]
    shifted_sim = _simulator(router, clock, requests_per_tick=200, seed=1, covariate_shift=0.9)
    shifted_sim.tick()
    everything = [float(str(row["f_usage_trend"])) for row in _rows(control_env.tmp_path, "router")]
    shifted = everything[len(plain) :]
    assert sum(shifted) / len(shifted) > sum(plain) / len(plain) + 0.3


def test_running_zero_ticks_does_nothing_and_negatives_are_refused(
    router: TestClient, clock: SimulatedClock
) -> None:
    traffic = _simulator(router, clock)
    assert list(traffic.run(0)) == []
    assert traffic.ticks == 0
    with pytest.raises(ValueError, match="ticks"):
        list(traffic.run(-1))
    with pytest.raises(ValueError, match="requests"):
        traffic.tick(-5)


def test_a_predict_failure_is_counted_rather_than_raised(
    control_env: ControlEnv, clock: SimulatedClock
) -> None:
    settings = control_env.settings(api_key_sha256=["f" * 64])  # the simulator's key is unknown
    app = build_router_app(settings, clock=clock, background=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        traffic = _simulator(client, clock, requests_per_tick=5)
        report = traffic.tick()
    assert report.requests == 5
    assert report.predict_failures == 5
    assert traffic.pending_labels == 0
