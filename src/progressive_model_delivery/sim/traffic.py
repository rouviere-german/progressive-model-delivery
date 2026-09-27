"""Synthetic HTTP traffic with delayed, lossy labels, on a simulated clock.

This is the end-to-end path: the simulator talks to a *real* router over HTTP (any
``httpx.Client``, including an in-process test client), so predictions go through
authentication, validation, sticky assignment, the shadow forwarder and the Parquet prediction
log, and labels go through the feedback endpoint's integrity checks (known request id, TTL,
deduplication, rate limit).

Time is simulated. :class:`SimulatedClock` is passed to the router, the shadow worker and the
controller, so a rollout that would take hours of wall clock replays in seconds without any
``sleep``: each tick advances the clock by ``tick_seconds``, labels become due when their sampled
delay has passed, and the controller's maturation window is honoured exactly as in production.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

from progressive_model_delivery.models.features import FEATURES
from progressive_model_delivery.models.synthetic import (
    LabelDelay,
    sample_features,
    sample_label_delays,
    sample_labels,
    subject_ids,
)

__all__ = ["PendingLabel", "SimulatedClock", "TickReport", "TrafficConfig", "TrafficSimulator"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class SimulatedClock:
    """A monotonic clock the simulation advances by hand (shared with every component)."""

    def __init__(self, start: float = 1_767_225_600.0) -> None:
        self._now = float(start)

    def __call__(self) -> float:
        return self._now

    @property
    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("a simulated clock never goes backwards")
        self._now += seconds
        return self._now


@dataclass(frozen=True, slots=True)
class TrafficConfig:
    """How much traffic a tick carries and how its labels behave.

    Because the simulated clock is discrete, every request of a tick is stamped with that tick's
    time and its label is delivered at the *next* tick. ``tick_seconds`` must therefore stay below
    the router's ``label_ttl_seconds``, or every label arrives after its TTL and is refused;
    :meth:`TrafficSimulator.assert_ttl_fits` checks that against a concrete TTL.
    """

    seed: int = 4_242
    requests_per_tick: int = 200
    tick_seconds: float = 60.0
    delay: LabelDelay = field(default_factory=LabelDelay)
    covariate_shift: float = 0.0
    """Moves ``discount_rate`` and ``usage_trend``; the drift precondition watches this."""

    def __post_init__(self) -> None:
        if self.requests_per_tick < 1:
            raise ValueError("requests_per_tick must be >= 1")
        if self.tick_seconds <= 0:
            raise ValueError("tick_seconds must be positive")


@dataclass(frozen=True, slots=True)
class PendingLabel:
    request_id: str
    label: int
    due_at: float


@dataclass(frozen=True, slots=True)
class TickReport:
    tick: int
    now: float
    requests: int
    predict_failures: int
    labels_submitted: int
    labels_accepted: int
    labels_refused: int

    def merged(self, other: TickReport) -> TickReport:
        return TickReport(
            tick=other.tick,
            now=other.now,
            requests=self.requests + other.requests,
            predict_failures=self.predict_failures + other.predict_failures,
            labels_submitted=self.labels_submitted + other.labels_submitted,
            labels_accepted=self.labels_accepted + other.labels_accepted,
            labels_refused=self.labels_refused + other.labels_refused,
        )


class TrafficSimulator:
    """Generate traffic against a router and deliver its labels when they come due."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        api_key: str,
        clock: SimulatedClock,
        config: TrafficConfig | None = None,
        flush: Callable[[], None] | None = None,
    ) -> None:
        self.client = client
        self.clock = clock
        self.config = config or TrafficConfig()
        self._headers = {"X-API-Key": api_key}
        self._flush = flush
        self._rng = np.random.default_rng(self.config.seed)
        self._pending: list[PendingLabel] = []
        self.ticks = 0
        self.total = TickReport(0, clock.now, 0, 0, 0, 0, 0)

    def assert_ttl_fits(self, label_ttl_seconds: float) -> None:
        """Fail fast when a tick is longer than the label TTL (every label would be refused)."""
        if self.config.tick_seconds >= label_ttl_seconds:
            raise ValueError(
                f"tick_seconds={self.config.tick_seconds} must be below the router's label TTL "
                f"({label_ttl_seconds}); labels are delivered one tick after the request"
            )

    # --- generation --------------------------------------------------------------------------

    def _batch(self, count: int) -> tuple[list[dict[str, Any]], IntArray, FloatArray]:
        """Ready-to-post request bodies plus their true labels and label delays."""
        frame = sample_features(count, self._rng, shift=self.config.covariate_shift)
        labels = sample_labels(frame, self._rng)
        delays = sample_label_delays(labels, self._rng, self.config.delay)
        subjects = subject_ids(count, self._rng)
        rows = frame.to_dict(orient="records")
        bodies = [
            {"subject_id": subject, "features": self._features(row)}
            for subject, row in zip(subjects, rows, strict=True)
        ]
        return bodies, labels, delays

    @staticmethod
    def _features(row: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in FEATURES:
            value = row[name]
            payload[name] = value if isinstance(value, str) else float(value)
        for integer in ("tenure_months", "support_tickets", "num_products"):
            payload[integer] = int(payload[integer])
        return payload

    # --- one tick ----------------------------------------------------------------------------

    def tick(self, requests: int | None = None) -> TickReport:
        """Advance the clock, send ``requests`` predictions, then deliver every due label."""
        count = self.config.requests_per_tick if requests is None else requests
        if count < 0:
            raise ValueError("requests must not be negative")
        now = self.clock.advance(self.config.tick_seconds)
        self.ticks += 1
        bodies, labels, delays = self._batch(max(count, 1))
        failures = 0
        for index in range(count):
            response = self.client.post(
                "/v1/predict", json=bodies[index], headers=self._headers
            )
            if response.status_code != 200:
                failures += 1
                continue
            request_id = str(response.json()["request_id"])
            delay = float(delays[index])
            if np.isfinite(delay):
                self._pending.append(PendingLabel(request_id, int(labels[index]), now + delay))
        submitted, accepted, refused = self._deliver(now)
        if self._flush is not None:
            self._flush()
        report = TickReport(
            tick=self.ticks,
            now=now,
            requests=count,
            predict_failures=failures,
            labels_submitted=submitted,
            labels_accepted=accepted,
            labels_refused=refused,
        )
        self.total = self.total.merged(report)
        return report

    def _deliver(self, now: float) -> tuple[int, int, int]:
        due = [item for item in self._pending if item.due_at <= now]
        if not due:
            return 0, 0, 0
        self._pending = [item for item in self._pending if item.due_at > now]
        accepted = refused = 0
        for item in due:
            response = self.client.post(
                "/v1/feedback",
                json={"request_id": item.request_id, "label": item.label},
                headers=self._headers,
            )
            if response.status_code == 202:
                accepted += 1
            else:
                refused += 1
        return len(due), accepted, refused

    def run(self, ticks: int, requests: int | None = None) -> Iterator[TickReport]:
        """Yield one report per tick (a generator, so a caller can reconcile between ticks)."""
        if ticks < 0:
            raise ValueError("ticks must not be negative")
        for _ in range(ticks):
            yield self.tick(requests)

    @property
    def pending_labels(self) -> int:
        return len(self._pending)
