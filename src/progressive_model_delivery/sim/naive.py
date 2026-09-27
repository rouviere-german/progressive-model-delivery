"""The practice this repository argues against: a fixed-horizon test looked at repeatedly.

:class:`PeekingStudy` runs the same arrival process as :class:`~.study.RolloutStudy` but decides
with a *fixed-horizon* two-proportion z-test on the cumulative error counts, re-evaluated at
every look and acted on as soon as it is significant ("refresh the dashboard until something
crosses"). The critical value is the correct one for a single look, which is exactly the mistake:
under repeated evaluation the family-wise error rate grows with the number of looks and the
nominal alpha no longer bounds anything.

The same class reports the honest baseline — one single evaluation at the horizon — so the
inflation can be attributed to peeking rather than to the test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from progressive_model_delivery.models.features import REGION_SHARES
from progressive_model_delivery.sim.scenarios import Scenario
from progressive_model_delivery.sim.study import binomial_ci
from progressive_model_delivery.stats.power import two_proportion_z

__all__ = ["PeekingResult", "PeekingStudy"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PeekingResult:
    scenario: str
    rollouts: int
    looks: int
    look_size: int
    weight: float
    alpha: float
    peeking_decisions: int
    final_look_decisions: int
    median_look_of_first_decision: float | None

    @property
    def peeking_rate(self) -> float:
        return self.peeking_decisions / self.rollouts

    @property
    def final_look_rate(self) -> float:
        return self.final_look_decisions / self.rollouts

    def to_json(self) -> dict[str, object]:
        low, high = binomial_ci(self.peeking_decisions, self.rollouts)
        flow, fhigh = binomial_ci(self.final_look_decisions, self.rollouts)
        return {
            "scenario": self.scenario,
            "rollouts": self.rollouts,
            "looks": self.looks,
            "look_size": self.look_size,
            "canary_weight": self.weight,
            "alpha": self.alpha,
            "peeking_decision_rate": round(self.peeking_rate, 5),
            "peeking_decision_rate_ci95": [round(low, 5), round(high, 5)],
            "single_look_decision_rate": round(self.final_look_rate, 5),
            "single_look_decision_rate_ci95": [round(flow, 5), round(fhigh, 5)],
            "median_look_of_first_decision": self.median_look_of_first_decision,
        }


class PeekingStudy:
    """Repeated two-sided z-tests on cumulative canary error counts, at nominal ``alpha``."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        rollouts: int,
        look_size: int,
        looks: int,
        weight: float,
        alpha: float = 0.05,
        seed: int,
        label_rate: float = 0.97,
        subgroup_shares: Sequence[float] = REGION_SHARES,
    ) -> None:
        if rollouts < 1 or look_size < 1 or looks < 1:
            raise ValueError("rollouts, look_size and looks must be positive")
        if not 0.0 < weight < 1.0:
            raise ValueError("weight must be in (0, 1)")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.scenario = scenario
        self.rollouts = rollouts
        self.look_size = look_size
        self.looks = looks
        self.weight = weight
        self.alpha = alpha
        self.label_rate = label_rate
        self.shares = np.asarray(subgroup_shares, dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def run(self) -> PeekingResult:
        critical = float(stats.norm.ppf(1.0 - self.alpha / 2.0))
        n_a = np.zeros(self.rollouts, dtype=np.float64)
        k_a = np.zeros(self.rollouts, dtype=np.float64)
        n_b = np.zeros(self.rollouts, dtype=np.float64)
        k_b = np.zeros(self.rollouts, dtype=np.float64)
        decided = np.zeros(self.rollouts, dtype=bool)
        final = 0
        first_look = np.full(self.rollouts, -1, dtype=np.int64)
        champion_rates = self.scenario.champion.error_array
        challenger_rates = self.scenario.challenger.error_array

        for look in range(1, self.looks + 1):
            arrivals = self._rng.multinomial(self.look_size, self.shares, size=self.rollouts)
            on_b = self._rng.binomial(arrivals, self.weight)
            on_a = arrivals - on_b
            labeled_a = self._rng.binomial(on_a, self.label_rate)
            labeled_b = self._rng.binomial(on_b, self.label_rate)
            n_a += labeled_a.sum(axis=-1)
            n_b += labeled_b.sum(axis=-1)
            k_a += self._rng.binomial(labeled_a, champion_rates).sum(axis=-1)
            k_b += self._rng.binomial(labeled_b, challenger_rates).sum(axis=-1)
            z = two_proportion_z(k_a, n_a, k_b, n_b)
            crossed = np.abs(z) >= critical
            first_look = np.where(crossed & ~decided, look, first_look)
            decided = decided | crossed
            if look == self.looks:
                final = int(np.count_nonzero(crossed))

        looked = first_look[first_look > 0].astype(np.float64)
        return PeekingResult(
            scenario=self.scenario.name,
            rollouts=self.rollouts,
            looks=self.looks,
            look_size=self.look_size,
            weight=self.weight,
            alpha=self.alpha,
            peeking_decisions=int(np.count_nonzero(decided)),
            final_look_decisions=final,
            median_look_of_first_decision=(
                None if looked.size == 0 else round(float(np.median(looked)), 2)
            ),
        )
