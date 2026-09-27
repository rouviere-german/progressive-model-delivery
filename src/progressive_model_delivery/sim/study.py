"""Vectorized rollout study: thousands of rollouts monitored continuously, same decision code.

The study drives :func:`~progressive_model_delivery.control.guardrails.evaluate` and
:func:`~progressive_model_delivery.control.guardrails.decide` — the exact functions the
controller calls — with a leading batch axis of shape ``(R,)``, so the measured error rates are
the error rates of the procedure that ships, not of a re-implementation.

Each *look* is one reconciliation pass: ``look_size`` requests arrive, the per-stage sufficient
statistics of every still-open rollout grow, every guardrail e-value is recomputed from scratch
over all stages, and the stage rule fires. Rollouts advance, roll back, request promotion or
abort independently.

Two modelling choices are approximations and are stated here rather than hidden:

* Per-request log-losses are not drawn one by one. Their running ``(count, sum, sum of squares)``
  are drawn from the Gaussian limit of those accumulators, which is the same limit the
  normal-mixture test itself assumes; the sum of squares is floored so the implied variance stays
  non-negative.
* Labels mature within one look. Real maturation is modelled end to end by the traffic simulator
  and the ``EvidenceReader`` (see ``tests/integration/test_evidence.py``); mixing it in here would
  only rescale the horizon.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import stats

from progressive_model_delivery.control.guardrails import (
    ArmCounts,
    ArmMoments,
    Decision,
    PairedCounts,
    PairedMoments,
    StageEvidence,
    decide,
    evaluate,
)
from progressive_model_delivery.control.spec import (
    DriftSpec,
    GuardrailSpec,
    Metric,
    RolloutSpec,
    StageKind,
    StageSpec,
)
from progressive_model_delivery.models.features import REGION_SHARES, REGIONS
from progressive_model_delivery.sim.scenarios import Scenario, paired_cell_probabilities
from progressive_model_delivery.stats.msprt import Moments

__all__ = [
    "LOOK_SECONDS",
    "RolloutStudy",
    "StudyResult",
    "binomial_ci",
    "study_spec",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

LOOK_SECONDS = 600.0
"""Simulated seconds between two reconciliation passes (10 minutes)."""


def study_spec(
    *,
    alpha_rollback: float = 0.025,
    alpha_promotion: float = 0.025,
    non_inferiority_ratio: float = 1.10,
    log_loss_margin: float = 0.02,
    slo_ms: float = 25.0,
    shadow_samples: int = 2_000,
    canary_samples: tuple[int, int, int, int] = (20, 60, 200, 400),
) -> RolloutSpec:
    """The policy the studies run: the production guardrails on a shortened sample schedule.

    The sample minimums are smaller than ``specs/standard.yaml`` so that a rollout traverses all
    five stages within a few dozen looks; that *increases* the number of peeks per rollout, which
    makes the validity claim harder rather than easier.
    """
    weights: tuple[int, int, int, int] = (1, 5, 25, 50)
    stages = [
        StageSpec(
            kind=StageKind.SHADOW,
            min_samples=shadow_samples,
            min_duration_seconds=0.0,
            max_duration_seconds=1e9,
        )
    ]
    stages += [
        StageSpec(
            kind=StageKind.CANARY,
            weight_percent=weight,
            min_samples=samples,
            min_duration_seconds=0.0,
            max_duration_seconds=1e9,
            require_non_inferiority=weight >= 25,
        )
        for weight, samples in zip(weights, canary_samples, strict=True)
    ]
    return RolloutSpec(
        name="study-policy",
        version=1,
        alpha_rollback=alpha_rollback,
        alpha_promotion=alpha_promotion,
        label_window_seconds=3_600.0,
        stages=tuple(stages),
        guardrails=(
            GuardrailSpec(
                name="error_rate", metric=Metric.ERROR, non_inferiority_ratio=non_inferiority_ratio
            ),
            GuardrailSpec(name="worst_subgroup_error", metric=Metric.ERROR, subgroups=True),
            GuardrailSpec(name="latency_slo", metric=Metric.LATENCY_SLO, slo_ms=slo_ms),
            GuardrailSpec(
                name="log_loss", metric=Metric.LOG_LOSS, non_inferiority_margin=log_loss_margin
            ),
        ),
        drift=DriftSpec(),
    )


def binomial_ci(successes: int, trials: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval for a rate, used to report study results honestly."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("0 <= successes <= trials and trials >= 1")
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(alpha / 2, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return lower, upper


# --- accumulators ----------------------------------------------------------------------------


def _zeros(rollouts: int, groups: int) -> FloatArray:
    return np.zeros((rollouts, groups), dtype=np.float64)


@dataclass(slots=True)
class _PairedBernoulli:
    """Running 2x2 table per (rollout, subgroup)."""

    cells: FloatArray  # shape (R, G, 4): n00, n01, n10, n11

    @classmethod
    def create(cls, rollouts: int, groups: int) -> _PairedBernoulli:
        return cls(np.zeros((rollouts, groups, 4), dtype=np.float64))

    def add(self, draws: IntArray, mask: BoolArray) -> None:
        self.cells += draws * mask[:, None, None]

    def counts(self) -> PairedCounts:
        return PairedCounts(
            self.cells[..., 0], self.cells[..., 1], self.cells[..., 2], self.cells[..., 3]
        )


@dataclass(slots=True)
class _ArmBernoulli:
    n_champion: FloatArray
    k_champion: FloatArray
    n_challenger: FloatArray
    k_challenger: FloatArray

    @classmethod
    def create(cls, rollouts: int, groups: int) -> _ArmBernoulli:
        return cls(*(_zeros(rollouts, groups) for _ in range(4)))

    def add(
        self,
        n_a: IntArray,
        k_a: IntArray,
        n_b: IntArray,
        k_b: IntArray,
        mask: BoolArray,
    ) -> None:
        keep = mask[:, None]
        self.n_champion += n_a * keep
        self.k_champion += k_a * keep
        self.n_challenger += n_b * keep
        self.k_challenger += k_b * keep

    def counts(self) -> ArmCounts:
        return ArmCounts(self.n_champion, self.k_champion, self.n_challenger, self.k_challenger)


@dataclass(slots=True)
class _MomentAccumulator:
    count: FloatArray
    total: FloatArray
    total_sq: FloatArray

    @classmethod
    def create(cls, rollouts: int, groups: int) -> _MomentAccumulator:
        return cls(*(_zeros(rollouts, groups) for _ in range(3)))

    def add(
        self, count: FloatArray, total: FloatArray, total_sq: FloatArray, mask: BoolArray
    ) -> None:
        keep = mask[:, None]
        self.count += count * keep
        self.total += total * keep
        self.total_sq += total_sq * keep

    def moments(self) -> Moments:
        # Floor the sum of squares at n * mean^2 so the implied sample variance is >= 0.
        safe = np.maximum(self.count, 1.0)
        floor = self.total**2 / safe
        return Moments(self.count, self.total, np.maximum(self.total_sq, floor))


@dataclass(slots=True)
class _ShadowStageState:
    """Paired accumulators: both models score every request, nothing is served."""

    error: _PairedBernoulli
    latency: dict[str, _PairedBernoulli]
    loss_diff: _MomentAccumulator
    exposure: FloatArray

    @classmethod
    def create(cls, rollouts: int, groups: int, latency_names: Sequence[str]) -> _ShadowStageState:
        return cls(
            error=_PairedBernoulli.create(rollouts, groups),
            latency={n: _PairedBernoulli.create(rollouts, groups) for n in latency_names},
            loss_diff=_MomentAccumulator.create(rollouts, groups),
            exposure=np.zeros(rollouts, dtype=np.float64),
        )

    def evidence(self) -> StageEvidence:
        counts = self.error.counts()
        return StageEvidence(
            kind=StageKind.SHADOW,
            weight=0.0,
            error=counts,
            latency={name: acc.counts() for name, acc in self.latency.items()},
            log_loss=PairedMoments(self.loss_diff.moments()),
            samples=counts.total,
            exposure=self.exposure,
        )


@dataclass(slots=True)
class _CanaryStageState:
    """Two-arm accumulators at a fixed traffic weight."""

    weight: float
    error: _ArmBernoulli
    latency: dict[str, _ArmBernoulli]
    loss_champion: _MomentAccumulator
    loss_challenger: _MomentAccumulator
    exposure: FloatArray

    @classmethod
    def create(
        cls, weight: float, rollouts: int, groups: int, latency_names: Sequence[str]
    ) -> _CanaryStageState:
        return cls(
            weight=weight,
            error=_ArmBernoulli.create(rollouts, groups),
            latency={n: _ArmBernoulli.create(rollouts, groups) for n in latency_names},
            loss_champion=_MomentAccumulator.create(rollouts, groups),
            loss_challenger=_MomentAccumulator.create(rollouts, groups),
            exposure=np.zeros(rollouts, dtype=np.float64),
        )

    def evidence(self) -> StageEvidence:
        counts = self.error.counts()
        return StageEvidence(
            kind=StageKind.CANARY,
            weight=self.weight,
            error=counts,
            latency={name: acc.counts() for name, acc in self.latency.items()},
            log_loss=ArmMoments(self.loss_champion.moments(), self.loss_challenger.moments()),
            samples=counts.n_challenger.sum(axis=-1),
            exposure=self.exposure,
        )


_StageState = _ShadowStageState | _CanaryStageState


def _median(values: FloatArray) -> float | None:
    """The median of a per-rollout summary, or ``None`` when no rollout ended that way."""
    return None if values.size == 0 else round(float(np.median(values)), 2)


def _mean(values: FloatArray) -> float | None:
    """The mean of a per-rollout summary, or ``None`` when no rollout ended that way."""
    return None if values.size == 0 else round(float(np.mean(values)), 2)


# --- results ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyResult:
    scenario: str
    rollouts: int
    looks: int
    look_size: int
    rolled_back: int
    promotion_requested: int
    aborted: int
    open_at_end: int
    breaches: dict[str, int] = field(default_factory=dict)
    median_looks_to_rollback: float | None = None
    median_exposure_to_rollback: float | None = None
    mean_exposure_to_rollback: float | None = None
    median_looks_to_promotion: float | None = None

    @property
    def rollback_rate(self) -> float:
        return self.rolled_back / self.rollouts

    @property
    def promotion_rate(self) -> float:
        return self.promotion_requested / self.rollouts

    def rollback_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        return binomial_ci(self.rolled_back, self.rollouts, confidence=confidence)

    def promotion_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        return binomial_ci(self.promotion_requested, self.rollouts, confidence=confidence)

    def to_json(self) -> dict[str, object]:
        low, high = self.rollback_ci()
        plow, phigh = self.promotion_ci()
        return {
            "scenario": self.scenario,
            "rollouts": self.rollouts,
            "looks": self.looks,
            "look_size": self.look_size,
            "rolled_back": self.rolled_back,
            "rollback_rate": round(self.rollback_rate, 5),
            "rollback_rate_ci95": [round(low, 5), round(high, 5)],
            "promotion_requested": self.promotion_requested,
            "promotion_rate": round(self.promotion_rate, 5),
            "promotion_rate_ci95": [round(plow, 5), round(phigh, 5)],
            "aborted": self.aborted,
            "open_at_end": self.open_at_end,
            "median_looks_to_rollback": self.median_looks_to_rollback,
            "median_exposure_to_rollback": self.median_exposure_to_rollback,
            "mean_exposure_to_rollback": self.mean_exposure_to_rollback,
            "median_looks_to_promotion": self.median_looks_to_promotion,
            "breaches": dict(sorted(self.breaches.items())),
        }


class RolloutStudy:
    """Run ``rollouts`` independent rollouts of one scenario under continuous monitoring.

    ``shadow_traffic=False`` models a challenger that cannot be mirrored (it has side effects),
    so every guardrail has to be proven on served canary traffic; combined with
    ``study_spec(shadow_samples=0)`` it measures the exposure a canary-only rollout costs.
    """

    def __init__(
        self,
        spec: RolloutSpec,
        scenario: Scenario,
        *,
        rollouts: int,
        look_size: int,
        max_looks: int,
        seed: int,
        label_rate: float = 0.97,
        subgroup_shares: Sequence[float] = REGION_SHARES,
        look_seconds: float = LOOK_SECONDS,
        shadow_traffic: bool = True,
    ) -> None:
        if rollouts < 1 or look_size < 1 or max_looks < 1:
            raise ValueError("rollouts, look_size and max_looks must be positive")
        if not 0.0 < label_rate <= 1.0:
            raise ValueError("label_rate must be in (0, 1]")
        shares = np.asarray(subgroup_shares, dtype=np.float64)
        if shares.shape != (len(REGIONS),) or not math.isclose(
            float(shares.sum()), 1.0, abs_tol=1e-9
        ):
            raise ValueError("subgroup_shares must have one entry per subgroup and sum to 1")
        self.spec = spec
        self.scenario = scenario
        self.rollouts = rollouts
        self.look_size = look_size
        self.max_looks = max_looks
        self.label_rate = label_rate
        self.shares = shares
        self.look_seconds = look_seconds
        self.shadow_traffic = shadow_traffic
        self._rng = np.random.default_rng(seed)
        self._latency_guardrails = tuple(
            g.name for g in spec.guardrails if g.metric is Metric.LATENCY_SLO
        )
        self._error_cells = paired_cell_probabilities(
            scenario.champion.error_array, scenario.challenger.error_array, scenario.error_coupling
        )
        self._latency_cells = paired_cell_probabilities(
            scenario.champion.latency_array,
            scenario.challenger.latency_array,
            scenario.latency_coupling,
        )

    # --- per-look sampling -------------------------------------------------------------------

    def _new_stage_state(self, stage: StageSpec) -> _StageState:
        groups = len(REGIONS)
        if stage.kind is StageKind.SHADOW:
            return _ShadowStageState.create(self.rollouts, groups, self._latency_guardrails)
        return _CanaryStageState.create(
            stage.weight, self.rollouts, groups, self._latency_guardrails
        )

    def _arrivals(self) -> IntArray:
        """Requests per (rollout, subgroup) in one look."""
        draws: IntArray = self._rng.multinomial(self.look_size, self.shares, size=self.rollouts)
        return draws

    def _moment_draw(
        self, count: IntArray, mean: float, sd: float
    ) -> tuple[FloatArray, FloatArray]:
        """Gaussian-limit draw of ``(sum, sum of squares)`` for ``count`` i.i.d. observations."""
        n = count.astype(np.float64)
        root = np.sqrt(np.maximum(n, 0.0))
        total = n * mean + root * sd * self._rng.standard_normal(n.shape)
        second_mean = mean * mean + sd * sd
        second_sd = math.sqrt(max(2.0 * sd**4 + 4.0 * mean**2 * sd**2, 1e-12))
        total_sq = n * second_mean + root * second_sd * self._rng.standard_normal(n.shape)
        return total, np.maximum(total_sq, 0.0)

    def _grow_shadow(self, state: _ShadowStageState, arrivals: IntArray, mask: BoolArray) -> None:
        labeled = self._rng.binomial(arrivals, self.label_rate)
        state.error.add(self._rng.multinomial(labeled, self._error_cells), mask)
        for accumulator in state.latency.values():
            accumulator.add(self._rng.multinomial(arrivals, self._latency_cells), mask)
        total, total_sq = self._moment_draw(
            labeled, self.scenario.paired_log_loss_mean, self.scenario.paired_log_loss_sd
        )
        state.loss_diff.add(labeled.astype(np.float64), total, total_sq, mask)

    def _grow_canary(self, state: _CanaryStageState, arrivals: IntArray, mask: BoolArray) -> None:
        on_challenger = self._rng.binomial(arrivals, state.weight)
        on_champion = arrivals - on_challenger
        labeled_a = self._rng.binomial(on_champion, self.label_rate)
        labeled_b = self._rng.binomial(on_challenger, self.label_rate)
        errors_a = self._rng.binomial(labeled_a, self.scenario.champion.error_array)
        errors_b = self._rng.binomial(labeled_b, self.scenario.challenger.error_array)
        state.error.add(labeled_a, errors_a, labeled_b, errors_b, mask)
        for accumulator in state.latency.values():
            accumulator.add(
                on_champion,
                self._rng.binomial(on_champion, self.scenario.champion.latency_array),
                on_challenger,
                self._rng.binomial(on_challenger, self.scenario.challenger.latency_array),
                mask,
            )
        for moments, count, profile in (
            (state.loss_champion, labeled_a, self.scenario.champion),
            (state.loss_challenger, labeled_b, self.scenario.challenger),
        ):
            total, total_sq = self._moment_draw(count, profile.log_loss_mean, profile.log_loss_sd)
            moments.add(count.astype(np.float64), total, total_sq, mask)
        state.exposure += on_challenger.sum(axis=-1) * mask

    # --- the loop ----------------------------------------------------------------------------

    def _grow_all(
        self,
        states: list[_StageState],
        stage_index: IntArray,
        open_mask: BoolArray,
    ) -> None:
        """Draw one look of traffic into whichever stage each still-open rollout sits in."""
        arrivals = self._arrivals()
        for index, state in enumerate(states):
            mask = open_mask & (stage_index == index)
            if not mask.any():
                continue
            if isinstance(state, _ShadowStageState):
                if self.shadow_traffic:
                    self._grow_shadow(state, arrivals, mask)
            else:
                self._grow_canary(state, arrivals, mask)

    def run(self) -> StudyResult:
        states = [self._new_stage_state(stage) for stage in self.spec.stages]
        stage_index = np.zeros(self.rollouts, dtype=np.int64)
        elapsed = np.zeros(self.rollouts, dtype=np.float64)
        open_mask = np.ones(self.rollouts, dtype=bool)
        terminal = np.full(self.rollouts, int(Decision.HOLD), dtype=np.int64)
        decided_look = np.full(self.rollouts, -1, dtype=np.int64)
        exposure_at_decision = np.zeros(self.rollouts, dtype=np.float64)
        breaches: dict[str, int] = {}
        looks = 0

        for look in range(1, self.max_looks + 1):
            looks = look
            self._grow_all(states, stage_index, open_mask)
            elapsed = elapsed + self.look_seconds * open_mask

            stages = [state.evidence() for state in states]
            results = evaluate(self.spec, stages, REGIONS)
            samples = np.stack([stage.samples for stage in stages])
            current_samples = np.take_along_axis(samples, stage_index[None, :], axis=0)[0]
            outcome = decide(
                self.spec,
                stage_index=stage_index,
                elapsed_seconds=elapsed,
                stage_samples=current_samples,
                results=results,
            )
            decision = outcome.decision
            exposure = np.sum([state.exposure for state in states], axis=0)

            rollback = open_mask & (decision == int(Decision.ROLLBACK))
            if rollback.any():
                for result in results:
                    if result.test != "harm":
                        continue
                    hit = int(np.count_nonzero(rollback & result.crossed))
                    if hit:
                        breaches[result.label] = breaches.get(result.label, 0) + hit
            advance = open_mask & (decision == int(Decision.ADVANCE))
            promote = open_mask & (decision == int(Decision.REQUEST_PROMOTION))
            abort = open_mask & (decision == int(Decision.ABORT))
            finished = rollback | promote | abort
            terminal = np.where(finished, decision, terminal)
            decided_look = np.where(finished, look, decided_look)
            exposure_at_decision = np.where(finished, exposure, exposure_at_decision)
            stage_index = np.where(advance, stage_index + 1, stage_index)
            elapsed = np.where(advance, 0.0, elapsed)
            open_mask = open_mask & ~finished
            if not open_mask.any():
                break

        rolled = terminal == int(Decision.ROLLBACK)
        promoted = terminal == int(Decision.REQUEST_PROMOTION)
        aborted = terminal == int(Decision.ABORT)
        return StudyResult(
            scenario=self.scenario.name,
            rollouts=self.rollouts,
            looks=looks,
            look_size=self.look_size,
            rolled_back=int(np.count_nonzero(rolled)),
            promotion_requested=int(np.count_nonzero(promoted)),
            aborted=int(np.count_nonzero(aborted)),
            open_at_end=int(np.count_nonzero(open_mask)),
            breaches=breaches,
            median_looks_to_rollback=_median(decided_look[rolled].astype(np.float64)),
            median_exposure_to_rollback=_median(exposure_at_decision[rolled]),
            mean_exposure_to_rollback=_mean(exposure_at_decision[rolled]),
            median_looks_to_promotion=_median(decided_look[promoted].astype(np.float64)),
        )
