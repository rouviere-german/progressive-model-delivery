"""Guardrail evaluation and the stage decision rule (vectorized over rollouts).

Inputs are per-stage sufficient statistics (:class:`StageEvidence`), with a leading batch shape
``B`` (``()`` for the controller's single rollout, ``(R,)`` for the statistical study) and a
trailing subgroup axis where it applies. The same functions therefore drive the controller and
the 2,000-rollout A/A study, so the study measures exactly the procedure that ships.

Tests per guardrail (e-values are multiplied across stages; each stage's e-value only uses
requests served in that stage, so the product is again an e-value):

========== ================================== ======================================
metric     shadow stage (paired)              canary stage (weight w)
========== ================================== ======================================
error      discordant pairs, pi0 = 1/2        events by arm, pi0 = w r / (w r + 1 - w)
latency    discordant pairs, pi0 = 1/2        events by arm, same
log_loss   paired mSPRT on loss differences   two-sample mSPRT (Welch variance)
========== ================================== ======================================

Harm tests use ``r = 1`` (or a zero difference) and every stage; non-inferiority and superiority
tests use canary stages only, because promotion evidence should come from the served arm.
Bonferroni splits ``alpha_rollback`` over all harm tests (subgroups included); promotion is an
intersection-union test at ``alpha_promotion``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np
import numpy.typing as npt

from progressive_model_delivery.control.spec import GuardrailSpec, Metric, RolloutSpec, StageKind
from progressive_model_delivery.stats.evalues import (
    Alternative,
    beta_mixture_log_e,
    discordant_null,
    ratio_null,
)
from progressive_model_delivery.stats.msprt import Moments, paired_mean_log_e, two_sample_mean_log_e

__all__ = [
    "ArmCounts",
    "ArmMoments",
    "Decision",
    "DecisionResult",
    "PairedCounts",
    "PairedMoments",
    "StageEvidence",
    "TestResult",
    "decide",
    "evaluate",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
TestKind = Literal["harm", "non_inferiority", "superiority"]


@dataclass(frozen=True, slots=True)
class PairedCounts:
    """Paired Bernoulli outcomes (champion event, challenger event), shape ``B + (G,)``."""

    n00: FloatArray
    n01: FloatArray
    n10: FloatArray
    n11: FloatArray

    def pooled(self) -> PairedCounts:
        return PairedCounts(
            self.n00.sum(axis=-1, keepdims=True),
            self.n01.sum(axis=-1, keepdims=True),
            self.n10.sum(axis=-1, keepdims=True),
            self.n11.sum(axis=-1, keepdims=True),
        )

    @property
    def total(self) -> FloatArray:
        result: FloatArray = (self.n00 + self.n01 + self.n10 + self.n11).sum(axis=-1)
        return result


@dataclass(frozen=True, slots=True)
class ArmCounts:
    """Bernoulli events by arm, shape ``B + (G,)``."""

    n_champion: FloatArray
    k_champion: FloatArray
    n_challenger: FloatArray
    k_challenger: FloatArray

    def pooled(self) -> ArmCounts:
        return ArmCounts(
            self.n_champion.sum(axis=-1, keepdims=True),
            self.k_champion.sum(axis=-1, keepdims=True),
            self.n_challenger.sum(axis=-1, keepdims=True),
            self.k_challenger.sum(axis=-1, keepdims=True),
        )


@dataclass(frozen=True, slots=True)
class PairedMoments:
    difference: Moments  # challenger loss minus champion loss, shape B + (G,)

    def pooled(self) -> Moments:
        d = self.difference
        return Moments(d.count.sum(axis=-1), d.total.sum(axis=-1), d.total_sq.sum(axis=-1))


@dataclass(frozen=True, slots=True)
class ArmMoments:
    champion: Moments
    challenger: Moments

    def pooled(self) -> tuple[Moments, Moments]:
        a, b = self.champion, self.challenger
        return (
            Moments(a.count.sum(axis=-1), a.total.sum(axis=-1), a.total_sq.sum(axis=-1)),
            Moments(b.count.sum(axis=-1), b.total.sum(axis=-1), b.total_sq.sum(axis=-1)),
        )


@dataclass(frozen=True, slots=True)
class StageEvidence:
    """Sufficient statistics of one stage. Counts use requests served *in* this stage."""

    kind: StageKind
    weight: float
    error: PairedCounts | ArmCounts
    latency: Mapping[str, PairedCounts | ArmCounts]  # keyed by latency guardrail name
    log_loss: PairedMoments | ArmMoments
    samples: FloatArray  # labeled, matured records involving the challenger, shape B
    exposure: FloatArray = field(default_factory=lambda: np.zeros(()))  # challenger-served


@dataclass(frozen=True, slots=True)
class TestResult:
    guardrail: str
    test: TestKind
    subgroup: str | None
    log_e: FloatArray
    threshold: float

    @property
    def crossed(self) -> BoolArray:
        result: BoolArray = self.log_e >= self.threshold
        return result

    @property
    def label(self) -> str:
        return self.guardrail if self.subgroup is None else f"{self.guardrail}[{self.subgroup}]"


def _bernoulli_stage_log_e(
    counts: PairedCounts | ArmCounts,
    *,
    kind: StageKind,
    weight: float,
    test: TestKind,
    ratio: float,
    strength: float,
) -> FloatArray:
    """log e of one stage for one Bernoulli test (shape ``B + (G,)``)."""
    if isinstance(counts, PairedCounts):
        if test != "harm":
            return np.zeros(counts.n01.shape)
        return beta_mixture_log_e(
            counts.n01,
            counts.n10,
            null_p=discordant_null(),
            alternative="greater",
            strength=strength,
        )
    if kind is not StageKind.CANARY or weight <= 0:
        return np.zeros(counts.k_challenger.shape)
    alternative = Alternative.GREATER if test == "harm" else Alternative.LESS
    return beta_mixture_log_e(
        counts.k_challenger,
        counts.k_champion,
        null_p=ratio_null(weight, ratio),
        alternative=alternative,
        strength=strength,
    )


def _log_loss_stage_log_e(
    moments: PairedMoments | ArmMoments, guardrail: GuardrailSpec, test: TestKind
) -> FloatArray:
    tau, min_n = guardrail.mixture_scale, guardrail.min_samples
    if isinstance(moments, PairedMoments):
        if test != "harm":
            return np.zeros(moments.difference.count.shape[:-1])
        return paired_mean_log_e(
            moments.pooled(), null_value=0.0, tau=tau, sided="greater", min_n=min_n
        )
    champion, challenger = moments.pooled()
    if test == "harm":
        null, sided = 0.0, "greater"
    elif test == "non_inferiority":
        null, sided = float(guardrail.non_inferiority_margin or 0.0), "less"
    else:
        null, sided = 0.0, "less"
    return two_sample_mean_log_e(
        champion, challenger, null_value=null, tau=tau, sided=sided, min_n=min_n
    )


def _bernoulli_counts(stage: StageEvidence, guardrail: GuardrailSpec) -> PairedCounts | ArmCounts:
    if guardrail.metric is Metric.ERROR:
        return stage.error
    return stage.latency[guardrail.name]


def evaluate(
    spec: RolloutSpec, stages: Sequence[StageEvidence], subgroups: Sequence[str]
) -> list[TestResult]:
    """Every guardrail test, with log e-values summed over stages."""
    if not stages:
        raise ValueError("at least one stage of evidence is required")
    batch = stages[0].samples.shape
    plan: list[tuple[GuardrailSpec, TestKind, str | None]] = []
    for guardrail in spec.guardrails:
        if guardrail.subgroups:
            plan.extend((guardrail, "harm", name) for name in subgroups)
            continue
        plan.append((guardrail, "harm", None))
        if guardrail.has_non_inferiority:
            plan.append((guardrail, "non_inferiority", None))
        if spec.promotion.require_superiority_on == guardrail.name:
            plan.append((guardrail, "superiority", None))
    harm_tests = sum(1 for _, test, _ in plan if test == "harm")
    harm_threshold = math.log(harm_tests / spec.alpha_rollback)
    promotion_threshold = math.log(1.0 / spec.alpha_promotion)

    results: list[TestResult] = []
    for guardrail, test, subgroup in plan:
        total = np.zeros(batch)
        for stage in stages:
            if guardrail.metric is Metric.LOG_LOSS:
                total = total + _log_loss_stage_log_e(stage.log_loss, guardrail, test)
                continue
            counts = _bernoulli_counts(stage, guardrail)
            ratio = (
                1.0 if test != "non_inferiority" else float(guardrail.non_inferiority_ratio or 1)
            )
            if subgroup is None:
                per = _bernoulli_stage_log_e(
                    counts.pooled(),
                    kind=stage.kind,
                    weight=stage.weight,
                    test=test,
                    ratio=ratio,
                    strength=spec.prior_strength,
                )
                total = total + per[..., 0]
            else:
                per = _bernoulli_stage_log_e(
                    counts,
                    kind=stage.kind,
                    weight=stage.weight,
                    test=test,
                    ratio=ratio,
                    strength=spec.prior_strength,
                )
                total = total + per[..., list(subgroups).index(subgroup)]
        threshold = harm_threshold if test == "harm" else promotion_threshold
        results.append(TestResult(guardrail.name, test, subgroup, total, threshold))
    return results


class Decision(IntEnum):
    HOLD = 0
    ADVANCE = 1
    ROLLBACK = 2
    REQUEST_PROMOTION = 3
    ABORT = 4


@dataclass(frozen=True, slots=True)
class DecisionResult:
    decision: IntArray
    harm_breached: BoolArray
    promotion_evidence: BoolArray
    minimums_met: BoolArray


def decide(
    spec: RolloutSpec,
    *,
    stage_index: npt.ArrayLike,
    elapsed_seconds: npt.ArrayLike,
    stage_samples: npt.ArrayLike,
    results: Sequence[TestResult],
    drift_ok: npt.ArrayLike = True,
    score_breach: npt.ArrayLike = False,
    awaiting_approval: npt.ArrayLike = False,
) -> DecisionResult:
    """The stage rule. Rollback is always allowed; advancing needs every precondition.

    * ROLLBACK when any harm e-value crosses ``log(K / alpha_rollback)`` or the label-free score
      guard fires (challenger outputs drifted while inputs did not).
    * ADVANCE / REQUEST_PROMOTION when the stage's minimum samples and duration are met, inputs
      show no drift, and (when the stage requires it, and always for the last stage) every
      non-inferiority/superiority e-value crosses ``log(1 / alpha_promotion)``.
    * ABORT when the stage's maximum duration passes without meeting those conditions.
    """
    index = np.asarray(stage_index, dtype=np.int64)
    elapsed = np.asarray(elapsed_seconds, dtype=np.float64)
    samples = np.asarray(stage_samples, dtype=np.float64)
    waiting = np.asarray(awaiting_approval, dtype=bool)
    shape = np.broadcast_shapes(index.shape, elapsed.shape, samples.shape)
    last = len(spec.stages) - 1
    min_samples = np.array([s.min_samples for s in spec.stages], dtype=np.float64)[index]
    min_duration = np.array([s.min_duration_seconds for s in spec.stages])[index]
    max_duration = np.array([s.max_duration_seconds for s in spec.stages])[index]
    needs_ni = np.array([s.require_non_inferiority for s in spec.stages])[index] | (index == last)

    harm = np.zeros(shape, dtype=bool)
    promotion = np.ones(shape, dtype=bool)
    for result in results:
        if result.test == "harm":
            harm = harm | result.crossed
        else:
            promotion = promotion & result.crossed
    rollback = harm | np.asarray(score_breach, dtype=bool)
    minimums = (samples >= min_samples) & (elapsed >= min_duration)
    ready = minimums & np.asarray(drift_ok, dtype=bool) & (promotion | ~needs_ni)
    decision = np.full(shape, int(Decision.HOLD), dtype=np.int64)
    decision = np.where(ready & (index < last), int(Decision.ADVANCE), decision)
    decision = np.where(ready & (index == last), int(Decision.REQUEST_PROMOTION), decision)
    decision = np.where(~ready & (elapsed > max_duration), int(Decision.ABORT), decision)
    decision = np.where(waiting & ~rollback, int(Decision.HOLD), decision)
    decision = np.where(rollback, int(Decision.ROLLBACK), decision)
    return DecisionResult(decision, harm, promotion, minimums)
