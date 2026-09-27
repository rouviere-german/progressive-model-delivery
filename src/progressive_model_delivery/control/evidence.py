"""Delayed-label joining: turn the prediction, shadow and label logs into per-stage evidence.

DuckDB reads the Parquet segments and joins router predictions with shadow predictions and
labels by request id. Only what was knowable at ``now`` is used (predictions with ``ts <= now``,
labels with ``received_at <= now``), and label-dependent statistics use *matured* requests only:
requests at least ``label_window_seconds`` old whose label arrived inside that window. Evaluating
matured cohorts in request order keeps outcome-dependent label delays (positives confirmed later
than negatives) from biasing the comparison; see
``docs/adr/0006-matured-cohorts-for-delayed-labels.md``. Requests whose label never arrived are
excluded (assumed missing at random with respect to the arm). Latency statistics need no label
and use every request.

A subject contributes at most one observation to each stage, and at a canary stage at most one per
arm: its first matured labelled request, and its first request for the label-free latency guardrail.
The arm comes from a consistent hash of the subject (``serving/router.py``), so a repeat visitor's
requests share one draw of the arm indicator and are not independent evidence for the canary
e-value's null; the paired shadow comparison clusters the same way, because one customer's
near-identical requests disagree in the same direction. Scoring per request instead inflates the A/A
false-rollback rate from 0.6% to 8% at five requests per subject (``bench/bench_validity.py``
measures both). Exposure and the request counts in ``counts`` stay per request: they are about
customers reached, not about evidence.

The window is enforced here rather than assumed: a label whose ``received_at`` is more than
``label_window_seconds`` after its request is counted in ``late_labels_ignored`` and never enters a
test, whatever the router's ``label_ttl_seconds`` happens to be. Without that, a TTL longer than the
window would fold late labels into cohorts that were already scored as matured -- exactly the
outcome-correlated bias the window exists to remove, and an e-process gaining increments for a past
cohort. ``ControlService.propose`` refuses the mismatched configuration as well.

The result is a deterministic function of the logs and ``now``, which is what makes controller
reconciliation idempotent and replayable. Every per-request aggregate is order-independent
(``min``, not ``any_value``), so duplicate rows cannot make the join depend on DuckDB's plan or on
segment file order; duplicates are counted into the evidence instead of being silently collapsed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from progressive_model_delivery.control.guardrails import (
    ArmCounts,
    ArmMoments,
    PairedCounts,
    PairedMoments,
    StageEvidence,
)
from progressive_model_delivery.control.spec import (
    GuardrailSpec,
    Metric,
    RolloutSpec,
    StageKind,
)
from progressive_model_delivery.control.store import RolloutRecord, StageWindow
from progressive_model_delivery.ml.drift import BinnedReference
from progressive_model_delivery.models.features import NUMERIC_FEATURES, REGIONS
from progressive_model_delivery.registry.release import ReleaseManifest
from progressive_model_delivery.stats.msprt import Moments

__all__ = ["CollectedEvidence", "DriftStatus", "EvidenceReader"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
_EPS = 1e-7

_JOIN_SQL = """
    WITH p AS (
        SELECT * FROM read_parquet($router) WHERE rollout_id = $rid AND ts <= $now
    ),
    l AS ({label_cte}),
    s AS ({shadow_cte})
    SELECT p.stage_index, p.arm, p.subgroup, p.subject_hash, p.score, p.latency_ms, p.ts,
           l.label, l.label_at, l.label_rows, s.s_score, s.s_latency, s.shadow_rows,
           {feature_cols}
    FROM p LEFT JOIN l USING (request_id) LEFT JOIN s USING (request_id)
    ORDER BY p.ts, p.request_id
"""
"""The delayed-label join.

The three placeholders are filled with fixed identifiers only: one of two constant label CTEs,
one of two constant shadow CTEs, and the ``NUMERIC_FEATURES`` column list. Every value -- parquet
paths, the rollout id and the maturation cutoff -- is a bound DuckDB parameter, so no caller input
ever reaches the SQL text.
"""


@dataclass(frozen=True, slots=True)
class DriftStatus:
    feature_psi: dict[str, float]
    score_psi: float | None
    samples: int
    drift_ok: bool
    score_breach: bool


@dataclass(frozen=True, slots=True)
class CollectedEvidence:
    stages: list[StageEvidence]
    drift: DriftStatus
    counts: dict[str, JsonValue]


def _unmask(column: Any) -> npt.NDArray[Any]:
    """Turn a DuckDB result column into a plain array, with SQL NULL as NaN.

    ``fetchnumpy`` returns a ``numpy.ma.MaskedArray`` for any column that contains NULLs, and
    ``np.asarray`` on such an array drops the mask and exposes *uninitialised* memory instead of
    the missing value. An unmatched outer join would then look like a real label of 0, which is a
    systematic bias against the better-calibrated model. Filling with NaN keeps "no label yet"
    distinguishable;
    ``tests/integration/test_evidence.py::test_unmask_turns_sql_nulls_into_nan_not_uninitialised_memory``
    pins this behaviour.
    """
    if isinstance(column, np.ma.MaskedArray):
        if column.dtype.kind == "f":
            filled: npt.NDArray[Any] = column.filled(np.nan)
            return filled
        return np.asarray(column.filled())
    return np.asarray(column)


def _first_per_subject(mask: BoolArray, subjects: npt.NDArray[Any]) -> BoolArray:
    """Keep only each subject's first selected row (rows arrive in ``(ts, request_id)`` order).

    ``np.unique(..., return_index=True)`` sorts stably, so the kept index is the earliest row for
    that subject: a deterministic choice, which replay depends on.
    """
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return mask
    _, first = np.unique(subjects[selected], return_index=True)
    keep = np.zeros(mask.shape, dtype=np.bool_)
    keep[selected[first]] = True
    return keep


def _log_loss(label: FloatArray, score: FloatArray) -> FloatArray:
    p = np.clip(score, _EPS, 1 - _EPS)
    result: FloatArray = -(label * np.log(p) + (1 - label) * np.log1p(-p))
    return result


def _by_group(values: FloatArray, mask: BoolArray, groups: npt.NDArray[np.int64]) -> FloatArray:
    result: FloatArray = np.bincount(
        groups[mask], weights=values[mask], minlength=len(REGIONS)
    ).astype(np.float64)
    return result


def _count(mask: BoolArray, groups: npt.NDArray[np.int64]) -> FloatArray:
    return _by_group(np.ones(mask.shape), mask, groups)


def _moments(values: FloatArray, mask: BoolArray, groups: npt.NDArray[np.int64]) -> Moments:
    return Moments(
        _count(mask, groups), _by_group(values, mask, groups), _by_group(values**2, mask, groups)
    )


def _paired(
    a: BoolArray, b: BoolArray, mask: BoolArray, groups: npt.NDArray[np.int64]
) -> PairedCounts:
    return PairedCounts(
        _count(mask & ~a & ~b, groups),
        _count(mask & ~a & b, groups),
        _count(mask & a & ~b, groups),
        _count(mask & a & b, groups),
    )


@dataclass(frozen=True, slots=True)
class _Rows:
    """The joined log as plain arrays, plus the per-request quantities every stage needs."""

    stage: npt.NDArray[np.int64]
    groups: npt.NDArray[np.int64]
    subject: npt.NDArray[Any]
    score: FloatArray
    latency: FloatArray
    s_score: FloatArray
    s_latency: FloatArray
    labeled: BoolArray
    has_shadow: BoolArray
    on_challenger: BoolArray
    served_error: BoolArray
    served_loss: FloatArray
    shadow_error: BoolArray
    shadow_loss: FloatArray
    quality: dict[str, JsonValue]
    """Counts that describe the join itself: late labels and duplicate rows, for the audit."""


def _rows(
    data: dict[str, Any],
    spec: RolloutSpec,
    now: float,
    *,
    champion: ReleaseManifest,
    challenger: ReleaseManifest,
) -> _Rows:
    """Turn the joined columns into decision-ready arrays (the maturation rule lives here)."""
    ts = np.asarray(data["ts"], dtype=np.float64)
    label_raw = np.asarray(data["label"], dtype=np.float64)
    # A missing arrival time sorts as "infinitely late", so it can never pass the window.
    label_at = np.nan_to_num(
        np.asarray(data["label_at"], dtype=np.float64), nan=np.inf, posinf=np.inf
    )
    arrived = ~np.isnan(label_raw)
    in_window = label_at <= ts + spec.label_window_seconds
    label = np.nan_to_num(label_raw)
    score = np.asarray(data["score"], dtype=np.float64)
    s_score = np.asarray(data["s_score"], dtype=np.float64)
    on_challenger = np.asarray(data["arm"]).astype(str) == "challenger"
    threshold = np.where(on_challenger, challenger.decision_threshold, champion.decision_threshold)
    shadow_score = np.nan_to_num(s_score)
    label_rows = np.nan_to_num(np.asarray(data["label_rows"], dtype=np.float64))
    shadow_rows = np.nan_to_num(np.asarray(data["shadow_rows"], dtype=np.float64))
    return _Rows(
        stage=np.asarray(data["stage_index"], dtype=np.int64),
        groups=np.array([REGIONS.index(str(g)) for g in data["subgroup"]], dtype=np.int64).reshape(
            -1
        ),
        subject=np.asarray(data["subject_hash"]).astype(str),
        score=score,
        latency=np.asarray(data["latency_ms"], dtype=np.float64),
        s_score=s_score,
        s_latency=np.nan_to_num(np.asarray(data["s_latency"], dtype=np.float64)),
        labeled=arrived & in_window & (ts <= now - spec.label_window_seconds),
        has_shadow=~np.isnan(s_score),
        on_challenger=on_challenger,
        served_error=(score >= threshold).astype(np.float64) != label,
        served_loss=_log_loss(label, score),
        shadow_error=(shadow_score >= challenger.decision_threshold).astype(np.float64) != label,
        shadow_loss=_log_loss(label, shadow_score),
        quality={
            # Observable poisoning attempts and configuration mistakes, in the audit evidence.
            "late_labels_ignored": int(np.count_nonzero(arrived & ~in_window)),
            "duplicate_label_requests": int(np.count_nonzero(label_rows > 1)),
            "duplicate_shadow_requests": int(np.count_nonzero(shadow_rows > 1)),
        },
    )


def _shadow_stage(
    window: StageWindow, rows: _Rows, guardrails: Sequence[GuardrailSpec]
) -> tuple[StageEvidence, dict[str, JsonValue]]:
    """Paired evidence: both models scored the same request, so nothing was assigned."""
    in_stage = rows.stage == window.index
    pair = in_stage & rows.has_shadow & ~rows.on_challenger
    # One comparison per subject: a repeat visitor's requests are one observation.
    pair_first = _first_per_subject(pair, rows.subject)
    labeled_pair = _first_per_subject(pair & rows.labeled, rows.subject)
    latency: dict[str, PairedCounts | ArmCounts] = {
        g.name: _paired(
            rows.latency > float(g.slo_ms or 0),
            rows.s_latency > float(g.slo_ms or 0),
            pair_first,
            rows.groups,
        )
        for g in guardrails
    }
    evidence = StageEvidence(
        kind=StageKind.SHADOW,
        weight=0.0,
        error=_paired(rows.served_error, rows.shadow_error, labeled_pair, rows.groups),
        latency=latency,
        log_loss=PairedMoments(
            _moments(rows.shadow_loss - rows.served_loss, labeled_pair, rows.groups)
        ),
        samples=np.asarray(float(labeled_pair.sum())),
    )
    return evidence, {
        "kind": "shadow",
        "requests": int(in_stage.sum()),
        "paired": int(pair.sum()),
        "paired_subjects": int(pair_first.sum()),
        "paired_labeled": int(labeled_pair.sum()),
    }


def _canary_stage(
    window: StageWindow, rows: _Rows, guardrails: Sequence[GuardrailSpec]
) -> tuple[StageEvidence, dict[str, JsonValue]]:
    """Two-arm evidence. The arm is drawn once per subject, so evidence counts subjects."""
    in_stage = rows.stage == window.index
    champion_mask = in_stage & ~rows.on_challenger
    challenger_mask = in_stage & rows.on_challenger
    champion_first = _first_per_subject(champion_mask, rows.subject)
    challenger_first = _first_per_subject(challenger_mask, rows.subject)
    lab_a = _first_per_subject(champion_mask & rows.labeled, rows.subject)
    lab_b = _first_per_subject(challenger_mask & rows.labeled, rows.subject)
    latency: dict[str, PairedCounts | ArmCounts] = {
        g.name: ArmCounts(
            _count(champion_first, rows.groups),
            _count(champion_first & (rows.latency > float(g.slo_ms or 0)), rows.groups),
            _count(challenger_first, rows.groups),
            _count(challenger_first & (rows.latency > float(g.slo_ms or 0)), rows.groups),
        )
        for g in guardrails
    }
    evidence = StageEvidence(
        kind=StageKind.CANARY,
        weight=window.weight_bp / 10_000,
        error=ArmCounts(
            _count(lab_a, rows.groups),
            _count(lab_a & rows.served_error, rows.groups),
            _count(lab_b, rows.groups),
            _count(lab_b & rows.served_error, rows.groups),
        ),
        latency=latency,
        log_loss=ArmMoments(
            _moments(rows.served_loss, lab_a, rows.groups),
            _moments(rows.served_loss, lab_b, rows.groups),
        ),
        samples=np.asarray(float(lab_b.sum())),
        exposure=np.asarray(float(challenger_mask.sum())),
    )
    return evidence, {
        "kind": "canary",
        "weight": window.weight_bp / 10_000,
        "requests": int(in_stage.sum()),
        "challenger_served": int(challenger_mask.sum()),
        "champion_subjects": int(champion_first.sum()),
        "challenger_subjects": int(challenger_first.sum()),
        "labeled_champion": int(lab_a.sum()),
        "labeled_challenger": int(lab_b.sum()),
    }


class EvidenceReader:
    def __init__(self, predlog_dir: Path) -> None:
        self.router_dir = predlog_dir / "router"
        self.shadow_dir = predlog_dir / "shadow"
        self.labels_dir = predlog_dir / "labels"

    @staticmethod
    def _files(directory: Path) -> list[str]:
        return sorted(str(p) for p in directory.glob("*.parquet")) if directory.exists() else []

    def _load(self, rollout_id: str, now: float) -> dict[str, Any]:
        router = self._files(self.router_dir)
        feature_cols = ", ".join(f"p.f_{spec.name}" for spec in NUMERIC_FEATURES)
        empty = {
            "stage_index": np.zeros(0, dtype=np.int64),
            "arm": np.zeros(0, dtype=object),
            "subgroup": np.zeros(0, dtype=object),
            "subject_hash": np.zeros(0, dtype=object),
            "score": np.zeros(0),
            "latency_ms": np.zeros(0),
            "ts": np.zeros(0),
            "label": np.zeros(0),
            "label_at": np.zeros(0),
            "label_rows": np.zeros(0),
            "s_score": np.zeros(0),
            "s_latency": np.zeros(0),
            "shadow_rows": np.zeros(0),
            **{f"f_{spec.name}": np.zeros(0) for spec in NUMERIC_FEATURES},
        }
        if not router:
            return empty
        labels = self._files(self.labels_dir)
        shadow = self._files(self.shadow_dir)
        params: dict[str, Any] = {"router": router, "rid": rollout_id, "now": now}
        label_cte = (
            "SELECT NULL::VARCHAR AS request_id, NULL::DOUBLE AS label, "
            "NULL::DOUBLE AS label_at, NULL::DOUBLE AS label_rows WHERE FALSE"
        )
        if labels:
            params["labels"] = labels
            label_cte = (
                "SELECT request_id, CAST(min(label) AS DOUBLE) AS label, "
                "CAST(min(received_at) AS DOUBLE) AS label_at, "
                "CAST(count(*) AS DOUBLE) AS label_rows "
                "FROM read_parquet($labels) WHERE received_at <= $now GROUP BY request_id"
            )
        shadow_cte = (
            "SELECT NULL::VARCHAR AS request_id, NULL::DOUBLE AS s_score, "
            "NULL::DOUBLE AS s_latency, NULL::DOUBLE AS shadow_rows WHERE FALSE"
        )
        if shadow:
            params["shadow"] = shadow
            shadow_cte = (
                "SELECT request_id, CAST(min(score) AS DOUBLE) AS s_score, "
                "CAST(min(latency_ms) AS DOUBLE) AS s_latency, "
                "CAST(count(*) AS DOUBLE) AS shadow_rows "
                "FROM read_parquet($shadow) WHERE rollout_id = $rid AND ts <= $now "
                "GROUP BY request_id"
            )
        query = _JOIN_SQL.format(
            label_cte=label_cte, shadow_cte=shadow_cte, feature_cols=feature_cols
        )
        with duckdb.connect(":memory:") as con:
            data = con.execute(query, params).fetchnumpy()
        return {key: _unmask(value) for key, value in data.items()}

    def collect(
        self,
        record: RolloutRecord,
        now: float,
        *,
        champion: ReleaseManifest,
        challenger: ReleaseManifest,
    ) -> CollectedEvidence:
        spec = record.spec
        data = self._load(record.rollout_id, now)
        rows = _rows(data, spec, now, champion=champion, challenger=challenger)
        guardrails = [g for g in spec.guardrails if g.metric is Metric.LATENCY_SLO]
        stages: list[StageEvidence] = []
        counts: dict[str, JsonValue] = dict(rows.quality)
        for window in record.stages:
            if window.kind is StageKind.SHADOW:
                evidence, stage_counts = _shadow_stage(window, rows, guardrails)
            else:
                evidence, stage_counts = _canary_stage(window, rows, guardrails)
            stages.append(evidence)
            counts[f"stage_{window.index}"] = stage_counts

        current = rows.stage == record.current.index
        if record.current.kind is StageKind.SHADOW:
            challenger_scores = rows.s_score[current & rows.has_shadow]
        else:
            challenger_scores = rows.score[current & rows.on_challenger]
        drift = self._drift(
            spec,
            {
                f"f_{s.name}": np.asarray(data[f"f_{s.name}"], dtype=np.float64)[current]
                for s in NUMERIC_FEATURES
            },
            rows.groups[current],
            challenger_scores,
            champion=champion,
            challenger=challenger,
        )
        return CollectedEvidence(stages=stages, drift=drift, counts=counts)

    @staticmethod
    def _drift(
        spec: RolloutSpec,
        features: dict[str, FloatArray],
        region_codes: npt.NDArray[np.int64],
        challenger_scores: FloatArray,
        *,
        champion: ReleaseManifest,
        challenger: ReleaseManifest,
    ) -> DriftStatus:
        n = int(region_codes.size)
        psi: dict[str, float] = {}
        if n >= spec.drift.min_samples:
            for spec_feature in NUMERIC_FEATURES:
                reference = champion.references.get(spec_feature.name)
                if reference is not None:
                    psi[spec_feature.name] = BinnedReference.from_json(reference).psi(
                        features[f"f_{spec_feature.name}"]
                    )
            region_reference = champion.references.get("region")
            if region_reference is not None:
                psi["region"] = BinnedReference.from_json(region_reference).psi(region_codes)
        drift_ok = n >= spec.drift.min_samples and all(
            value < spec.drift.feature_psi_threshold for value in psi.values()
        )
        score_psi: float | None = None
        score_reference = challenger.references.get("__score__")
        if score_reference is not None and challenger_scores.size >= spec.drift.min_samples:
            score_psi = BinnedReference.from_json(score_reference).psi(challenger_scores)
        score_breach = (
            drift_ok and score_psi is not None and score_psi > spec.drift.score_psi_threshold
        )
        return DriftStatus(
            feature_psi={k: round(v, 6) for k, v in psi.items() if math.isfinite(v)},
            score_psi=None if score_psi is None else round(score_psi, 6),
            samples=n,
            drift_ok=bool(drift_ok),
            score_breach=bool(score_breach),
        )
