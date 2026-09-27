"""Delayed-label joining: turn the prediction, shadow and label logs into per-stage evidence.

DuckDB reads the Parquet segments and joins router predictions with shadow predictions and
labels by request id. Only what was knowable at ``now`` is used (predictions with ``ts <= now``,
labels with ``received_at <= now``), and label-dependent statistics use *matured* requests only:
requests at least ``label_window_seconds`` old, whose labels (accepted only within that TTL)
have all arrived. Evaluating matured cohorts in request order keeps outcome-dependent label
delays (positives confirmed later than negatives) from biasing the comparison; see
``docs/adr/0003-matured-cohorts-for-delayed-labels.md``. Requests whose label never arrived are
excluded (assumed missing at random with respect to the arm). Latency statistics need no label
and use every request.

The result is a deterministic function of the logs and ``now``, which is what makes controller
reconciliation idempotent and replayable.
"""

from __future__ import annotations

import math
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
from progressive_model_delivery.control.spec import Metric, RolloutSpec, StageKind
from progressive_model_delivery.control.store import RolloutRecord
from progressive_model_delivery.ml.drift import BinnedReference
from progressive_model_delivery.models.features import NUMERIC_FEATURES, REGIONS
from progressive_model_delivery.registry.release import ReleaseManifest
from progressive_model_delivery.stats.msprt import Moments

__all__ = ["CollectedEvidence", "DriftStatus", "EvidenceReader"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
_EPS = 1e-7


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
    distinguishable; ``tests/unit/test_evidence_unmask.py`` pins this behaviour.
    """
    if isinstance(column, np.ma.MaskedArray):
        if column.dtype.kind == "f":
            filled: npt.NDArray[Any] = column.filled(np.nan)
            return filled
        return np.asarray(column.filled())
    return np.asarray(column)


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
            "score": np.zeros(0),
            "latency_ms": np.zeros(0),
            "ts": np.zeros(0),
            "label": np.zeros(0),
            "s_score": np.zeros(0),
            "s_latency": np.zeros(0),
            **{f"f_{spec.name}": np.zeros(0) for spec in NUMERIC_FEATURES},
        }
        if not router:
            return empty
        labels = self._files(self.labels_dir)
        shadow = self._files(self.shadow_dir)
        params: dict[str, Any] = {"router": router, "rid": rollout_id, "now": now}
        label_cte = "SELECT NULL::VARCHAR AS request_id, NULL::DOUBLE AS label WHERE FALSE"
        if labels:
            params["labels"] = labels
            label_cte = (
                "SELECT request_id, CAST(any_value(label) AS DOUBLE) AS label "
                "FROM read_parquet($labels) WHERE received_at <= $now GROUP BY request_id"
            )
        shadow_cte = (
            "SELECT NULL::VARCHAR AS request_id, NULL::DOUBLE AS s_score, "
            "NULL::DOUBLE AS s_latency WHERE FALSE"
        )
        if shadow:
            params["shadow"] = shadow
            shadow_cte = (
                "SELECT request_id, any_value(score) AS s_score, any_value(latency_ms) AS "
                "s_latency FROM read_parquet($shadow) WHERE rollout_id = $rid AND ts <= $now "
                "GROUP BY request_id"
            )
        query = f"""
            WITH p AS (
                SELECT * FROM read_parquet($router) WHERE rollout_id = $rid AND ts <= $now
            ),
            l AS ({label_cte}),
            s AS ({shadow_cte})
            SELECT p.stage_index, p.arm, p.subgroup, p.score, p.latency_ms, p.ts,
                   l.label, s.s_score, s.s_latency, {feature_cols}
            FROM p LEFT JOIN l USING (request_id) LEFT JOIN s USING (request_id)
            ORDER BY p.ts, p.request_id
        """  # noqa: S608 - only fixed column names are interpolated; values are bound parameters
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
        stage = np.asarray(data["stage_index"], dtype=np.int64)
        arm = np.asarray(data["arm"]).astype(str)
        groups = np.array(
            [REGIONS.index(str(g)) for g in data["subgroup"]], dtype=np.int64
        ).reshape(-1)
        score = np.asarray(data["score"], dtype=np.float64)
        latency = np.asarray(data["latency_ms"], dtype=np.float64)
        ts = np.asarray(data["ts"], dtype=np.float64)
        label_raw = np.asarray(data["label"], dtype=np.float64)
        s_score = np.asarray(data["s_score"], dtype=np.float64)
        s_latency = np.asarray(data["s_latency"], dtype=np.float64)
        labeled = ~np.isnan(label_raw) & (ts <= now - spec.label_window_seconds)
        label = np.nan_to_num(label_raw)
        has_shadow = ~np.isnan(s_score)
        on_challenger = arm == "challenger"
        threshold = np.where(
            on_challenger, challenger.decision_threshold, champion.decision_threshold
        )
        served_error = (score >= threshold).astype(np.float64) != label
        served_loss = _log_loss(label, score)
        shadow_score = np.nan_to_num(s_score)
        shadow_error = (shadow_score >= challenger.decision_threshold).astype(np.float64) != label
        shadow_loss = _log_loss(label, shadow_score)
        latency_guardrails = [g for g in spec.guardrails if g.metric is Metric.LATENCY_SLO]

        stages: list[StageEvidence] = []
        counts: dict[str, JsonValue] = {}
        for window in record.stages:
            in_stage = stage == window.index
            if window.kind is StageKind.SHADOW:
                pair = in_stage & has_shadow & ~on_challenger
                labeled_pair = pair & labeled
                latency_counts: dict[str, PairedCounts | ArmCounts] = {
                    g.name: _paired(
                        latency > float(g.slo_ms or 0),
                        np.nan_to_num(s_latency) > float(g.slo_ms or 0),
                        pair,
                        groups,
                    )
                    for g in latency_guardrails
                }
                evidence = StageEvidence(
                    kind=StageKind.SHADOW,
                    weight=0.0,
                    error=_paired(served_error, shadow_error, labeled_pair, groups),
                    latency=latency_counts,
                    log_loss=PairedMoments(
                        _moments(shadow_loss - served_loss, labeled_pair, groups)
                    ),
                    samples=np.asarray(float(labeled_pair.sum())),
                )
                counts[f"stage_{window.index}"] = {
                    "kind": "shadow",
                    "requests": int(in_stage.sum()),
                    "paired": int(pair.sum()),
                    "paired_labeled": int(labeled_pair.sum()),
                }
            else:
                champion_mask = in_stage & ~on_challenger
                challenger_mask = in_stage & on_challenger
                lab_a, lab_b = champion_mask & labeled, challenger_mask & labeled
                latency_counts = {
                    g.name: ArmCounts(
                        _count(champion_mask, groups),
                        _count(champion_mask & (latency > float(g.slo_ms or 0)), groups),
                        _count(challenger_mask, groups),
                        _count(challenger_mask & (latency > float(g.slo_ms or 0)), groups),
                    )
                    for g in latency_guardrails
                }
                evidence = StageEvidence(
                    kind=StageKind.CANARY,
                    weight=window.weight_bp / 10_000,
                    error=ArmCounts(
                        _count(lab_a, groups),
                        _count(lab_a & served_error, groups),
                        _count(lab_b, groups),
                        _count(lab_b & served_error, groups),
                    ),
                    latency=latency_counts,
                    log_loss=ArmMoments(
                        _moments(served_loss, lab_a, groups), _moments(served_loss, lab_b, groups)
                    ),
                    samples=np.asarray(float(lab_b.sum())),
                    exposure=np.asarray(float(challenger_mask.sum())),
                )
                counts[f"stage_{window.index}"] = {
                    "kind": "canary",
                    "weight": window.weight_bp / 10_000,
                    "requests": int(in_stage.sum()),
                    "challenger_served": int(challenger_mask.sum()),
                    "labeled_champion": int(lab_a.sum()),
                    "labeled_challenger": int(lab_b.sum()),
                }
            stages.append(evidence)

        current = stage == record.current.index
        if record.current.kind is StageKind.SHADOW:
            challenger_scores = s_score[current & has_shadow]
        else:
            challenger_scores = score[current & on_challenger]
        drift = self._drift(
            spec,
            {
                f"f_{s.name}": np.asarray(data[f"f_{s.name}"], dtype=np.float64)[current]
                for s in NUMERIC_FEATURES
            },
            groups[current],
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
