"""Delayed-label joining: what the controller is allowed to know at a given moment.

The evidence reader is the only place where "a label arrived" turns into "this request counts".
Its rules are tested against hand-written logs, because these are the cases that silently bias a
comparison if they are wrong:

* a request whose label has not arrived yet is not an observation, and must never be read as a
  label of 0 (the DuckDB masked-array trap);
* a request younger than the maturation window is excluded even if its label is already there,
  because label delay correlates with the outcome;
* latency needs no label, so latency statistics use every request;
* shadow rows are paired with the router row of the same request id, and only for that rollout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from progressive_model_delivery.control.evidence import EvidenceReader, _unmask
from progressive_model_delivery.control.guardrails import (
    ArmCounts,
    ArmMoments,
    PairedCounts,
)
from progressive_model_delivery.control.spec import RolloutSpec, StageKind, load_spec
from progressive_model_delivery.control.store import ControlStore, RolloutRecord, StageWindow
from progressive_model_delivery.models.features import NUMERIC_FEATURES, REGIONS
from progressive_model_delivery.models.pipeline import DemoRegistry
from progressive_model_delivery.models.synthetic import sample_features
from progressive_model_delivery.registry.release import ReleaseManifest
from progressive_model_delivery.serving.predlog import LABEL_SCHEMA, SegmentWriter

pytestmark = pytest.mark.integration

SPEC_YAML = """
name: evidence-test
version: 1
label_window_seconds: 100
prior_strength: 10
stages:
  - kind: shadow
    min_samples: 10
    min_duration_seconds: 0
  - kind: canary
    weight_percent: 25
    min_samples: 10
    min_duration_seconds: 0
guardrails:
  - name: error_rate
    metric: error
    non_inferiority_ratio: 1.1
  - name: worst_subgroup_error
    metric: error
    subgroups: true
  - name: latency_slo
    metric: latency_slo
    slo_ms: 20
  - name: log_loss
    metric: log_loss
    non_inferiority_margin: 0.05
drift:
  min_samples: 50
"""
ROLLOUT_ID = "1" * 32
OTHER_ROLLOUT = "2" * 32


@pytest.fixture
def spec() -> RolloutSpec:
    return load_spec(SPEC_YAML)


@pytest.fixture
def manifests(trained_registry: DemoRegistry) -> tuple[ReleaseManifest, ReleaseManifest]:
    store = trained_registry.store
    champion = store.read(trained_registry.models.release_id("champion")).manifest
    challenger = store.read(trained_registry.models.release_id("better")).manifest
    return champion, challenger


def _record(spec: RolloutSpec, *, stage_index: int) -> RolloutRecord:
    windows = [StageWindow(index=0, kind=StageKind.SHADOW, weight_bp=0, started_at=0.0)]
    if stage_index >= 1:
        windows.append(
            StageWindow(index=1, kind=StageKind.CANARY, weight_bp=2_500, started_at=500.0)
        )
    return RolloutRecord(
        rollout_id=ROLLOUT_ID,
        champion="a" * 64,
        challenger="b" * 64,
        spec=spec,
        spec_sha256=spec.sha256(),
        status="active",
        stage_index=stage_index,
        stages=tuple(windows),
        proposed_by="dana",
        created_at=0.0,
        salt="c" * 32,
        revision=0,
    )


class Log:
    """A hand-written prediction, shadow and label log over realistic feature rows.

    Features and regions are drawn from the same generator the router sees in production, so the
    drift statistics in these tests are computed against the champion's real PSI references.
    """

    def __init__(self, root: Path, *, seed: int = 21, pool: int = 512) -> None:
        self.router = SegmentWriter(root / "router", prefix="router")
        self.shadow = SegmentWriter(root / "shadow", prefix="shadow")
        self.labels = SegmentWriter(root / "labels", schema=LABEL_SCHEMA, prefix="labels")
        self.root = root
        self._n = 0
        frame = sample_features(pool, np.random.default_rng(seed))
        # to_dict types its keys as Hashable; this frame's columns are the str FEATURES.
        self._pool = cast("list[dict[str, Any]]", frame.to_dict(orient="records"))

    def request(
        self,
        *,
        score: float,
        ts: float,
        stage_index: int = 0,
        arm: str = "champion",
        subgroup: str | None = None,
        latency_ms: float = 5.0,
        rollout_id: str = ROLLOUT_ID,
        shadow_score: float | None = None,
        shadow_latency_ms: float = 5.0,
        label: int | None = None,
        label_at: float | None = None,
        subject_hash: str | None = None,
    ) -> str:
        self._n += 1
        request_id = f"{self._n:032x}"
        sampled = self._pool[self._n % len(self._pool)]
        features = {f"f_{s.name}": float(sampled[s.name]) for s in NUMERIC_FEATURES}
        subgroup = str(sampled["region"]) if subgroup is None else subgroup
        row: dict[str, Any] = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "ts": ts,
            "stage_index": stage_index,
            "arm": arm,
            "weight": 0.25 if stage_index else 0.0,
            "release_id": "b" * 64 if arm == "challenger" else "a" * 64,
            "subject_hash": f"{self._n:032x}" if subject_hash is None else subject_hash,
            "subgroup": subgroup,
            "score": score,
            "latency_ms": latency_ms,
            **features,
        }
        self.router.append(row)
        if shadow_score is not None:
            self.shadow.append(
                {
                    **row,
                    "arm": "shadow",
                    "weight": 0.0,
                    "release_id": "b" * 64,
                    "score": shadow_score,
                    "latency_ms": shadow_latency_ms,
                }
            )
        if label is not None:
            self.labels.append(
                {
                    "request_id": request_id,
                    "label": label,
                    "received_at": ts + 1.0 if label_at is None else label_at,
                }
            )
        return request_id

    def flush(self) -> None:
        for writer in (self.router, self.shadow, self.labels):
            writer.flush()


@pytest.fixture
def log(tmp_path: Path) -> Log:
    return Log(tmp_path / "predlog")


def _collect(
    log: Log,
    spec: RolloutSpec,
    manifests: tuple[ReleaseManifest, ReleaseManifest],
    *,
    now: float,
    stage_index: int = 0,
) -> Any:
    log.flush()
    reader = EvidenceReader(log.root)
    champion, challenger = manifests
    return reader.collect(
        _record(spec, stage_index=stage_index), now, champion=champion, challenger=challenger
    )


# --- the masked-array trap ----------------------------------------------------------------------


def test_unmask_turns_sql_nulls_into_nan_not_uninitialised_memory() -> None:
    masked = np.ma.masked_array([1.5, 0.0, 0.25], mask=[False, True, False])
    filled = _unmask(masked)
    assert not isinstance(filled, np.ma.MaskedArray)
    assert np.isnan(filled[1])
    assert filled[0] == 1.5
    strings = np.ma.masked_array(np.array(["a", "b"], dtype=object), mask=[False, True])
    assert _unmask(strings).shape == (2,)
    plain = np.array([1, 2, 3])
    assert _unmask(plain) is not None
    assert _unmask(plain).tolist() == [1, 2, 3]


def test_a_request_without_a_label_is_not_an_observation(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    """Regression: unlabelled rows once read as label 0 and penalised the better model."""
    for _ in range(40):  # both models say "positive", the label never arrives
        log.request(score=0.9, ts=0.0, shadow_score=0.95)
    for _ in range(10):  # a labelled, matured pair where both are right
        log.request(score=0.9, ts=0.0, shadow_score=0.95, label=1)
    collected = _collect(log, spec, manifests, now=1_000.0)
    counts = collected.stages[0].error
    assert isinstance(counts, PairedCounts)
    assert float(counts.total) == 10.0
    assert collected.counts["stage_0"] == {
        "kind": "shadow",
        "requests": 50,
        "paired": 50,
        "paired_subjects": 50,
        "paired_labeled": 10,
    }


# --- maturation ---------------------------------------------------------------------------------


def test_a_request_younger_than_the_label_window_is_excluded(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    for index in range(20):
        log.request(score=0.9, ts=float(index), shadow_score=0.9, label=1)
    # window is 100 s: at now = 105 only requests with ts <= 5 have matured.
    matured = _collect(log, spec, manifests, now=105.0)
    assert float(matured.stages[0].error.total) == 6.0
    later = _collect(log, spec, manifests, now=200.0)
    assert float(later.stages[0].error.total) == 20.0


def test_predictions_and_labels_from_the_future_are_invisible(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    log.request(score=0.9, ts=10.0, shadow_score=0.9, label=1, label_at=11.0)
    log.request(score=0.9, ts=500.0, shadow_score=0.9, label=1, label_at=501.0)
    collected = _collect(log, spec, manifests, now=300.0)
    assert collected.counts["stage_0"]["requests"] == 1
    # The label of the visible request arrived before `now`, so it counts.
    assert float(collected.stages[0].error.total) == 1.0

    late_label = _collect(log, spec, manifests, now=10_000.0)
    assert float(late_label.stages[0].error.total) == 2.0


def test_a_label_that_arrives_after_now_does_not_count_yet(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    log.request(score=0.9, ts=0.0, shadow_score=0.9, label=1, label_at=5_000.0)
    collected = _collect(log, spec, manifests, now=1_000.0)
    error = collected.stages[0].error
    assert isinstance(error, PairedCounts)  # the shadow stage is always paired
    assert float(error.total) == 0.0
    assert collected.counts["stage_0"]["paired"] == 1


def test_a_label_that_arrived_after_its_window_never_counts(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    """A label the router should have refused must not fold into an already-matured cohort.

    ``/v1/feedback`` accepts a label only inside ``APP_LABEL_TTL_SECONDS``, and the control plane
    refuses a spec whose ``label_window_seconds`` is shorter than that TTL -- but the reader is the
    component whose output the e-processes consume, so it enforces the invariant itself: a label
    that arrived more than ``label_window_seconds`` after its request is not evidence, at any
    ``now``. Otherwise a cohort scored as matured at ``now`` gains observations later, and the
    anytime-valid argument (increments over disjoint, already-published cohorts) breaks.
    """
    log.request(score=0.9, ts=0.0, shadow_score=0.9, label=1, label_at=5_000.0)
    log.request(score=0.9, ts=0.0, shadow_score=0.9, label=1, label_at=50.0)
    for now in (200.0, 1_000.0, 6_000.0, 10_000.0):
        collected = _collect(log, spec, manifests, now=now)
        error = collected.stages[0].error
        assert isinstance(error, PairedCounts)
        assert float(error.total) == (1.0 if now >= 150.0 else 0.0), now
        assert collected.counts["stage_0"]["paired"] == 2
    assert collected.counts["late_labels_ignored"] == 1


def test_a_duplicate_label_row_in_another_segment_is_collapsed(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    """One writer cannot duplicate a label (the contract forbids it); two segments could."""
    request_id = log.request(score=0.9, ts=0.0, shadow_score=0.9, label=1)
    log.flush()
    log.labels.append({"request_id": request_id, "label": 1, "received_at": 2.0})
    collected = _collect(log, spec, manifests, now=1_000.0)
    assert float(collected.stages[0].error.total) == 1.0
    assert collected.counts["duplicate_label_requests"] == 1


def test_conflicting_duplicate_rows_collapse_deterministically_and_are_counted(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    """Which duplicate wins may not depend on DuckDB's plan or on segment file order.

    Replay is a claimed property (``control/controller.py``), so every per-request aggregate in the
    join is order-independent: the smallest label, the earliest arrival, the smallest shadow score
    and latency. A disagreement is a poisoning attempt or a writer bug, so it is also counted into
    the evidence rather than quietly collapsed.
    """
    request_id = log.request(score=0.9, ts=0.0, shadow_score=0.9, label=1, subgroup="north")
    log.flush()
    log.labels.append({"request_id": request_id, "label": 0, "received_at": 3.0})
    sampled = sample_features(1, np.random.default_rng(5)).to_dict(orient="records")[0]
    log.shadow.append(
        {
            "request_id": request_id,
            "rollout_id": ROLLOUT_ID,
            "ts": 0.0,
            "stage_index": 0,
            "arm": "shadow",
            "weight": 0.0,
            "release_id": "b" * 64,
            "subject_hash": "0" * 32,
            "subgroup": "north",
            "score": 0.1,
            "latency_ms": 99.0,
            **{f"f_{s.name}": float(sampled[s.name]) for s in NUMERIC_FEATURES},
        }
    )
    results = [_collect(log, spec, manifests, now=1_000.0) for _ in range(5)]
    for collected in results:
        error = collected.stages[0].error
        assert isinstance(error, PairedCounts)
        # label 0 wins (the smaller), so the champion's 0.9 is wrong and the shadow's 0.1 is right.
        assert float(error.n01.sum()) == 0.0
        assert float(error.n10.sum()) == 1.0
        assert collected.counts["duplicate_label_requests"] == 1
        assert collected.counts["duplicate_shadow_requests"] == 1
        latency = collected.stages[0].latency["latency_slo"]
        assert isinstance(latency, PairedCounts)
        # min(latency_ms) = 5 ms for the shadow row, so neither arm exceeds the 20 ms SLO.
        assert float(latency.n00.sum()) == 1.0


# --- what the paired and two-arm tables contain --------------------------------------------------


def test_the_shadow_table_counts_disagreements_in_the_right_direction(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    # label 1: a score below 0.5 is wrong. Champion right, challenger wrong -> n01.
    for _ in range(7):
        log.request(score=0.9, ts=0.0, shadow_score=0.1, label=1, subgroup="north")
    # Champion wrong, challenger right -> n10.
    for _ in range(3):
        log.request(score=0.1, ts=0.0, shadow_score=0.9, label=1, subgroup="north")
    for _ in range(5):  # both right -> n00
        log.request(score=0.9, ts=0.0, shadow_score=0.8, label=1, subgroup="north")
    for _ in range(2):  # both wrong -> n11
        log.request(score=0.1, ts=0.0, shadow_score=0.2, label=1, subgroup="north")
    counts = _collect(log, spec, manifests, now=1_000.0).stages[0].error
    assert isinstance(counts, PairedCounts)
    north = REGIONS.index("north")
    assert float(counts.n01[north]) == 7.0
    assert float(counts.n10[north]) == 3.0
    assert float(counts.n00[north]) == 5.0
    assert float(counts.n11[north]) == 2.0


def test_subgroups_are_separated(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    for _ in range(4):
        log.request(score=0.9, ts=0.0, shadow_score=0.1, label=1, subgroup="south")
    for _ in range(6):
        log.request(score=0.9, ts=0.0, shadow_score=0.9, label=1, subgroup="north")
    counts = _collect(log, spec, manifests, now=1_000.0).stages[0].error
    assert isinstance(counts, PairedCounts)
    assert float(counts.n01[REGIONS.index("south")]) == 4.0
    assert float(counts.n01[REGIONS.index("north")]) == 0.0
    assert float(counts.pooled().n01[0]) == 4.0


def test_latency_uses_every_request_not_only_the_labelled_ones(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    for _ in range(9):  # challenger over the 20 ms SLO, champion under it, no labels at all
        log.request(score=0.9, ts=0.0, latency_ms=5.0, shadow_score=0.9, shadow_latency_ms=30.0)
    collected = _collect(log, spec, manifests, now=1_000.0)
    latency = collected.stages[0].latency["latency_slo"]
    assert isinstance(latency, PairedCounts)
    assert float(latency.n01.sum()) == 9.0
    error = collected.stages[0].error
    assert isinstance(error, PairedCounts)  # the shadow stage is always paired
    assert float(error.total) == 0.0


def test_a_canary_stage_splits_by_arm_and_counts_exposure(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    for _ in range(12):
        log.request(score=0.9, ts=600.0, stage_index=1, arm="champion", label=1)
    for _ in range(4):
        log.request(score=0.1, ts=600.0, stage_index=1, arm="challenger", label=1)
    for _ in range(2):  # served but not labelled yet: exposure without an observation
        log.request(score=0.1, ts=600.0, stage_index=1, arm="challenger")
    collected = _collect(log, spec, manifests, now=1_000.0, stage_index=1)
    counts = collected.stages[1].error
    assert isinstance(counts, ArmCounts)
    assert float(counts.n_champion.sum()) == 12.0
    assert float(counts.k_champion.sum()) == 0.0
    assert float(counts.n_challenger.sum()) == 4.0
    assert float(counts.k_challenger.sum()) == 4.0
    assert float(collected.stages[1].samples) == 4.0
    assert float(collected.stages[1].exposure) == 6.0
    assert collected.counts["stage_1"]["challenger_served"] == 6


def test_another_rollouts_rows_are_never_read(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    for _ in range(5):
        log.request(score=0.9, ts=0.0, shadow_score=0.1, label=1)
    for _ in range(50):
        log.request(score=0.1, ts=0.0, shadow_score=0.9, label=1, rollout_id=OTHER_ROLLOUT)
    collected = _collect(log, spec, manifests, now=1_000.0)
    assert collected.counts["stage_0"]["requests"] == 5
    counts = collected.stages[0].error
    assert isinstance(counts, PairedCounts)
    assert float(counts.n01.sum()) == 5.0
    assert float(counts.n10.sum()) == 0.0


def test_an_empty_log_yields_zero_evidence(
    tmp_path: Path, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    reader = EvidenceReader(tmp_path / "nothing")
    champion, challenger = manifests
    collected = reader.collect(
        _record(spec, stage_index=0), 100.0, champion=champion, challenger=challenger
    )
    error = collected.stages[0].error
    assert isinstance(error, PairedCounts)  # the shadow stage is always paired
    assert float(error.total) == 0.0
    assert collected.drift.samples == 0
    assert collected.drift.drift_ok is False
    assert collected.drift.score_psi is None


def test_the_log_survives_missing_shadow_and_label_directories(
    tmp_path: Path, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    log = Log(tmp_path / "partial")
    log.request(score=0.9, ts=0.0)
    log.router.flush()
    reader = EvidenceReader(log.root)
    champion, challenger = manifests
    collected = reader.collect(
        _record(spec, stage_index=0), 1_000.0, champion=champion, challenger=challenger
    )
    stage = collected.counts["stage_0"]
    assert isinstance(stage, dict)
    assert stage["paired"] == 0


# --- drift --------------------------------------------------------------------------------------


def test_drift_needs_a_minimum_sample_before_it_reports_anything(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    for _ in range(10):
        log.request(score=0.5, ts=0.0, shadow_score=0.5)
    small = _collect(log, spec, manifests, now=1_000.0)
    assert small.drift.feature_psi == {}
    assert small.drift.drift_ok is False
    assert small.drift.samples == 10


def test_a_shifted_feature_distribution_fails_the_drift_precondition(
    tmp_path: Path, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    log = Log(tmp_path / "drifted")
    rows = 200
    for _ in range(rows):
        log.request(score=0.5, ts=0.0, shadow_score=0.5)
    # Rewrite the buffered rows with an extreme monthly_spend so PSI explodes.
    for row in log.router._buffer:
        row["f_monthly_spend"] = 499.0
    collected = _collect(log, spec, manifests, now=1_000.0)
    assert collected.drift.samples == rows
    assert collected.drift.feature_psi["monthly_spend"] > 0.2
    assert collected.drift.drift_ok is False


def test_the_score_guard_fires_when_outputs_move_while_inputs_do_not(
    tmp_path: Path, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    """A label-free signal: stable features but a challenger whose score distribution moved."""
    log = Log(tmp_path / "score-drift")
    for _ in range(300):
        log.request(score=0.5, ts=0.0, shadow_score=0.999)
    collected = _collect(log, spec, manifests, now=1_000.0)
    assert collected.drift.drift_ok is True  # inputs are fine
    assert collected.drift.score_psi is not None
    assert collected.drift.score_psi > spec.drift.score_psi_threshold
    assert collected.drift.score_breach is True


def test_the_controller_can_read_a_log_written_by_the_router(
    tmp_path: Path, spec: RolloutSpec, trained_registry: DemoRegistry
) -> None:
    """A sanity check that the hand-written log in this module matches the real schema."""
    store = ControlStore(tmp_path / "control.sqlite")
    assert store.champion() is None
    log = Log(tmp_path / "schema")
    log.request(score=0.5, ts=0.0, shadow_score=0.5, label=1)
    log.flush()
    written = pd.read_parquet(next((log.root / "router").glob("*.parquet")))
    assert set(written.columns) >= {
        "request_id",
        "rollout_id",
        "ts",
        "stage_index",
        "arm",
        "weight",
        "release_id",
        "subject_hash",
        "subgroup",
        "score",
        "latency_ms",
        *(f"f_{s.name}" for s in NUMERIC_FEATURES),
    }
    assert "subject_id" not in written.columns


def test_a_subject_contributes_one_observation_to_a_stage(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    """A repeat visitor is one observation per arm, not one per request.

    The canary e-value's null is ``P(event on the challenger arm) = w r / (w r + 1 - w)``, which
    needs one independent draw of the arm indicator per observation. The router draws the arm once
    per subject (a consistent hash of the HMAC-ed subject id -- that stickiness is the product
    requirement), so five requests from one customer are one draw. Counting them as five
    over-counts the evidence in the direction that matters: ``bench/bench_validity.py`` measures
    the A/A false-rollback rate at 8% instead of 0.6% when five-request subjects are scored per
    request. The reader therefore keeps each subject's first matured observation per stage and arm.
    """
    for subject in ("a" * 32, "b" * 32, "c" * 32):
        for _ in range(5):
            # score 0.9 against label 0: every one of these requests is an error, and slow.
            log.request(
                score=0.9,
                ts=0.0,
                stage_index=1,
                arm="challenger",
                label=0,
                subject_hash=subject,
                latency_ms=30.0,
            )
    for subject in ("d" * 32, "e" * 32):
        for _ in range(4):
            log.request(
                score=0.9, ts=0.0, stage_index=1, arm="champion", label=1, subject_hash=subject
            )
    collected = _collect(log, spec, manifests, now=1_000.0, stage_index=1)

    error = collected.stages[1].error
    assert isinstance(error, ArmCounts)
    assert float(error.n_challenger.sum()) == 3.0
    assert float(error.k_challenger.sum()) == 3.0
    assert float(error.n_champion.sum()) == 2.0
    log_loss = collected.stages[1].log_loss
    assert isinstance(log_loss, ArmMoments)
    assert float(log_loss.challenger.count.sum()) == 3.0
    latency = collected.stages[1].latency["latency_slo"]
    assert isinstance(latency, ArmCounts)
    assert float(latency.n_challenger.sum()) == 3.0
    assert float(latency.k_challenger.sum()) == 3.0

    stage = collected.counts["stage_1"]
    assert isinstance(stage, dict)
    # Exposure is still every request the challenger answered: customers, not observations.
    assert stage["challenger_served"] == 15
    assert stage["challenger_subjects"] == 3
    assert stage["labeled_challenger"] == 3
    assert float(collected.stages[1].exposure) == 15.0


def test_a_subject_contributes_one_paired_observation_to_the_shadow_stage(
    log: Log, spec: RolloutSpec, manifests: tuple[ReleaseManifest, ReleaseManifest]
) -> None:
    """Shadow pairs cluster by subject too: one customer's requests carry one model comparison.

    Nothing is assigned in the shadow stage -- both models score every request -- but the paired
    (sequential McNemar) e-value still needs independent discordant pairs, and a repeat visitor's
    near-identical requests disagree in the same direction. The same rule applies.
    """
    for subject in ("a" * 32, "b" * 32):
        for _ in range(6):
            log.request(
                score=0.9, ts=0.0, shadow_score=0.1, label=1, subject_hash=subject, subgroup="north"
            )
    collected = _collect(log, spec, manifests, now=1_000.0)
    error = collected.stages[0].error
    assert isinstance(error, PairedCounts)
    assert float(error.total) == 2.0
    assert float(error.n01.sum()) == 2.0
    stage = collected.counts["stage_0"]
    assert isinstance(stage, dict)
    assert stage["paired"] == 12
    assert stage["paired_subjects"] == 2
    assert stage["paired_labeled"] == 2
