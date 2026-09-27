"""Deterministic LightGBM training of the demo champion and its challengers.

Models are trained single-threaded with ``deterministic=True`` so the same seed and data give the
same model text, and they are exported as LightGBM's text format only (never pickle). The
challenger catalogue covers the rollout scenarios of the spec:

* ``champion``: a deliberately modest model (small sample, shallow trees).
* ``better``: more data and capacity, so lower error and log-loss everywhere.
* ``identical``: exactly the champion (A/A).
* ``worse``: under-fitted, globally worse.
* ``subgroup_worse``: its ``south`` training rows carry labels from the stale label model (before
  the usage reversal in that region), so it regresses there only and looks fine globally.
* ``feature_order_bug``: trained on a frame whose ``tenure_months`` and ``monthly_spend`` values
  were swapped while the column names were not, the classic pipeline bug that passes every
  schema and name check and only shows up as bad predictions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from progressive_model_delivery.ml.drift import BinnedReference
from progressive_model_delivery.models.features import (
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    REGIONS,
    SUBGROUP_FEATURE,
)
from progressive_model_delivery.models.synthetic import label_probability

__all__ = [
    "CATALOGUE",
    "TrainedModel",
    "TrainingConfig",
    "evaluate_scores",
    "reference_distributions",
    "to_matrix",
    "train_model",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    name: str
    n_train: int
    num_boost_round: int
    num_leaves: int
    learning_rate: float
    seed: int
    min_data_in_leaf: int = 20
    stale_labels_in_region: str | None = None
    swap_feature_values: tuple[str, str] | None = None

    def params(self) -> dict[str, object]:
        return {
            "objective": "binary",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "verbose": -1,
        }


CATALOGUE: Mapping[str, TrainingConfig] = {
    "champion": TrainingConfig("champion", 3_000, 30, 7, 0.1, seed=11),
    "better": TrainingConfig("better", 20_000, 200, 31, 0.05, seed=12),
    "identical": TrainingConfig("champion", 3_000, 30, 7, 0.1, seed=11),
    "worse": TrainingConfig("worse", 800, 6, 3, 0.1, seed=13),
    "subgroup_worse": TrainingConfig(
        "subgroup_worse", 20_000, 200, 31, 0.05, seed=14, stale_labels_in_region="south"
    ),
    "feature_order_bug": TrainingConfig(
        "feature_order_bug",
        20_000,
        200,
        31,
        0.05,
        seed=15,
        swap_feature_values=("tenure_months", "monthly_spend"),
    ),
}


def to_matrix(frame: pd.DataFrame, feature_order: tuple[str, ...] = FEATURE_ORDER) -> FloatArray:
    """Feature matrix in ``feature_order``; the region becomes its integer code."""
    columns = []
    for name in feature_order:
        if name == SUBGROUP_FEATURE:
            codes = pd.Categorical(frame[name], categories=list(REGIONS)).codes
            if np.any(codes < 0):
                raise ValueError("unknown region in frame")
            columns.append(codes.astype(np.float64))
        else:
            columns.append(frame[name].to_numpy(dtype=np.float64))
    result: FloatArray = np.column_stack(columns)
    return result


def evaluate_scores(
    labels: npt.ArrayLike, scores: npt.ArrayLike, regions: npt.ArrayLike
) -> dict[str, float]:
    """Holdout metrics, including the error rate per region."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1 - 1e-7)
    region = np.asarray(regions)
    errors = (p >= 0.5).astype(np.int64) != y
    metrics = {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "error_rate": float(errors.mean()),
        "positive_rate": float(y.mean()),
    }
    for name in REGIONS:
        mask = region == name
        if mask.any():
            metrics[f"error_rate_{name}"] = float(errors[mask].mean())
    return metrics


def reference_distributions(
    holdout: pd.DataFrame, scores: npt.ArrayLike
) -> dict[str, dict[str, list[float]]]:
    """PSI references for every feature and for the model score on the holdout set."""
    references: dict[str, dict[str, list[float]]] = {}
    for spec in NUMERIC_FEATURES:
        references[spec.name] = BinnedReference.from_sample(
            holdout[spec.name].to_numpy(dtype=np.float64)
        ).to_json()
    codes = pd.Categorical(holdout[SUBGROUP_FEATURE], categories=list(REGIONS)).codes
    references[SUBGROUP_FEATURE] = BinnedReference.from_codes(
        codes, n_categories=len(REGIONS)
    ).to_json()
    references["__score__"] = BinnedReference.from_sample(scores).to_json()
    return references


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """A trained model with two views of its quality.

    ``metrics`` and ``references`` come from the training pipeline's own validation frame, which
    went through the same (possibly buggy) preparation as the training frame; they are what the
    signed manifest records. ``live_metrics`` score the clean holdout, which is what serving will
    actually see. For a healthy model the two agree; for the pipeline-bug challengers they do
    not, which is the point of progressive delivery.
    """

    config: TrainingConfig
    model_text: str
    metrics: dict[str, float]
    live_metrics: dict[str, float]
    references: dict[str, dict[str, list[float]]]
    feature_order: tuple[str, ...] = field(default=FEATURE_ORDER)


def _prepare_frame(frame: pd.DataFrame, config: TrainingConfig, *, seed: int) -> pd.DataFrame:
    """Apply the training pipeline's preparation (including its bugs) to a frame."""
    prepared = frame.copy()
    if config.stale_labels_in_region is not None:
        mask = (prepared[SUBGROUP_FEATURE] == config.stale_labels_in_region).to_numpy()
        rng = np.random.default_rng(seed)
        stale = label_probability(prepared.loc[mask], south_interaction=False)
        labels = prepared["label"].to_numpy().copy()
        labels[mask] = (rng.random(int(mask.sum())) < stale).astype(np.int64)
        prepared["label"] = labels
    if config.swap_feature_values is not None:
        first, second = config.swap_feature_values
        prepared[[first, second]] = prepared[[second, first]].to_numpy()
    return prepared


def train_model(
    config: TrainingConfig, train_frame: pd.DataFrame, holdout: pd.DataFrame
) -> TrainedModel:
    """Train one model, evaluate it (pipeline view and live view) and export LightGBM text."""
    if len(train_frame) < config.n_train:
        raise ValueError(f"{config.name} needs {config.n_train} rows, got {len(train_frame)}")
    prepared = _prepare_frame(train_frame.iloc[: config.n_train], config, seed=config.seed)
    validation = _prepare_frame(holdout, config, seed=config.seed + 1)
    dataset = lgb.Dataset(
        to_matrix(prepared),
        label=prepared["label"].to_numpy(dtype=np.float64),
        feature_name=list(FEATURE_ORDER),
        categorical_feature=[SUBGROUP_FEATURE],
        free_raw_data=True,
        params={"verbose": -1},
    )
    booster = lgb.train(config.params(), dataset, num_boost_round=config.num_boost_round)
    pipeline_scores = np.asarray(booster.predict(to_matrix(validation)), dtype=np.float64)
    live_scores = np.asarray(booster.predict(to_matrix(holdout)), dtype=np.float64)
    return TrainedModel(
        config=config,
        model_text=str(booster.model_to_string()),
        metrics=evaluate_scores(validation["label"], pipeline_scores, validation[SUBGROUP_FEATURE]),
        live_metrics=evaluate_scores(holdout["label"], live_scores, holdout[SUBGROUP_FEATURE]),
        references=reference_distributions(validation, pipeline_scores),
    )
