from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from progressive_model_delivery.ml.drift import BinnedReference
from progressive_model_delivery.models.features import FEATURE_ORDER, REGIONS, region_code
from progressive_model_delivery.models.pipeline import DemoRegistry
from progressive_model_delivery.models.scoring import ModelScorer, ScoringError
from progressive_model_delivery.models.synthetic import (
    LabelDelay,
    label_probability,
    make_dataset,
    sample_features,
    sample_label_delays,
    subject_ids,
)
from progressive_model_delivery.models.training import CATALOGUE, to_matrix, train_model

pytestmark = pytest.mark.unit


def test_datasets_are_deterministic_and_region_shares_hold() -> None:
    a, b = make_dataset(5_000, seed=3), make_dataset(5_000, seed=3)
    pd.testing.assert_frame_equal(a, b)
    shares = a["region"].value_counts(normalize=True)
    assert shares["north"] == pytest.approx(0.40, abs=0.03)
    assert shares["south"] == pytest.approx(0.12, abs=0.02)
    assert 0.10 < a["label"].mean() < 0.25


def test_the_south_interaction_reverses_the_usage_effect() -> None:
    frame = sample_features(2, np.random.default_rng(0))
    frame["region"] = "south"
    frame["usage_trend"] = [-0.5, 0.5]
    current = label_probability(frame)
    stale = label_probability(frame, south_interaction=False)
    assert current[1] > current[0]  # usage now raises churn in south
    assert stale[1] < stale[0]  # it used to lower it


def test_label_delays_are_slower_for_positives_and_some_labels_are_lost() -> None:
    rng = np.random.default_rng(1)
    labels = np.array([0] * 20_000 + [1] * 20_000)
    delays = sample_label_delays(labels, rng, LabelDelay(100.0, 400.0, 0.05))
    finite = np.isfinite(delays)
    assert 0.04 < 1 - finite.mean() < 0.06
    assert np.median(delays[:20_000][finite[:20_000]]) < np.median(delays[20_000:][finite[20_000:]])
    with pytest.raises(ValueError, match="loss_rate"):
        LabelDelay(1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        LabelDelay(0.0, 1.0, 0.0)


def test_generator_validation_and_ids() -> None:
    with pytest.raises(ValueError, match="n must"):
        sample_features(0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="region_shares"):
        sample_features(5, np.random.default_rng(0), region_shares=(1.0,))
    ids = subject_ids(3, np.random.default_rng(0))
    assert all(item.startswith("cust-") and len(item) == 12 for item in ids)
    assert region_code("west") == 3
    with pytest.raises(ValueError, match="unknown region"):
        region_code("mars")


def test_training_is_deterministic_and_text_only() -> None:
    train, holdout = make_dataset(3_000, seed=1), make_dataset(1_000, seed=2)
    first = train_model(CATALOGUE["champion"], train, holdout)
    second = train_model(CATALOGUE["identical"], train, holdout)
    assert first.model_text == second.model_text
    assert first.model_text.startswith("tree\n")
    with pytest.raises(ValueError, match="needs"):
        train_model(CATALOGUE["better"], train, holdout)


def test_catalogue_scenarios_behave_as_designed(trained_registry: DemoRegistry) -> None:
    metrics = trained_registry.models.live_metrics
    champion = metrics["champion"]
    assert metrics["better"]["error_rate"] < champion["error_rate"]
    assert metrics["better"]["log_loss"] < champion["log_loss"]
    assert metrics["identical"] == champion
    assert metrics["worse"]["error_rate"] > champion["error_rate"] + 0.02
    # The subgroup challenger looks fine globally but regresses in the south region.
    subgroup = metrics["subgroup_worse"]
    assert subgroup["error_rate"] <= champion["error_rate"] + 0.005
    assert subgroup["error_rate_south"] > champion["error_rate_south"] + 0.02
    assert metrics["feature_order_bug"]["auc"] < 0.7
    # The pipeline bugs are invisible in the pipeline's own (signed) validation metrics.
    pipeline = trained_registry.models.metrics
    assert pipeline["feature_order_bug"]["auc"] > 0.85
    assert pipeline["subgroup_worse"]["error_rate_south"] < subgroup["error_rate_south"]


def test_scorer_validates_features() -> None:
    train = make_dataset(500, seed=4)
    dataset = lgb.Dataset(
        to_matrix(train),
        label=train["label"].to_numpy(dtype=np.float64),
        feature_name=list(FEATURE_ORDER),
        categorical_feature=["region"],
        params={"verbose": -1},
    )
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 1}, dataset, 3)
    scorer = ModelScorer(
        booster, feature_order=FEATURE_ORDER, categorical_levels={"region": REGIONS}
    )
    row = {str(k): v for k, v in train.drop(columns=["label"]).iloc[0].to_dict().items()}
    assert 0.0 <= scorer.predict_one(row) <= 1.0
    assert scorer.predict_many([]).shape == (0,)
    for broken, message in (
        ({**row, "region": "mars"}, "unknown category"),
        ({k: v for k, v in row.items() if k != "tenure_months"}, "missing feature"),
        ({**row, "discount_rate": float("nan")}, "finite"),
        ({**row, "discount_rate": "0.1"}, "numeric"),
        ({**row, "num_products": True}, "numeric"),
    ):
        with pytest.raises(ScoringError, match=message):
            scorer.predict_one(broken)


def test_binned_reference_psi_detects_shift_and_round_trips() -> None:
    rng = np.random.default_rng(0)
    reference = BinnedReference.from_sample(rng.normal(size=5_000))
    assert reference.psi(rng.normal(size=5_000)) < 0.02
    assert reference.psi(rng.normal(loc=1.0, size=5_000)) > 0.2
    assert BinnedReference.from_json(reference.to_json()) == reference
    codes = BinnedReference.from_codes([0, 1, 1, 2, 3, 3, 3, 3], n_categories=4)
    assert codes.fractions == (0.125, 0.25, 0.125, 0.5)
    assert codes.psi([3] * 8) > 1.0
    with pytest.raises(ValueError, match="one more fraction"):
        BinnedReference((0.0,), (1.0,))
    with pytest.raises(ValueError, match="sorted"):
        BinnedReference((1.0, 0.0), (0.2, 0.3, 0.5))
    with pytest.raises(ValueError, match="sum to 1"):
        BinnedReference((0.0,), (0.2, 0.3))
    with pytest.raises(ValueError, match="out of range"):
        BinnedReference.from_codes([5], n_categories=4)
    with pytest.raises(ValueError, match="bins"):
        BinnedReference.from_sample([1.0, 2.0], bins=1)
    with pytest.raises(ValueError, match="n_categories"):
        BinnedReference.from_codes([0], n_categories=1)
