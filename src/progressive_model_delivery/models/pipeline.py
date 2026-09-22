"""Train, fingerprint, sign and register the demo champion and challengers.

Reproducibility: the datasets are generated from a seed, written to Parquet and fingerprinted;
each model's run manifest records the seed, dataset SHA-256s, package versions, parameters and
holdout metrics; the signed release pins the run manifest digest and names the champion release
as its parent (lineage). Training all six models takes a few seconds on one CPU core.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from progressive_model_delivery.ml.contracts import feature_frame_schema, validate_frame
from progressive_model_delivery.ml.manifest import create_manifest, seed_everything
from progressive_model_delivery.models.features import FEATURE_ORDER, REGIONS, SUBGROUP_FEATURE
from progressive_model_delivery.models.synthetic import make_dataset
from progressive_model_delivery.models.training import CATALOGUE, TrainedModel, train_model
from progressive_model_delivery.registry.release import ReleaseStore, SignedRelease
from progressive_model_delivery.registry.signing import SigningKey, TrustStore

__all__ = ["DemoModels", "DemoRegistry", "build_demo_models", "render_model_card"]

MODEL_NAME = "churn-risk"
_PACKAGES = ("numpy", "pandas", "pandera", "pydantic", "lightgbm", "scikit-learn", "scipy")


@dataclass(frozen=True, slots=True)
class DemoModels:
    """Published releases plus both quality views (see :class:`TrainedModel`)."""

    releases: Mapping[str, SignedRelease]
    metrics: Mapping[str, Mapping[str, float]]
    live_metrics: Mapping[str, Mapping[str, float]]
    dataset_sha256: tuple[str, ...]

    def release_id(self, name: str) -> str:
        return self.releases[name].release_id


def render_model_card(name: str, trained: TrainedModel, dataset_sha256: tuple[str, ...]) -> str:
    """A per-release model card, stored content-addressed next to the model text."""
    cfg = trained.config
    metrics = "\n".join(
        f"| {key} | {value:.4f} |" for key, value in sorted(trained.metrics.items())
    )
    intended = {
        "champion": "Current production baseline for the churn-risk demo.",
        "better": "Candidate replacement with more training data and capacity.",
        "identical": "A/A control: byte-identical to the champion model text.",
        "worse": "Deliberately under-fitted model for rollback demonstrations.",
        "subgroup_worse": "Trained with stale labels for the south region (subgroup regression).",
        "feature_order_bug": "Trained with two feature columns swapped (pipeline bug).",
    }.get(name, "Demo model.")
    return (
        f"# Model card: {MODEL_NAME} / {name}\n\n"
        f"{intended}\n\n"
        "- Task: binary classification of a synthetic churn event (no real customers).\n"
        f"- Algorithm: LightGBM gradient-boosted trees, {cfg.num_boost_round} rounds, "
        f"{cfg.num_leaves} leaves, learning rate {cfg.learning_rate}, seed {cfg.seed}.\n"
        f"- Training rows: {cfg.n_train}; datasets (SHA-256): {', '.join(dataset_sha256)}.\n"
        f"- Features (in order): {', '.join(FEATURE_ORDER)}.\n"
        "- Decision threshold: 0.5. Artifact format: LightGBM text (no pickle).\n"
        "- Limitations: synthetic data only; subgroup = region; metrics are on a synthetic "
        "holdout from the same generator.\n\n"
        "| holdout metric | value |\n|---|---|\n"
        f"{metrics}\n"
    )


def build_demo_models(
    root: Path,
    signing_key: SigningKey,
    *,
    seed: int = 20_260_922,
    n_train: int = 20_000,
    n_holdout: int = 20_000,
    names: tuple[str, ...] = tuple(CATALOGUE),
    clock_now: datetime | None = None,
) -> DemoModels:
    """Generate data, train the catalogue and publish signed releases under ``root``."""
    seed_everything(seed)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = validate_frame(make_dataset(n_train, seed=seed), feature_frame_schema(labeled=True))
    holdout = validate_frame(
        make_dataset(n_holdout, seed=seed + 1), feature_frame_schema(labeled=True)
    )
    train_path = data_dir / "train.parquet"
    holdout_path = data_dir / "holdout.parquet"
    train.to_parquet(train_path, index=False)
    holdout.to_parquet(holdout_path, index=False)

    store = ReleaseStore(root / "registry")
    created = clock_now or datetime.now(UTC)
    releases: dict[str, SignedRelease] = {}
    metrics: dict[str, dict[str, float]] = {}
    live_metrics: dict[str, dict[str, float]] = {}
    dataset_sha: tuple[str, ...] = ()
    ordered = ("champion", *(name for name in names if name != "champion"))
    for name in ordered:
        trained = train_model(CATALOGUE[name], train, holdout)
        manifest = create_manifest(
            seed=trained.config.seed,
            datasets=[train_path, holdout_path],
            params={"model": name, **{k: str(v) for k, v in trained.config.params().items()}},
            metrics=trained.metrics,
            packages=_PACKAGES,
            run_id=f"{name}-{seed}",
            clock=lambda: created,
        )
        dataset_sha = tuple(item.sha256 for item in manifest.datasets)
        releases[name] = store.publish(
            model_name=MODEL_NAME,
            release=name,
            model_text=trained.model_text,
            model_card=render_model_card(name, trained, dataset_sha),
            run_manifest=manifest,
            feature_order=trained.feature_order,
            categorical_levels={SUBGROUP_FEATURE: REGIONS},
            subgroup_feature=SUBGROUP_FEATURE,
            metrics=trained.metrics,
            references=trained.references,
            signing_key=signing_key,
            parent_release=None if name == "champion" else releases["champion"].release_id,
            created_at=created,
        )
        metrics[name] = dict(trained.metrics)
        live_metrics[name] = dict(trained.live_metrics)
    return DemoModels(
        releases=releases, metrics=metrics, live_metrics=live_metrics, dataset_sha256=dataset_sha
    )


@dataclass(frozen=True, slots=True)
class DemoRegistry:
    """Trained demo models with the signing key that published them and a matching trust store."""

    root: Path
    key: SigningKey
    models: DemoModels

    @classmethod
    def create(cls, root: Path, *, seed: int = 20_260_922) -> DemoRegistry:
        key = SigningKey.generate()
        return cls(root=root, key=key, models=build_demo_models(root, key, seed=seed))

    @property
    def store(self) -> ReleaseStore:
        return ReleaseStore(self.root / "registry")

    @property
    def trust(self) -> TrustStore:
        return TrustStore([self.key.public()])
