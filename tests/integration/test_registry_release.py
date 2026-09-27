from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from progressive_model_delivery.models.features import REGIONS
from progressive_model_delivery.models.pipeline import DemoRegistry
from progressive_model_delivery.models.synthetic import make_dataset
from progressive_model_delivery.registry.release import (
    ReleaseStore,
    ReleaseVerificationError,
)
from progressive_model_delivery.registry.signing import SigningKey, TrustStore

pytestmark = pytest.mark.integration


@pytest.fixture
def registry_copy(trained_registry: DemoRegistry, tmp_path: Path) -> ReleaseStore:
    """A private copy of the trained registry that a test may tamper with."""
    shutil.copytree(trained_registry.root / "registry", tmp_path / "registry")
    return ReleaseStore(tmp_path / "registry")


def _reason(store: ReleaseStore, release_id: str, trust: TrustStore) -> str:
    with pytest.raises(ReleaseVerificationError) as info:
        store.load_verified(release_id, trust)
    return info.value.reason


def test_every_demo_release_verifies_and_scores(trained_registry: DemoRegistry) -> None:
    store, trust = trained_registry.store, trained_registry.trust
    frame = make_dataset(50, seed=9).drop(columns=["label"])
    rows = [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]
    for name, release in trained_registry.models.releases.items():
        model = store.load_verified(release.release_id, trust)
        scores = model.scorer.predict_many(rows)
        assert scores.shape == (50,)
        assert np.all((scores >= 0) & (scores <= 1)), name
        assert model.manifest.categorical_levels["region"] == REGIONS
        assert "Model card" in store.model_card(release.release_id, trust)


def test_lineage_and_reproducibility_are_pinned(trained_registry: DemoRegistry) -> None:
    releases = trained_registry.models.releases
    champion = releases["champion"].manifest
    better = releases["better"].manifest
    assert champion.parent_release is None
    assert better.parent_release == releases["champion"].release_id
    assert (
        better.dataset_sha256 == champion.dataset_sha256 == trained_registry.models.dataset_sha256
    )
    # The A/A challenger is byte-identical to the champion but is its own release.
    identical = releases["identical"]
    assert identical.manifest.artifact_sha256 == champion.artifact_sha256
    assert identical.release_id != releases["champion"].release_id


def test_unsigned_release_is_rejected(
    registry_copy: ReleaseStore, trained_registry: DemoRegistry
) -> None:
    release_id = trained_registry.models.release_id("better")
    path = registry_copy.root / "releases" / f"{release_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["signature"] = "unsigned"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert _reason(registry_copy, release_id, trained_registry.trust) == "unsigned"


def test_release_signed_by_an_untrusted_key_is_rejected(
    trained_registry: DemoRegistry,
) -> None:
    stranger = TrustStore([SigningKey.generate().public()])
    release_id = trained_registry.models.release_id("better")
    assert _reason(trained_registry.store, release_id, stranger) == "untrusted_key"


def test_tampered_manifest_is_rejected(
    registry_copy: ReleaseStore, trained_registry: DemoRegistry
) -> None:
    release_id = trained_registry.models.release_id("worse")
    path = registry_copy.root / "releases" / f"{release_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["manifest"]["metrics"]["error_rate"] = 0.01  # make a bad model look good
    path.write_text(json.dumps(data), encoding="utf-8")
    assert _reason(registry_copy, release_id, trained_registry.trust) == "manifest_tampered"


def test_re_signed_manifest_under_the_new_hash_needs_a_trusted_key(
    registry_copy: ReleaseStore, trained_registry: DemoRegistry
) -> None:
    """An attacker who renames the file to the new hash still cannot produce the signature."""
    release_id = trained_registry.models.release_id("worse")
    signed = registry_copy.read(release_id)
    forged_manifest = signed.manifest.model_copy(update={"release": "better-v9"})
    forged = signed.model_copy(update={"manifest": forged_manifest})
    new_id = forged.release_id
    (registry_copy.root / "releases" / f"{new_id}.json").write_text(
        forged.model_dump_json(), encoding="utf-8"
    )
    assert _reason(registry_copy, new_id, trained_registry.trust) == "bad_signature"


def test_tampered_model_bytes_are_rejected_before_parsing(
    registry_copy: ReleaseStore, trained_registry: DemoRegistry
) -> None:
    release = trained_registry.models.releases["better"]
    blob = registry_copy.registry._blob_path(release.manifest.artifact_sha256)
    text = blob.read_text(encoding="utf-8")
    blob.write_text(text.replace("leaf_value=", "leaf_value=9", 1), encoding="utf-8")
    assert _reason(registry_copy, release.release_id, trained_registry.trust) == (
        "artifact_tampered"
    )


def test_publishing_refuses_anything_but_lightgbm_text(
    tmp_path: Path, trained_registry: DemoRegistry
) -> None:
    store = ReleaseStore(tmp_path / "registry")
    signed = trained_registry.models.releases["champion"]
    with pytest.raises(ReleaseVerificationError) as info:
        store.publish(
            model_name="churn-risk",
            release="pickled",
            model_text="\x80\x04\x95 not a model",
            model_card="card",
            run_manifest=trained_registry.store.registry.get(
                "churn-risk", signed.manifest.version_digest
            ).manifest,
            feature_order=signed.manifest.feature_order,
            categorical_levels=signed.manifest.categorical_levels,
            subgroup_feature="region",
            metrics={},
            references={},
            signing_key=trained_registry.key,
        )
    assert info.value.reason == "not_lightgbm_text"


def test_feature_order_mismatch_between_manifest_and_model_is_rejected(
    tmp_path: Path, trained_registry: DemoRegistry
) -> None:
    store = ReleaseStore(tmp_path / "registry")
    source = trained_registry.models.releases["champion"]
    version = trained_registry.store.registry.get("churn-risk", source.manifest.version_digest)
    text = trained_registry.store.registry.read_blob(source.manifest.artifact_sha256).decode()
    order = source.manifest.feature_order
    swapped = (order[1], order[0], *order[2:])
    release = store.publish(
        model_name="churn-risk",
        release="misordered",
        model_text=text,
        model_card="card",
        run_manifest=version.manifest,
        feature_order=swapped,
        categorical_levels=source.manifest.categorical_levels,
        subgroup_feature="region",
        metrics={},
        references={},
        signing_key=trained_registry.key,
    )
    assert _reason(store, release.release_id, trained_registry.trust) == "feature_mismatch"


def test_unknown_and_malformed_release_ids(registry_copy: ReleaseStore) -> None:
    trust = TrustStore()
    assert _reason(registry_copy, "0" * 64, trust) == "unknown_release"
    assert _reason(registry_copy, "../../etc/passwd", trust) == "bad_release_id"
    (registry_copy.root / "releases" / ("1" * 64 + ".json")).write_text("{", encoding="utf-8")
    assert _reason(registry_copy, "1" * 64, trust) == "malformed_release"
    assert len(registry_copy.release_ids()) >= 6
