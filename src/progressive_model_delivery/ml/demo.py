"""Offline ML demo: contract check, manifest, registry round trip and drift monitoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import JsonValue

from progressive_model_delivery.ml.contracts import (
    DataContractError,
    make_synthetic_events,
    validate_events,
)
from progressive_model_delivery.ml.drift import DriftMonitor
from progressive_model_delivery.ml.manifest import create_manifest, seed_everything
from progressive_model_delivery.ml.registry import ModelRegistry
from progressive_model_delivery.ml.serialization import dumps_npz, loads_npz

__all__ = ["run_ml_demo"]


def run_ml_demo(workdir: Path) -> dict[str, JsonValue]:
    workdir.mkdir(parents=True, exist_ok=True)
    seed = 42
    seed_everything(seed)

    reference = validate_events(make_synthetic_events(2_000, seed=seed))
    dataset_path = workdir / "events.csv"
    reference.to_csv(dataset_path, index=False, lineterminator="\n")

    broken = reference.copy()
    broken.loc[0, "amount"] = -5.0
    broken.loc[1, "label"] = 3
    try:
        validate_events(broken)
        contract_failures: list[JsonValue] = []
    except DataContractError as exc:
        contract_failures = [dict(item) for item in exc.failures]

    # Toy "model": the midpoint between class-conditional mean amounts.
    amounts = reference["amount"].to_numpy(dtype=np.float64)
    labels = reference["label"].to_numpy(dtype=np.int64)
    threshold = float((amounts[labels == 0].mean() + amounts[labels == 1].mean()) / 2)
    accuracy = float(np.mean((amounts > threshold).astype(np.int64) == labels))

    manifest = create_manifest(
        seed=seed,
        datasets=[dataset_path],
        params={"model": "mean-midpoint-threshold"},
        metrics={"train_accuracy": accuracy},
        run_id="demo-run",
    )
    registry = ModelRegistry(workdir / "registry")
    version = registry.register(
        "amount-threshold",
        {"weights.npz": dumps_npz({"threshold": np.array([threshold])})},
        manifest,
        metrics={"train_accuracy": accuracy},
    )
    registry.set_alias("amount-threshold", "production", version.digest)
    restored = loads_npz(
        registry.load_files(registry.get("amount-threshold", "production"))["weights.npz"]
    )

    monitor = DriftMonitor({"amount": amounts})
    stable = monitor.check({"amount": make_synthetic_events(2_000, seed=7)["amount"].to_numpy()})
    shifted = monitor.check(
        {"amount": make_synthetic_events(2_000, seed=8)["amount"].to_numpy() * 1.5}
    )

    return {
        "rows": len(reference),
        "dataset_sha256": manifest.datasets[0].sha256,
        "contract_failures": contract_failures,
        "model_version": version.digest,
        "restored_threshold_matches": bool(np.allclose(restored["threshold"], [threshold])),
        "stable_psi": round(stable.features[0].psi, 4),
        "stable_drifted": stable.drifted,
        "shifted_psi": round(shifted.features[0].psi, 4),
        "shifted_drifted": shifted.drifted,
    }
