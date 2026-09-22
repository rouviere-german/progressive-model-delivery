"""Scoring with a verified LightGBM booster, building feature vectors in the manifest's order."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

__all__ = ["ModelScorer", "ScoringError"]

FloatArray = npt.NDArray[np.float64]


class ScoringError(ValueError):
    """The request cannot be scored (missing feature, unknown category, non-finite value)."""


class ModelScorer:
    """Wraps a booster whose feature names were checked against its signed manifest."""

    def __init__(
        self,
        booster: lgb.Booster,
        *,
        feature_order: Sequence[str],
        categorical_levels: Mapping[str, Sequence[str]],
        threshold: float = 0.5,
    ) -> None:
        self._booster = booster
        self.feature_order = tuple(feature_order)
        self._levels = {name: tuple(levels) for name, levels in categorical_levels.items()}
        self.threshold = threshold

    def _value(self, name: str, raw: object) -> float:
        if name in self._levels:
            try:
                return float(self._levels[name].index(str(raw)))
            except ValueError:
                raise ScoringError(f"unknown category for {name}") from None
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ScoringError(f"feature {name} must be numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise ScoringError(f"feature {name} must be finite")
        return value

    def matrix(self, rows: Sequence[Mapping[str, object]]) -> FloatArray:
        out = np.empty((len(rows), len(self.feature_order)), dtype=np.float64)
        for i, row in enumerate(rows):
            for j, name in enumerate(self.feature_order):
                if name not in row:
                    raise ScoringError(f"missing feature {name}")
                out[i, j] = self._value(name, row[name])
        return out

    def predict_many(self, rows: Sequence[Mapping[str, object]]) -> FloatArray:
        if not rows:
            return np.empty(0, dtype=np.float64)
        result: FloatArray = np.asarray(
            self._booster.predict(self.matrix(rows), num_threads=1), dtype=np.float64
        )
        return result

    def predict_one(self, features: Mapping[str, object]) -> float:
        return float(self.predict_many([features])[0])
