"""Population Stability Index and two-sample Kolmogorov-Smirnov drift monitors.

PSI uses quantile bins of the reference sample: ``sum((a - e) * ln(a / e))`` over bin fractions,
with fractions floored at ``eps``. Common reading: < 0.1 stable, 0.1-0.2 moderate, > 0.2 shift.

The KS statistic is the maximum distance between the two empirical CDFs. Its p-value uses the
asymptotic Kolmogorov distribution with Stephens' small-sample correction
``lambda = (sqrt(ne) + 0.12 + 0.11 / sqrt(ne)) * D`` (Stephens 1970; Press et al., Numerical
Recipes, section 14.3). With very large samples KS flags negligible shifts, so the monitor also
requires a minimum effect size before calling a feature drifted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "BinnedReference",
    "DriftMonitor",
    "DriftReport",
    "FeatureDrift",
    "KSResult",
    "kolmogorov_survival",
    "ks_2samp",
    "psi",
]

FloatArray = npt.NDArray[np.float64]


def _as_sample(values: npt.ArrayLike, label: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f"{label} sample is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} sample contains NaN or infinite values")
    return array


def psi(
    expected: npt.ArrayLike, actual: npt.ArrayLike, *, bins: int = 10, eps: float = 1e-4
) -> float:
    """Population Stability Index of ``actual`` against the ``expected`` reference."""
    if bins < 2:
        raise ValueError("bins must be >= 2")
    reference = _as_sample(expected, "expected")
    current = _as_sample(actual, "actual")
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        value = float(edges[0])
        inner = np.array([value, np.nextafter(value, np.inf)])
    else:
        inner = edges[1:-1]
    n_bins = inner.size + 1
    ref_frac = (
        np.bincount(np.searchsorted(inner, reference, side="right"), minlength=n_bins)
        / reference.size
    )
    cur_frac = (
        np.bincount(np.searchsorted(inner, current, side="right"), minlength=n_bins) / current.size
    )
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def kolmogorov_survival(lam: float) -> float:
    """Q_KS(lambda) = 2 * sum_{j>=1} (-1)^(j-1) exp(-2 j^2 lambda^2), clamped to [0, 1]."""
    if lam < 0.2:
        return 1.0
    total = 0.0
    for j in range(1, 101):
        term = 2.0 * (-1.0) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
    return min(1.0, max(0.0, total))


@dataclass(frozen=True, slots=True)
class BinnedReference:
    """A stored reference distribution: inner bin edges and the reference fraction per bin.

    Release manifests carry one per feature (and one for the model score), so PSI on live traffic
    needs no access to the training data. Numeric features use reference quantiles as edges;
    categorical codes ``0..k-1`` use edges at ``0.5, 1.5, ...``.
    """

    inner_edges: tuple[float, ...]
    fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.fractions) != len(self.inner_edges) + 1:
            raise ValueError("a reference needs exactly one more fraction than inner edges")
        if list(self.inner_edges) != sorted(self.inner_edges):
            raise ValueError("inner edges must be sorted")
        if not math.isclose(sum(self.fractions), 1.0, abs_tol=1e-6):
            raise ValueError("fractions must sum to 1")

    @classmethod
    def from_sample(cls, values: npt.ArrayLike, *, bins: int = 10) -> BinnedReference:
        if bins < 2:
            raise ValueError("bins must be >= 2")
        sample = _as_sample(values, "reference")
        edges = np.unique(np.quantile(sample, np.linspace(0.0, 1.0, bins + 1)))[1:-1]
        counts = np.bincount(np.searchsorted(edges, sample, side="right"), minlength=edges.size + 1)
        return cls(tuple(float(x) for x in edges), tuple(float(x) for x in counts / sample.size))

    @classmethod
    def from_codes(cls, codes: npt.ArrayLike, *, n_categories: int) -> BinnedReference:
        if n_categories < 2:
            raise ValueError("n_categories must be >= 2")
        sample = _as_sample(codes, "reference").astype(np.int64)
        if np.any((sample < 0) | (sample >= n_categories)):
            raise ValueError("category codes out of range")
        counts = np.bincount(sample, minlength=n_categories)
        return cls(
            tuple(k + 0.5 for k in range(n_categories - 1)),
            tuple(float(x) for x in counts / sample.size),
        )

    def fractions_of(self, values: npt.ArrayLike) -> FloatArray:
        sample = _as_sample(values, "current")
        counts = np.bincount(
            np.searchsorted(np.asarray(self.inner_edges), sample, side="right"),
            minlength=len(self.fractions),
        )
        result: FloatArray = counts / sample.size
        return result

    def psi(self, values: npt.ArrayLike, *, eps: float = 1e-4) -> float:
        """PSI of ``values`` against this reference (same formula as :func:`psi`)."""
        current = np.clip(self.fractions_of(values), eps, None)
        reference = np.clip(np.asarray(self.fractions, dtype=np.float64), eps, None)
        return float(np.sum((current - reference) * np.log(current / reference)))

    def to_json(self) -> dict[str, list[float]]:
        return {"inner_edges": list(self.inner_edges), "fractions": list(self.fractions)}

    @classmethod
    def from_json(cls, data: dict[str, list[float]]) -> BinnedReference:
        return cls(tuple(data["inner_edges"]), tuple(data["fractions"]))


@dataclass(frozen=True, slots=True)
class KSResult:
    statistic: float
    pvalue: float


def ks_2samp(sample_a: npt.ArrayLike, sample_b: npt.ArrayLike) -> KSResult:
    a = np.sort(_as_sample(sample_a, "first"))
    b = np.sort(_as_sample(sample_b, "second"))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    statistic = float(np.max(np.abs(cdf_a - cdf_b)))
    effective = math.sqrt(a.size * b.size / (a.size + b.size))
    lam = (effective + 0.12 + 0.11 / effective) * statistic
    return KSResult(statistic=statistic, pvalue=kolmogorov_survival(lam))


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


@dataclass(frozen=True, slots=True)
class DriftReport:
    features: tuple[FeatureDrift, ...]

    @property
    def drifted(self) -> bool:
        return any(item.drifted for item in self.features)

    @property
    def drifted_features(self) -> tuple[str, ...]:
        return tuple(item.feature for item in self.features if item.drifted)


class DriftMonitor:
    """Compare current feature samples against stored reference samples."""

    def __init__(
        self,
        reference: Mapping[str, npt.ArrayLike],
        *,
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.01,
        ks_min_statistic: float = 0.05,
        bins: int = 10,
    ) -> None:
        if not reference:
            raise ValueError("reference must contain at least one feature")
        self._reference = {name: _as_sample(values, name) for name, values in reference.items()}
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.ks_min_statistic = ks_min_statistic
        self.bins = bins

    def check(self, current: Mapping[str, npt.ArrayLike]) -> DriftReport:
        missing = sorted(set(self._reference) - set(current))
        if missing:
            raise ValueError(f"current data is missing features: {', '.join(missing)}")
        results = []
        for name in sorted(self._reference):
            reference = self._reference[name]
            score = psi(reference, current[name], bins=self.bins)
            ks = ks_2samp(reference, current[name])
            drifted = score > self.psi_threshold or (
                ks.pvalue < self.ks_alpha and ks.statistic >= self.ks_min_statistic
            )
            results.append(FeatureDrift(name, score, ks.statistic, ks.pvalue, drifted))
        return DriftReport(tuple(results))
