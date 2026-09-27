"""The data-generating process of the rollout study: arm profiles and their dependence.

The study works at the level of the *guardrail events*, not of raw model scores, because the
quantity under test is the controller's error control, and that is a property of the procedure
under a stated null. Every scenario therefore fixes, per subgroup:

* ``error``: P(the served prediction is wrong),
* ``latency_exceed``: P(latency > SLO), derived from an explicit lognormal latency model,
* ``log_loss_mean`` / ``log_loss_sd``: the per-request log-loss distribution.

Dependence between the two arms matters in the shadow stage, where both models score the same
request. A single coupling parameter ``k`` mixes the two extreme joints with the correct
marginals:

* with probability ``k`` the pair is drawn from the *monotone* coupling (one latent uniform
  decides both arms, so the models fail on exactly the same hard requests),
* with probability ``1 - k`` the two arms are independent.

At ``k = 1`` two identical models produce no discordant pairs at all and the paired test has
nothing to work with; at ``k = 0`` the models disagree as often as chance allows. Real
champion/challenger pairs sit in between, so the study uses ``k = 0.85`` by default and the
sensitivity to ``k`` is reported by ``bench/bench_validity.py``.

Effect sizes for the pipeline-bug challengers are not invented: ``measured_scenario`` reads the
per-region error rate and log-loss that the actually trained LightGBM challengers reach on the
clean holdout, so ``bench/bench_power.py`` reports power against the real regressions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy import stats

from progressive_model_delivery.models.features import REGIONS

__all__ = [
    "BASE_LATENCY",
    "CHAMPION_LOG_LOSS_SD",
    "ArmProfile",
    "LatencyModel",
    "Scenario",
    "better",
    "error_regression",
    "identical",
    "inferior_beyond_margin",
    "latency_regression",
    "measured_scenario",
    "paired_cell_probabilities",
    "subgroup_regression",
]

FloatArray = npt.NDArray[np.float64]
CHAMPION_LOG_LOSS_SD = 0.55


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Lognormal serving latency, so a "+20 ms" regression maps to a real exceedance rate."""

    median_ms: float
    sigma: float

    def __post_init__(self) -> None:
        if self.median_ms <= 0 or self.sigma <= 0:
            raise ValueError("median_ms and sigma must be positive")

    def exceed_probability(self, slo_ms: float, *, shift_ms: float = 0.0) -> float:
        """P(shift_ms + X > slo_ms) for X lognormal with this median and shape."""
        if slo_ms <= 0:
            raise ValueError("slo_ms must be positive")
        remaining = slo_ms - shift_ms
        if remaining <= 0:
            return 1.0
        z = (math.log(remaining) - math.log(self.median_ms)) / self.sigma
        return float(stats.norm.sf(z))


BASE_LATENCY = LatencyModel(median_ms=8.0, sigma=0.5)


@dataclass(frozen=True, slots=True)
class ArmProfile:
    """Per-subgroup guardrail rates of one arm (champion or challenger)."""

    error: tuple[float, ...]
    latency_exceed: tuple[float, ...]
    log_loss_mean: float
    log_loss_sd: float = CHAMPION_LOG_LOSS_SD

    def __post_init__(self) -> None:
        for name, rates in (("error", self.error), ("latency_exceed", self.latency_exceed)):
            if len(rates) != len(REGIONS):
                raise ValueError(f"{name} needs one rate per subgroup ({len(REGIONS)})")
            if any(not 0.0 < rate < 1.0 for rate in rates):
                raise ValueError(f"{name} rates must be in (0, 1)")
        if self.log_loss_mean <= 0 or self.log_loss_sd <= 0:
            raise ValueError("log_loss_mean and log_loss_sd must be positive")

    @classmethod
    def uniform(
        cls,
        *,
        error: float,
        latency_exceed: float,
        log_loss_mean: float,
        log_loss_sd: float = CHAMPION_LOG_LOSS_SD,
    ) -> ArmProfile:
        groups = len(REGIONS)
        return cls((error,) * groups, (latency_exceed,) * groups, log_loss_mean, log_loss_sd)

    @property
    def error_array(self) -> FloatArray:
        return np.asarray(self.error, dtype=np.float64)

    @property
    def latency_array(self) -> FloatArray:
        return np.asarray(self.latency_exceed, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Scenario:
    """A champion/challenger pair plus the dependence between their per-request outcomes."""

    name: str
    champion: ArmProfile
    challenger: ArmProfile
    error_coupling: float = 0.85
    latency_coupling: float = 0.5
    log_loss_correlation: float = 0.9

    def __post_init__(self) -> None:
        for name, value in (
            ("error_coupling", self.error_coupling),
            ("latency_coupling", self.latency_coupling),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1.0 < self.log_loss_correlation < 1.0:
            raise ValueError("log_loss_correlation must be in (-1, 1)")

    @property
    def paired_log_loss_mean(self) -> float:
        """Mean of the paired loss difference (challenger minus champion)."""
        return self.challenger.log_loss_mean - self.champion.log_loss_mean

    @property
    def paired_log_loss_sd(self) -> float:
        """SD of the paired loss difference under the stated correlation."""
        a, b, rho = self.champion.log_loss_sd, self.challenger.log_loss_sd, self.log_loss_correlation
        return math.sqrt(max(a * a + b * b - 2.0 * rho * a * b, 1e-6))

    def worst_subgroup_gap(self) -> float:
        """Largest per-subgroup error increase of the challenger (0 when it is never worse)."""
        return float(np.max(self.challenger.error_array - self.champion.error_array))


def paired_cell_probabilities(
    p_a: npt.ArrayLike, p_b: npt.ArrayLike, coupling: float
) -> FloatArray:
    """Joint cell probabilities ``[p00, p01, p10, p11]`` of (champion event, challenger event).

    ``p01`` is "only the challenger failed". Marginals are exactly ``p_a`` and ``p_b`` for every
    ``coupling`` in [0, 1]; see the module docstring for the mixture.
    """
    if not 0.0 <= coupling <= 1.0:
        raise ValueError("coupling must be in [0, 1]")
    a = np.asarray(p_a, dtype=np.float64)
    b = np.asarray(p_b, dtype=np.float64)
    if np.any((a <= 0) | (a >= 1) | (b <= 0) | (b >= 1)):
        raise ValueError("event probabilities must be in (0, 1)")
    monotone = np.stack(
        [
            1.0 - np.maximum(a, b),
            np.maximum(b - a, 0.0),
            np.maximum(a - b, 0.0),
            np.minimum(a, b),
        ],
        axis=-1,
    )
    independent = np.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b], axis=-1)
    cells: FloatArray = coupling * monotone + (1.0 - coupling) * independent
    return cells


# --- scenario constructors -------------------------------------------------------------------

_BASE_ERROR = 0.20
_BASE_LOG_LOSS = 0.45


def _champion(slo_ms: float) -> ArmProfile:
    return ArmProfile.uniform(
        error=_BASE_ERROR,
        latency_exceed=BASE_LATENCY.exceed_probability(slo_ms),
        log_loss_mean=_BASE_LOG_LOSS,
    )


def identical(*, slo_ms: float = 25.0, coupling: float = 0.85) -> Scenario:
    """A/A: the challenger is the champion. Any decision against it is a false decision."""
    champion = _champion(slo_ms)
    return Scenario("identical", champion, champion, error_coupling=coupling)


def error_regression(delta_pp: float, *, slo_ms: float = 25.0) -> Scenario:
    """The challenger's error rate is ``delta_pp`` percentage points higher everywhere."""
    if delta_pp <= 0:
        raise ValueError("delta_pp must be positive")
    champion = _champion(slo_ms)
    delta = delta_pp / 100.0
    challenger = replace(
        champion,
        error=tuple(rate + delta for rate in champion.error),
        log_loss_mean=champion.log_loss_mean + 2.0 * delta,
    )
    return Scenario(f"error_+{delta_pp:g}pp", champion, challenger)


def subgroup_regression(
    region: str, delta_pp: float, *, slo_ms: float = 25.0, global_delta_pp: float = 0.0
) -> Scenario:
    """The challenger regresses in one subgroup only (and is unchanged, or better, elsewhere)."""
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}")
    champion = _champion(slo_ms)
    rates = []
    for name, rate in zip(REGIONS, champion.error, strict=True):
        shift = delta_pp if name == region else global_delta_pp
        rates.append(rate + shift / 100.0)
    challenger = replace(champion, error=tuple(rates))
    return Scenario(f"subgroup_{region}_+{delta_pp:g}pp", champion, challenger)


def latency_regression(shift_ms: float, *, slo_ms: float = 25.0) -> Scenario:
    """The challenger answers ``shift_ms`` later; error and log-loss are unchanged."""
    if shift_ms <= 0:
        raise ValueError("shift_ms must be positive")
    champion = _champion(slo_ms)
    challenger = replace(
        champion,
        latency_exceed=(BASE_LATENCY.exceed_probability(slo_ms, shift_ms=shift_ms),)
        * len(REGIONS),
    )
    return Scenario(f"latency_+{shift_ms:g}ms", champion, challenger, latency_coupling=0.5)


def better(*, delta_pp: float = 2.0, slo_ms: float = 25.0) -> Scenario:
    """A genuinely better challenger: lower error and lower log-loss everywhere."""
    if not 0 < delta_pp * 0.01 < _BASE_ERROR:
        raise ValueError("delta_pp must leave a positive error rate")
    champion = _champion(slo_ms)
    delta = delta_pp / 100.0
    challenger = replace(
        champion,
        error=tuple(rate - delta for rate in champion.error),
        log_loss_mean=champion.log_loss_mean - 2.0 * delta,
    )
    return Scenario(f"better_-{delta_pp:g}pp", champion, challenger)


def inferior_beyond_margin(ratio: float, *, slo_ms: float = 25.0) -> Scenario:
    """The challenger's error rate is ``ratio`` times the champion's (use ratio > the margin).

    This is the null of the *promotion* test: a challenger at or beyond the non-inferiority
    margin must be promoted with probability at most ``alpha_promotion``.
    """
    if ratio <= 1.0:
        raise ValueError("ratio must exceed 1")
    champion = _champion(slo_ms)
    challenger = replace(
        champion,
        error=tuple(min(rate * ratio, 0.95) for rate in champion.error),
        log_loss_mean=champion.log_loss_mean * ratio,
    )
    return Scenario(f"inferior_x{ratio:g}", champion, challenger)


def measured_scenario(
    name: str,
    champion_metrics: Mapping[str, float],
    challenger_metrics: Mapping[str, float],
    *,
    slo_ms: float = 25.0,
    regions: Sequence[str] = REGIONS,
) -> Scenario:
    """Build a scenario from measured holdout metrics of two trained models.

    ``*_metrics`` are :func:`~progressive_model_delivery.models.training.evaluate_scores`
    outputs, so they carry ``error_rate_<region>`` and ``log_loss``. Latency is left identical
    (LightGBM boosters of this size score in tens of microseconds either way).
    """

    def profile(metrics: Mapping[str, float]) -> ArmProfile:
        rates = tuple(
            min(max(float(metrics[f"error_rate_{region}"]), 1e-4), 0.95) for region in regions
        )
        return ArmProfile(
            error=rates,
            latency_exceed=(BASE_LATENCY.exceed_probability(slo_ms),) * len(regions),
            log_loss_mean=max(float(metrics["log_loss"]), 1e-4),
        )

    return Scenario(name, profile(champion_metrics), profile(challenger_metrics))
