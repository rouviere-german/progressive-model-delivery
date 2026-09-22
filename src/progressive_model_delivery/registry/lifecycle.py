"""The release lifecycle: staged -> shadow -> canary (1/5/25/50%) -> production -> retired.

Each stage is its own frozen class and exposes only the transitions that are legal from it, so in
type-checked code an illegal transition does not type-check (``Staged().promote()`` is a mypy
error, not a runtime surprise). Canary weights are an enumeration of basis points, so a weight
outside 1/5/25/50% cannot be constructed either.

Stored state comes back as ``(stage, weight_bp)`` records; :func:`from_record` rebuilds the typed
state and rejects impossible combinations, and :func:`apply` performs a named action on a
dynamically loaded state, raising :class:`IllegalTransitionError` for anything not in the graph:

    staged     --start_shadow-->    shadow
    staged     --withdraw-->        retired
    shadow     --start_canary(w)--> canary(w)
    shadow     --roll_back-->       rolled_back
    canary(w)  --increase(w')-->    canary(w')      only for w' > w
    canary(w)  --promote-->         production
    canary(w)  --roll_back-->       rolled_back
    production --retire-->          retired

``retired`` and ``rolled_back`` are terminal: a rolled-back model needs a new release to try again.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

__all__ = [
    "Action",
    "Canary",
    "CanaryWeight",
    "IllegalTransitionError",
    "LifecycleState",
    "Production",
    "Retired",
    "RolledBack",
    "Shadow",
    "Stage",
    "Staged",
    "apply",
    "from_record",
    "to_record",
]


class IllegalTransitionError(ValueError):
    """The requested action is not an edge of the lifecycle graph from the current stage."""


class Stage(StrEnum):
    STAGED = "staged"
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"
    RETIRED = "retired"
    ROLLED_BACK = "rolled_back"


class CanaryWeight(IntEnum):
    """Canary traffic share in basis points."""

    W1 = 100
    W5 = 500
    W25 = 2_500
    W50 = 5_000

    @property
    def fraction(self) -> float:
        return self.value / 10_000


class Action(StrEnum):
    START_SHADOW = "start_shadow"
    WITHDRAW = "withdraw"
    START_CANARY = "start_canary"
    INCREASE = "increase"
    PROMOTE = "promote"
    ROLL_BACK = "roll_back"
    RETIRE = "retire"


@dataclass(frozen=True, slots=True)
class Staged:
    stage = Stage.STAGED

    def start_shadow(self) -> Shadow:
        return Shadow()

    def withdraw(self) -> Retired:
        return Retired()


@dataclass(frozen=True, slots=True)
class Shadow:
    stage = Stage.SHADOW

    def start_canary(self, weight: CanaryWeight) -> Canary:
        return Canary(CanaryWeight(weight))

    def roll_back(self) -> RolledBack:
        return RolledBack()


@dataclass(frozen=True, slots=True)
class Canary:
    weight: CanaryWeight
    stage = Stage.CANARY

    def __post_init__(self) -> None:
        CanaryWeight(self.weight)  # raises ValueError for anything but 1/5/25/50%

    def increase(self, weight: CanaryWeight) -> Canary:
        target = CanaryWeight(weight)
        if target <= self.weight:
            raise IllegalTransitionError(
                f"canary weight can only increase ({self.weight.fraction:.0%} -> "
                f"{target.fraction:.0%} refused)"
            )
        return Canary(target)

    def promote(self) -> Production:
        return Production()

    def roll_back(self) -> RolledBack:
        return RolledBack()


@dataclass(frozen=True, slots=True)
class Production:
    stage = Stage.PRODUCTION

    def retire(self) -> Retired:
        return Retired()


@dataclass(frozen=True, slots=True)
class Retired:
    stage = Stage.RETIRED


@dataclass(frozen=True, slots=True)
class RolledBack:
    stage = Stage.ROLLED_BACK


LifecycleState = Staged | Shadow | Canary | Production | Retired | RolledBack


def to_record(state: LifecycleState) -> tuple[str, int]:
    """Persisted form: ``(stage, weight in basis points)``; the weight is 0 outside canary."""
    weight = int(state.weight) if isinstance(state, Canary) else 0
    return str(state.stage), weight


def from_record(stage: str, weight_bp: int) -> LifecycleState:
    parsed = Stage(stage)
    if parsed is Stage.CANARY:
        try:
            return Canary(CanaryWeight(weight_bp))
        except ValueError as exc:
            raise IllegalTransitionError(f"invalid canary weight {weight_bp} bp") from exc
    if weight_bp != 0:
        raise IllegalTransitionError(f"stage {parsed} cannot carry a traffic weight")
    simple: dict[Stage, LifecycleState] = {
        Stage.STAGED: Staged(),
        Stage.SHADOW: Shadow(),
        Stage.PRODUCTION: Production(),
        Stage.RETIRED: Retired(),
        Stage.ROLLED_BACK: RolledBack(),
    }
    return simple[parsed]


def apply(  # noqa: PLR0911 - one return per edge keeps the graph readable
    state: LifecycleState, action: Action | str, weight: CanaryWeight | int | None = None
) -> LifecycleState:
    """Apply a named action to a loaded state, or raise :class:`IllegalTransitionError`."""
    act = Action(action)
    match state, act:
        case Staged(), Action.START_SHADOW:
            return state.start_shadow()
        case Staged(), Action.WITHDRAW:
            return state.withdraw()
        case Shadow(), Action.START_CANARY:
            return state.start_canary(_weight(weight))
        case Shadow() | Canary(), Action.ROLL_BACK:
            return state.roll_back()
        case Canary(), Action.INCREASE:
            return state.increase(_weight(weight))
        case Canary(), Action.PROMOTE:
            return state.promote()
        case Production(), Action.RETIRE:
            return state.retire()
    raise IllegalTransitionError(f"{act} is not allowed from {state.stage}")


def _weight(weight: CanaryWeight | int | None) -> CanaryWeight:
    if weight is None:
        raise IllegalTransitionError("a canary transition needs a weight")
    try:
        return CanaryWeight(int(weight))
    except ValueError as exc:
        raise IllegalTransitionError(f"invalid canary weight {weight} bp") from exc
