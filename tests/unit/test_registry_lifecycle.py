from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from progressive_model_delivery.registry.lifecycle import (
    Action,
    Canary,
    CanaryWeight,
    IllegalTransitionError,
    LifecycleState,
    Production,
    Retired,
    RolledBack,
    Shadow,
    Stage,
    Staged,
    apply,
    from_record,
    to_record,
)

pytestmark = pytest.mark.unit

LEGAL_EDGES = {
    (Stage.STAGED, Action.START_SHADOW, Stage.SHADOW),
    (Stage.STAGED, Action.WITHDRAW, Stage.RETIRED),
    (Stage.SHADOW, Action.START_CANARY, Stage.CANARY),
    (Stage.SHADOW, Action.ROLL_BACK, Stage.ROLLED_BACK),
    (Stage.CANARY, Action.INCREASE, Stage.CANARY),
    (Stage.CANARY, Action.PROMOTE, Stage.PRODUCTION),
    (Stage.CANARY, Action.ROLL_BACK, Stage.ROLLED_BACK),
    (Stage.PRODUCTION, Action.RETIRE, Stage.RETIRED),
}
ORDER = [Stage.STAGED, Stage.SHADOW, Stage.CANARY, Stage.PRODUCTION]


def test_the_happy_path_is_typed_end_to_end() -> None:
    state = Staged().start_shadow().start_canary(CanaryWeight.W1)
    for weight in (CanaryWeight.W5, CanaryWeight.W25, CanaryWeight.W50):
        state = state.increase(weight)
    production = state.promote()
    assert isinstance(production, Production)
    assert isinstance(production.retire(), Retired)
    assert isinstance(Shadow().roll_back(), RolledBack)


@given(
    actions=st.lists(
        st.tuples(
            st.sampled_from(list(Action)), st.sampled_from([None, 100, 500, 2_500, 5_000, 7])
        ),
        max_size=25,
    )
)
def test_random_action_sequences_only_ever_follow_legal_edges(
    actions: list[tuple[Action, int | None]],
) -> None:
    """Property: illegal transitions are refused, never skipped, and weights only increase."""
    state: LifecycleState = Staged()
    for action, weight in actions:
        before = state
        try:
            state = apply(state, action, weight)
        except IllegalTransitionError:
            assert state == before
            continue
        assert (before.stage, action, state.stage) in LEGAL_EDGES
        if isinstance(before, Canary) and isinstance(state, Canary):
            assert state.weight > before.weight
        if before.stage in ORDER and state.stage in ORDER:
            assert ORDER.index(state.stage) - ORDER.index(before.stage) in (0, 1)
        assert from_record(*to_record(state)) == state


@pytest.mark.parametrize(
    ("state", "action"),
    [
        (Staged(), Action.PROMOTE),
        (Staged(), Action.START_CANARY),
        (Shadow(), Action.PROMOTE),
        (Production(), Action.ROLL_BACK),
        (RolledBack(), Action.START_SHADOW),
        (Retired(), Action.START_SHADOW),
    ],
)
def test_skipping_stages_or_leaving_terminal_states_is_illegal(
    state: LifecycleState, action: Action
) -> None:
    with pytest.raises(IllegalTransitionError):
        apply(state, action, CanaryWeight.W5)


def test_canary_weights_are_a_closed_set_and_only_increase() -> None:
    with pytest.raises(ValueError, match="CanaryWeight"):
        Canary(CanaryWeight(300))
    with pytest.raises(IllegalTransitionError, match="only increase"):
        Canary(CanaryWeight.W25).increase(CanaryWeight.W5)
    with pytest.raises(IllegalTransitionError, match="needs a weight"):
        apply(Shadow(), Action.START_CANARY)
    with pytest.raises(IllegalTransitionError, match="invalid canary weight"):
        apply(Shadow(), Action.START_CANARY, 42)
    assert CanaryWeight.W25.fraction == pytest.approx(0.25)


def test_records_round_trip_and_reject_impossible_combinations() -> None:
    assert to_record(Canary(CanaryWeight.W5)) == ("canary", 500)
    assert to_record(Shadow()) == ("shadow", 0)
    assert from_record("production", 0) == Production()
    with pytest.raises(IllegalTransitionError):
        from_record("canary", 1234)
    with pytest.raises(IllegalTransitionError):
        from_record("shadow", 500)
    with pytest.raises(ValueError, match="Stage"):
        from_record("prod", 0)
