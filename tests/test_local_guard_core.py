import itertools
import pytest
from hypothesis import given, strategies as st

from experiments.local_guard_core import (ActiveGuard, GuardPolicy, Outcome,
    StopAndRollback, CommitCheckpoint, advance_guard, state_record)


def oracle(policy, events):
    committed, confirmed, failures = [], -1, 0
    for block, outcome in enumerate(events):
        stop = outcome == 'no_candidate'
        if outcome == 'local_fail':
            stop |= (policy == 'strict' or (policy == 'warmup2' and block >= 2) or
                     (policy == 'confirm2' and failures >= 1))
        if stop:
            return committed[:confirmed+1], confirmed, block
        committed.append(block)
        if outcome == 'local_pass':
            confirmed, failures = block, 0
        else:
            failures += 1
    return committed, confirmed, None


@given(policy=st.sampled_from(list(GuardPolicy)),
       events=st.lists(st.sampled_from(list(Outcome)), min_size=1, max_size=12))
def test_generated_sequences_match_independent_rollback_model(policy, events):
    state, committed = ActiveGuard(), []
    for block, outcome in enumerate(events):
        previous = state_record(state)
        decision = advance_guard(policy, state, block, outcome)
        assert state_record(state) == previous  # transition must not mutate its input
        if isinstance(decision, StopAndRollback):
            committed = [b for b in committed if b <= decision.state.retained_through]
            assert (committed, decision.state.retained_through, block) == oracle(policy.value, events)
            with pytest.raises(ValueError, match='stopped'):
                advance_guard(policy, decision.state, block+1, Outcome.PASS)
            break
        assert isinstance(decision, CommitCheckpoint)
        assert decision.locally_confirmed == (outcome is Outcome.PASS)
        committed.append(block)
        state = decision.state
        assert state.next_block == block+1
    else:
        assert (committed, state.last_confirmed_block, None) == oracle(policy.value, events)


@pytest.mark.parametrize('policy', list(GuardPolicy))
def test_all_successful_prefixes_are_identical_and_hard_failure_never_warms_up(policy):
    state = ActiveGuard()
    for block in range(5):
        result = advance_guard(policy, state, block, Outcome.PASS)
        assert result == CommitCheckpoint(ActiveGuard(block+1, 0), True)
        state = result.state
    result = advance_guard(policy, ActiveGuard(), 0, Outcome.NO_CANDIDATE)
    assert isinstance(result, StopAndRollback) and result.state.retained_through == -1


def test_warmup_window_is_exact_and_two_failures_rollback_before_both():
    state = ActiveGuard()
    for block in range(2):
        state = advance_guard('warmup2', state, block, Outcome.FAIL).state
    result = advance_guard('warmup2', state, 2, Outcome.FAIL)
    assert isinstance(result, StopAndRollback) and result.state.retained_through == -1
    state = advance_guard('confirm2', ActiveGuard(), 0, Outcome.PASS).state
    state = advance_guard('confirm2', state, 1, Outcome.FAIL).state
    result = advance_guard('confirm2', state, 2, Outcome.FAIL)
    assert result.state.retained_through == 0


def test_a_success_resets_confirmation_and_duplicate_events_are_invalid():
    state = ActiveGuard()
    for block, outcome in enumerate([Outcome.FAIL, Outcome.PASS, Outcome.FAIL, Outcome.PASS]):
        result = advance_guard('confirm2', state, block, outcome)
        assert isinstance(result, CommitCheckpoint)
        state = result.state
    assert state.last_confirmed_block == 3 and state.consecutive_failures == 0
    with pytest.raises(ValueError, match='out of order'):
        advance_guard('strict', state, 3, Outcome.PASS)
    with pytest.raises(ValueError):
        ActiveGuard(1, 2)


def test_exhaustive_short_traces_cover_each_window_and_confirmation_boundary():
    for policy, events in itertools.product(GuardPolicy, itertools.product(Outcome, repeat=5)):
        state, committed = ActiveGuard(), []
        for block, outcome in enumerate(events):
            result = advance_guard(policy, state, block, outcome)
            if isinstance(result, StopAndRollback):
                committed = [b for b in committed if b <= result.state.retained_through]
                assert (committed, result.state.retained_through, block) == oracle(policy.value, events)
                break
            committed.append(block)
            state = result.state
        else:
            assert (committed, state.last_confirmed_block, None) == oracle(policy.value, events)
