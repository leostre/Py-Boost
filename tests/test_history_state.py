"""Tests for the article-defined HASBoost history and scheduler."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from hypothesis import given, strategies as st


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py_boost"
    / "multioutput"
    / "history_state.py"
)
SPEC = importlib.util.spec_from_file_location("history_state", MODULE_PATH)
history_state = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = history_state
SPEC.loader.exec_module(history_state)

decide_stable_regime = history_state.decide_stable_regime
decide_fixed_warmup = history_state.decide_fixed_warmup
advance_robust_scheduler = history_state.advance_robust_scheduler
RobustSchedulerConfig = history_state.RobustSchedulerConfig
RobustSchedulerState = history_state.RobustSchedulerState
AdaptiveBudgetConfig = history_state.AdaptiveBudgetConfig
AdaptiveBudgetState = history_state.AdaptiveBudgetState
advance_adaptive_budget = history_state.advance_adaptive_budget
resolve_budget_levels = history_state.resolve_budget_levels
relative_volatility = history_state.relative_volatility
robust_window_score = history_state.robust_window_score
update_scale_history = history_state.update_scale_history


def test_relative_volatility_uses_previous_scale_history():
    current_scale = np.array([11.0, 8.0])
    previous_scale = np.array([10.0, 10.0])
    previous_history = np.array([4.0, 6.0])

    volatility = relative_volatility(
        current_scale,
        previous_scale,
        previous_history,
        eps=1e-12,
    )

    assert np.isclose(volatility, 3.0 / 10.0)


def test_first_history_update_starts_from_zero_state():
    current_scale = np.array([2.0, 6.0])

    updated = update_scale_history(None, current_scale, alpha=0.75)

    np.testing.assert_allclose(updated, [0.5, 1.5])


def test_scheduler_applies_warmup_and_hysteresis():
    warmup = decide_stable_regime(1, 0.1, False, 0.3, 0.5, 2)
    enabled = decide_stable_regime(2, 0.2, False, 0.3, 0.5, 2)
    retained = decide_stable_regime(3, 0.4, True, 0.3, 0.5, 2)
    disabled = decide_stable_regime(4, 0.6, True, 0.3, 0.5, 2)

    assert warmup.stable_regime is False
    assert enabled.stable_regime is True
    assert retained.stable_regime is True
    assert disabled.stable_regime is False


def test_fixed_warmup_keeps_exactly_k_full_trees():
    decisions = [decide_fixed_warmup(iteration, 5) for iteration in range(1, 8)]

    assert [decision.stable_regime for decision in decisions] == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_robust_scheduler_requires_history_and_patience_before_activation():
    config = RobustSchedulerConfig(
        enable_threshold=0.2,
        disable_threshold=0.4,
        min_warmup_rounds=5,
        window_size=4,
        min_observations=3,
        patience_on=2,
    )
    state = RobustSchedulerState()

    decisions = []
    for iteration, observation in enumerate((0.1, 0.1, 0.1, 0.1, 0.1), start=1):
        state, decision = advance_robust_scheduler(
            iteration, observation, state, config
        )
        decisions.append(decision)

    assert all(not decision.stable_regime for decision in decisions[:4])
    assert decisions[4].stable_regime is False

    state, decision = advance_robust_scheduler(6, 0.1, state, config)
    assert decision.stable_regime is True
    assert decision.reason == "patience_satisfied"


def test_robust_scheduler_does_not_reuse_a_stale_observation_for_patience():
    config = RobustSchedulerConfig(
        enable_threshold=0.2,
        disable_threshold=0.4,
        min_warmup_rounds=1,
        window_size=2,
        min_observations=1,
        patience_on=2,
    )
    state, first = advance_robust_scheduler(
        1, 0.1, RobustSchedulerState(), config
    )
    state, missing = advance_robust_scheduler(2, None, state, config)

    assert first.stable_regime is False
    assert missing.stable_regime is False
    assert missing.reason == "waiting_for_observation"


def test_robust_score_is_resistant_to_one_large_outlier():
    score = robust_window_score((0.10, 0.11, 0.09, 100.0), robust_k=1.0)

    assert score < 0.2


def test_adaptive_budget_keeps_warmup_state_full_and_probes_next_level():
    config = AdaptiveBudgetConfig((25, 50, 75, 100), min_warmup_rounds=5)
    state = AdaptiveBudgetState(current_budget=75)

    next_state, decision = advance_adaptive_budget(5, None, state, config)

    assert next_state.current_budget == 100
    assert decision.output_budget == 100
    assert decision.probe_budget == 75
    assert decision.sketching_enabled is False


def test_adaptive_budget_moves_down_after_safe_patience_and_up_on_unsafe_risk():
    config = AdaptiveBudgetConfig(
        (25, 50, 75, 100),
        safe_threshold=0.10,
        unsafe_threshold=0.15,
        min_warmup_rounds=5,
        patience_down=2,
    )
    state = AdaptiveBudgetState(current_budget=75)

    state, waiting = advance_adaptive_budget(6, 0.08, state, config, probe_observation=0.08)
    state, reduced = advance_adaptive_budget(11, 0.09, state, config, probe_observation=0.09)
    state, restored = advance_adaptive_budget(16, 0.20, state, config)

    assert waiting.output_budget == 75
    assert reduced.output_budget == 50
    assert reduced.reason == "safe_patience_satisfied"
    assert restored.output_budget == 75
    assert restored.reason == "risk_above_unsafe_threshold"


def test_adaptive_budget_waits_after_an_increase_before_reducing_again():
    config = AdaptiveBudgetConfig(
        (25, 50, 75, 100),
        min_warmup_rounds=5,
        patience_down=2,
        min_dwell_rounds=5,
        cooldown_after_increase_rounds=10,
    )
    state = AdaptiveBudgetState(
        current_budget=75,
        consecutive_safe=1,
        observations=3,
        last_transition_iteration=10,
        cooldown_until_iteration=20,
    )

    state, blocked = advance_adaptive_budget(15, 0.05, state, config, probe_observation=0.05)
    state, reduced = advance_adaptive_budget(20, 0.05, state, config, probe_observation=0.05)

    assert blocked.output_budget == 75
    assert blocked.reason == "waiting_for_increase_cooldown"
    assert blocked.cooldown_remaining == 5
    assert reduced.output_budget == 50
    assert reduced.transitioned is True


def test_adaptive_budget_unsafe_risk_bypasses_minimum_dwell():
    config = AdaptiveBudgetConfig(
        (25, 50, 75, 100),
        min_warmup_rounds=5,
        min_dwell_rounds=10,
        cooldown_after_increase_rounds=10,
    )
    state = AdaptiveBudgetState(
        current_budget=50,
        last_transition_iteration=14,
    )

    state, decision = advance_adaptive_budget(15, 0.25, state, config)

    assert state.current_budget == 75
    assert state.last_transition_iteration == 15
    assert state.cooldown_until_iteration == 25
    assert decision.transitioned is True
    assert decision.reason == "risk_above_unsafe_threshold"


def test_first_reduction_requires_two_fresh_prospective_observations():
    config = AdaptiveBudgetConfig((25, 50, 75, 100), patience_down=2)
    state, warmup = advance_adaptive_budget(5, None, AdaptiveBudgetState(75), config)
    state, first = advance_adaptive_budget(6, None, state, config, probe_observation=0.08)
    state, stale = advance_adaptive_budget(7, None, state, config)
    state, second = advance_adaptive_budget(16, None, state, config, probe_observation=0.09)
    assert [warmup.output_budget, first.output_budget, stale.output_budget] == [100] * 3
    assert first.transitioned is False
    assert second.output_budget == 75
    assert second.transitioned is True


@given(risk=st.floats(min_value=0.0, max_value=0.10, allow_nan=False))
def test_safe_current_budget_does_not_authorize_an_unmeasured_reduction(risk):
    config = AdaptiveBudgetConfig((25, 50, 75, 100), patience_down=1, min_warmup_rounds=1)
    state, decision = advance_adaptive_budget(2, risk, AdaptiveBudgetState(75), config)
    assert state.current_budget == 75
    assert decision.probe_budget == 50


@given(
    levels=st.lists(
        st.integers(min_value=1, max_value=500),
        min_size=1,
        max_size=12,
    ),
    n_outputs=st.integers(min_value=1, max_value=500),
)
def test_resolved_budget_levels_are_unique_bounded_and_include_full(levels, n_outputs):
    resolved = resolve_budget_levels(tuple(levels), n_outputs)

    assert resolved == tuple(sorted(set(resolved)))
    assert all(1 <= budget <= n_outputs for budget in resolved)
    assert resolved[-1] == n_outputs


@given(
    current_index=st.integers(min_value=0, max_value=3),
    observation=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_adaptive_budget_changes_by_at_most_one_level(current_index, observation):
    budgets = (25, 50, 75, 100)
    config = AdaptiveBudgetConfig(budgets, min_warmup_rounds=1, patience_down=1)
    state = AdaptiveBudgetState(current_budget=budgets[current_index])

    next_state, decision = advance_adaptive_budget(2, observation, state, config)
    next_index = budgets.index(next_state.current_budget)

    assert abs(next_index - current_index) <= 1
    assert decision.output_budget in budgets
    if observation is None:
        assert next_state.current_budget == state.current_budget


@given(
    alpha=st.floats(min_value=0.0, max_value=0.999, allow_nan=False, allow_infinity=False),
    previous=st.lists(
        st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
    current=st.data(),
)
def test_ema_history_stays_between_previous_and_current(alpha, previous, current):
    current_values = current.draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=len(previous),
            max_size=len(previous),
        )
    )
    previous_array = np.asarray(previous)
    current_array = np.asarray(current_values)

    updated = update_scale_history(previous_array, current_array, alpha)

    lower = np.minimum(previous_array, current_array)
    upper = np.maximum(previous_array, current_array)
    assert np.allclose(np.maximum(updated, lower), updated)
    assert np.allclose(np.minimum(updated, upper), updated)
