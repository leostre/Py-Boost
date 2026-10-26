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
relative_volatility = history_state.relative_volatility
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

    assert np.all(updated >= np.minimum(previous_array, current_array))
    assert np.all(updated <= np.maximum(previous_array, current_array))
