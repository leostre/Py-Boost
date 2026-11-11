"""Energy-budget equivalence, scale invariance, and named corner cases."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from py_boost.multioutput.budget_policy import gradient_energy_budget


def test_budget_covers_the_prespecified_energy_mass():
    assert gradient_energy_budget(np.array([6., 3., 1.]), 0.90) == 2
    assert gradient_energy_budget(np.zeros(4), 0.90) == 4
    assert gradient_energy_budget(np.array([6., 3., 1.]), 0.50, min_budget=2) == 2


@given(values=st.lists(st.integers(min_value=1, max_value=10000), min_size=1, max_size=100),
       percentage=st.integers(min_value=1, max_value=99))
def test_energy_budget_matches_a_simple_prefix_reference(values, percentage):
    fraction = percentage / 100
    ordered = sorted(values, reverse=True)
    expected = next(i for i in range(1, len(ordered) + 1)
                    if sum(ordered[:i]) >= fraction * sum(ordered))
    actual = gradient_energy_budget(np.array(values, dtype=np.float64), fraction)
    assert actual == expected
    assert gradient_energy_budget(np.array(values, dtype=np.float64) * 2, fraction) == actual


def test_invalid_energy_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        gradient_energy_budget(np.array([np.nan]))
