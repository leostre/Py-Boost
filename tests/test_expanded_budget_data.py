import numpy as np
import pytest
from hypothesis import given, strategies as st
from experiments.expanded_budget_data import (indicator_with_complete_label_domain,
    validate_roles, control_selection_partition)


@given(c=st.integers(1, 50), origin=st.integers(0, 1))
def test_label_origin_is_unambiguous_and_roundtrips(c, origin):
    rows = [(float(i+origin),) for i in range(c)] + [(), tuple(range(origin, c+origin))]
    y, mapping = indicator_with_complete_label_domain(rows, c)
    np.testing.assert_array_equal(y[:c], np.eye(c))
    assert not y[c].any() and y[c+1].all()
    assert mapping == {str(i+origin): i for i in range(c)}


@pytest.mark.parametrize('rows,c', [([(1,), (2,)], 3), ([(0,), (2,)], 3),
    ([(.5,), (1,)], 2), ([(0, 0), (1,)], 2), ([(float('nan'),)], 1)])
def test_incomplete_ambiguous_or_duplicate_labels_rejected(rows, c):
    with pytest.raises(ValueError):
        indicator_with_complete_label_domain(rows, c)


def test_roles_are_conserved_disjoint_and_exhaustive():
    parts = dict(zip(('train', 'calibration', 'control', 'selection', 'test'),
                     [np.array([i], np.int64) for i in range(5)]))
    validate_roles(parts, 5)
    parts['test'] = np.array([0])
    with pytest.raises(ValueError):
        validate_roles(parts, 5)


@given(seed=st.integers(0, 10000), task=st.sampled_from(['multiclass', 'multilabel']))
def test_control_selection_uses_only_prior_selection_and_is_reproducible(seed, task):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, 60) if task == 'multiclass' else rng.integers(0, 2, (60, 3))
    selection = np.arange(30, 60)
    a, b = control_selection_partition(y, selection, task, 3, seed)
    again = control_selection_partition(y.copy(), selection.copy(), task, 3, seed)
    np.testing.assert_array_equal(a, again[0])
    np.testing.assert_array_equal(b, again[1])
    np.testing.assert_array_equal(np.sort(np.r_[a, b]), selection)
    assert not np.intersect1d(a, b).size
    changed = y.copy()
    changed[:30] = 0
    replay = control_selection_partition(changed, selection, task, 3, seed)
    np.testing.assert_array_equal(a, replay[0])
    np.testing.assert_array_equal(b, replay[1])
