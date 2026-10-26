import numpy as np
import pytest
from hypothesis import given, strategies as st
from experiments.leaf_geometry_core import tree_centered_leaf_basis
from experiments.shared_leaf_basis import functional_basis


@given(seed=st.integers(0, 10000), weighted=st.booleans(), task=st.sampled_from(['multilabel', 'multiclass']))
def test_leaf_geometry_is_invariant_to_arbitrary_per_tree_constant_gauge(seed, weighted, task):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(12, 5))
    offsets = np.array([0, 5, 12])
    f = np.r_[np.full(5, .2), np.full(7, 1/7)]
    shifts = rng.normal(size=(2, 5))
    changed = v.copy()
    changed[:5] += shifts[0]
    changed[5:] += shifts[1]
    mean, b, singular = tree_centered_leaf_basis(v, offsets, f, task, visitation_weighted=weighted)
    mean2, b2, singular2 = tree_centered_leaf_basis(changed, offsets, f, task, visitation_weighted=weighted)
    np.testing.assert_allclose(mean2-mean, shifts.sum(axis=0), atol=1e-12)
    np.testing.assert_allclose(singular2, singular, atol=1e-12)
    np.testing.assert_allclose(b[:, :1]@b[:, :1].T, b2[:, :1]@b2[:, :1].T, atol=1e-10)


def test_functional_geometry_detects_cross_tree_correlation_not_only_leaf_energy():
    v = np.array([[2., 0], [-2., 0], [0, 1.], [0, -1.]])
    f = np.full(4, .5)
    mean, leaf, _ = tree_centered_leaf_basis(v, np.array([0, 2, 4]), f, 'multilabel', visitation_weighted=True)
    z = np.array([[2., 1], [-2., -1]])
    functional = functional_basis(z, 'multilabel')[1]
    np.testing.assert_allclose(leaf[:, :1]@leaf[:, :1].T, np.array([[1., 0], [0, 0]]), atol=1e-12)
    expected = np.outer([2., 1.], [2., 1.])/5
    np.testing.assert_allclose(functional[:, :1]@functional[:, :1].T, expected, atol=1e-12)


def test_constant_tree_does_not_waste_a_basis_direction_and_invalid_frequency_is_rejected():
    v = np.array([[1., 0], [-1., 0], [0, 100.]])
    offsets = np.array([0, 2, 3])
    f = np.array([.5, .5, 1.])
    mean, b, _ = tree_centered_leaf_basis(v, offsets, f, 'multilabel')
    np.testing.assert_array_equal(mean, [0., 100.])
    np.testing.assert_allclose(b[:, :1]@b[:, :1].T, np.diag([1., 0.]), atol=1e-12)
    with pytest.raises(ValueError, match='sum to one'):
        tree_centered_leaf_basis(v, offsets, f*.9, 'multilabel')
