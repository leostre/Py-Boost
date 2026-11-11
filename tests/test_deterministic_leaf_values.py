"""GPU checks for the opt-in deterministic leaf-value reduction."""

import os

import numpy as np
import pytest

from py_boost import GradientBoosting


pytestmark = pytest.mark.skipif(
    os.environ.get('HASBOOST_GPU_ACCEPTANCE') != '1',
    reason='orx GPU acceptance only',
)


def _expected_values(grad, hess, nodes, rows, groups, max_nodes, lr, lambda_l2):
    values = np.zeros((max_nodes, grad.shape[1]), dtype=np.float64)
    for node in range(max_nodes):
        for output, group in enumerate(groups):
            selected = rows[nodes[rows, group] == node]
            grad_sum = grad[selected, output].astype(np.float64).sum()
            hess_output = output if hess.shape[1] > 1 else 0
            hess_sum = hess[selected, hess_output].astype(np.float64).sum()
            values[node, output] = -grad_sum * lr / (hess_sum + lambda_l2)
    return values


def test_parameter_is_opt_in_and_preserved_in_model_params():
    default = GradientBoosting('multilabel')
    enabled = GradientBoosting('multilabel', deterministic_leaf_values=True)
    assert default.params['deterministic_leaf_values'] is False
    assert enabled.params['deterministic_leaf_values'] is True


def test_deterministic_leaf_values_are_exact_across_row_permutations():
    import cupy as cp

    from py_boost.gpu.utils import calc_node_values

    rng = np.random.default_rng(71)
    nrows, nout, max_nodes = 73, 4, 3
    grad = rng.normal(size=(nrows, nout)).astype(np.float32)
    hess = rng.uniform(0.2, 1.7, size=(nrows, nout)).astype(np.float32)
    nodes = rng.integers(0, max_nodes, size=(nrows, 2), dtype=np.int32)
    groups = np.asarray([0, 1, 0, 1], dtype=np.uint64)
    rows = np.arange(nrows, dtype=np.uint64)
    first_rows = rng.permutation(rows)
    second_rows = rng.permutation(rows)

    common = (
        cp.asarray(grad),
        cp.asarray(hess),
        cp.asarray(nodes),
    )
    first = calc_node_values(
        *common,
        cp.asarray(first_rows),
        cp.asarray(groups),
        max_nodes,
        lr=0.17,
        lambda_l2=0.4,
        deterministic=True,
    ).get()
    second = calc_node_values(
        *common,
        cp.asarray(second_rows),
        cp.asarray(groups),
        max_nodes,
        lr=0.17,
        lambda_l2=0.4,
        deterministic=True,
    ).get()

    assert np.array_equal(first, second)
    expected = _expected_values(
        grad, hess, nodes, rows, groups, max_nodes, 0.17, 0.4
    )
    np.testing.assert_allclose(first, expected, rtol=2e-6, atol=2e-6)


def test_deterministic_leaf_values_support_shared_hessian_column():
    import cupy as cp

    from py_boost.gpu.utils import calc_node_values

    grad = np.asarray(
        [[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0], [7.0, -8.0]],
        dtype=np.float32,
    )
    hess = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    nodes = np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.int32)
    rows = np.asarray([3, 0, 2], dtype=np.uint64)
    groups = np.asarray([0, 1], dtype=np.uint64)
    actual = calc_node_values(
        cp.asarray(grad),
        cp.asarray(hess),
        cp.asarray(nodes),
        cp.asarray(rows),
        cp.asarray(groups),
        2,
        lr=0.5,
        lambda_l2=0.25,
        deterministic=True,
    ).get()
    expected = _expected_values(
        grad, hess, nodes, rows, groups, 2, 0.5, 0.25
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
