"""Contracts for exact, output-independent child row counts."""

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from py_boost.gpu.utils import _valid_child_counts


@st.composite
def _count_contracts(draw):
    total = draw(st.integers(min_value=0, max_value=10_000))
    left = draw(st.integers(min_value=0, max_value=total))
    minimum = draw(st.integers(min_value=0, max_value=10_000))
    return total, left, minimum


@given(_count_contracts())
@settings(max_examples=200, deadline=None)
def test_exact_child_count_contract_matches_inclusive_reference(contract):
    """A child with exactly ``min_data_in_leaf`` rows is valid, not one row short."""

    total, left, minimum = contract
    observed = _valid_child_counts(
        np.uint64(total), np.uint64(left), minimum
    )
    expected = left >= minimum and total - left >= minimum
    assert bool(observed) is expected


def test_exact_child_count_named_boundary_19_20_21():
    total = np.uint64(50)
    left = np.asarray([19, 20, 21], dtype=np.uint64)
    np.testing.assert_array_equal(
        _valid_child_counts(total, left, 20),
        np.asarray([False, True, True]),
    )


requires_gpu = pytest.mark.skipif(
    os.environ.get('HASBOOST_GPU_ACCEPTANCE') != '1',
    reason='GPU acceptance is an orx-managed experiment',
)


def _boundary_case(left_count, n_outputs, hessian_scale):
    import cupy as cp

    from py_boost.gpu.utils import pad_and_move

    n_rows = 50
    encoded = np.full((n_rows, 1), 2, dtype=np.uint8)
    encoded[:left_count, 0] = 1
    left_grad = np.linspace(0.5, 1.5, n_outputs, dtype=np.float32)
    right_grad = -np.linspace(0.25, 1.25, n_outputs, dtype=np.float32)
    grad = np.empty((n_rows, n_outputs), dtype=np.float32)
    grad[:left_count] = left_grad
    grad[left_count:] = right_grad
    hess = np.linspace(0.05, 0.25, n_rows, dtype=np.float32)[:, np.newaxis]
    hess *= np.float32(hessian_scale)
    return (
        pad_and_move(encoded),
        cp.asarray(grad),
        cp.asarray(hess),
        cp.zeros(n_rows, dtype=cp.int32),
        cp.arange(n_rows, dtype=cp.uint64),
    )


@requires_gpu
@given(
    left_count=st.sampled_from((19, 20, 21)),
    n_outputs=st.sampled_from((1, 3, 7)),
    hessian_scale=st.sampled_from((0.01, 1.0, 100.0)),
)
@settings(max_examples=12, deadline=None)
def test_full_and_candidate_paths_share_exact_boundary_contract(
    left_count, n_outputs, hessian_scale
):
    """Validity ignores Hessian scale and output count in both GPU scoring paths."""

    import cupy as cp

    from py_boost.gpu.utils import (
        _exact_candidate_gain_details,
        _loss_for_statistics,
    )

    arr, grad, hess, nodes, rows = _boundary_case(
        left_count, n_outputs, hessian_scale
    )
    candidates = {
        'features': cp.asarray([[0]], dtype=cp.int64),
        'split_bins': cp.asarray([[1]], dtype=cp.int64),
        'nan_left': cp.asarray([[True]], dtype=cp.bool_),
    }
    params = {'max_bin': 256, 'lambda_l2': 1.0, 'min_data_in_leaf': 20}
    full_loss, _, full_histogram = _loss_for_statistics(
        arr,
        grad,
        hess,
        nodes,
        rows,
        cp.asarray([0], dtype=cp.uint64),
        1,
        params,
        return_components=True,
        return_histogram=True,
    )
    statistics, details = _exact_candidate_gain_details(
        arr,
        grad,
        hess,
        nodes,
        rows,
        candidates,
        1,
        lambda_l2=1.0,
        min_data_in_leaf=20,
    )
    expected_valid = left_count >= 20

    assert int(statistics['prefix_counts'][0, 0].get()) == left_count
    assert int(details['left_count'][0, 0].get()) == left_count
    np.testing.assert_array_equal(
        full_histogram['counts'][0, 0, 1].get(), left_count
    )
    np.testing.assert_allclose(
        statistics['prefix'][:, 0, 0].get(),
        full_histogram['gh'][:, 0, 0, 1].get(),
        rtol=5e-5,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        statistics['totals'][:, 0].get(),
        full_histogram['gh'][:, 0, 0, -1].get(),
        rtol=5e-5,
        atol=5e-3,
    )
    np.testing.assert_array_equal(
        details['valid'][:, 0, 0].get(),
        np.full(n_outputs, expected_valid),
    )
    np.testing.assert_allclose(
        details['gains'][0, 0].get(),
        full_loss[0, 0, 1, 0].get(),
        rtol=5e-5,
        atol=5e-3,
    )

    reversed_statistics, reversed_details = _exact_candidate_gain_details(
        arr,
        grad,
        hess,
        nodes,
        rows[::-1],
        candidates,
        1,
        lambda_l2=1.0,
        min_data_in_leaf=20,
    )
    np.testing.assert_array_equal(
        reversed_statistics['prefix_counts'].get(),
        statistics['prefix_counts'].get(),
    )
    np.testing.assert_array_equal(
        reversed_details['valid'].get(), details['valid'].get()
    )
    np.testing.assert_allclose(
        reversed_details['gains'].get(), details['gains'].get(), rtol=5e-5, atol=5e-3
    )


@requires_gpu
def test_nan_directions_and_inactive_rows_use_exact_counts():
    """NaN routing changes integer left counts; inactive rows contribute nothing."""

    import cupy as cp

    from py_boost.gpu.utils import (
        _exact_candidate_gain_details,
        _loss_for_statistics,
        pad_and_move,
    )

    active_values = np.asarray([0] * 7 + [1] * 20 + [2] * 23, dtype=np.uint8)
    encoded = np.concatenate((active_values, np.zeros(5, dtype=np.uint8)))[:, None]
    node_indexes = np.concatenate((np.zeros(50, dtype=np.int32), -np.ones(5, dtype=np.int32)))
    grad = np.linspace(-1.0, 1.0, 55, dtype=np.float32)[:, None]
    hess = np.linspace(0.05, 0.25, 55, dtype=np.float32)[:, None]
    arr = pad_and_move(encoded)
    grad_gpu = cp.asarray(grad)
    hess_gpu = cp.asarray(hess)
    nodes = cp.asarray(node_indexes)
    rows = cp.arange(55, dtype=cp.uint64)
    candidates = {
        'features': cp.asarray([[0, 0]], dtype=cp.int64),
        'split_bins': cp.asarray([[1, 1]], dtype=cp.int64),
        'nan_left': cp.asarray([[True, False]], dtype=cp.bool_),
    }
    params = {'max_bin': 256, 'lambda_l2': 1.0, 'min_data_in_leaf': 20}
    full_loss, _, full_histogram = _loss_for_statistics(
        arr,
        grad_gpu,
        hess_gpu,
        nodes,
        rows,
        cp.asarray([0], dtype=cp.uint64),
        1,
        params,
        return_components=True,
        return_histogram=True,
    )
    statistics, details = _exact_candidate_gain_details(
        arr,
        grad_gpu,
        hess_gpu,
        nodes,
        rows,
        candidates,
        1,
        lambda_l2=1.0,
        min_data_in_leaf=20,
    )

    np.testing.assert_array_equal(statistics['counts'].get(), [50])
    np.testing.assert_array_equal(statistics['prefix_counts'].get(), [[27, 27]])
    np.testing.assert_array_equal(statistics['nan_counts'].get(), [[7, 7]])
    np.testing.assert_array_equal(details['left_count'].get(), [[27, 20]])
    np.testing.assert_allclose(
        statistics['prefix'][:, 0].get(),
        cp.stack((full_histogram['gh'][:, 0, 0, 1],) * 2, axis=1).get(),
        rtol=5e-5,
        atol=5e-3,
    )
    np.testing.assert_array_equal(details['valid'][:, 0, :].get(), [[True, True]])
    np.testing.assert_allclose(
        details['gains'][0].get(),
        cp.asarray([full_loss[0, 0, 1, 0], full_loss[0, 0, 1, 1]]).get(),
        rtol=5e-5,
        atol=5e-3,
    )


@requires_gpu
@given(order_seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=8, deadline=None)
def test_fused_candidate_counts_match_independent_numpy_reference(order_seed):
    """Mask reuse preserves exact counts under row reordering and inactive rows."""

    import cupy as cp

    from py_boost.gpu.utils import _selected_candidate_statistics, pad_and_move

    rng = np.random.default_rng(20260922)
    encoded = rng.integers(0, 5, size=(61, 3), dtype=np.uint8)
    node_indexes = np.arange(61, dtype=np.int32) % 2
    node_indexes[::7] = -1
    features = np.asarray([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
    split_bins = np.asarray([[1, 2, 3], [2, 1, 3]], dtype=np.int64)
    candidates = {
        'features': cp.asarray(features),
        'split_bins': cp.asarray(split_bins),
        'nan_left': cp.asarray([[True, False, True], [False, True, False]]),
    }
    order = np.random.default_rng(order_seed).permutation(61).astype(np.uint64)
    statistics = _selected_candidate_statistics(
        pad_and_move(encoded),
        cp.asarray(rng.normal(size=(61, 2)).astype(np.float32)),
        cp.asarray(rng.uniform(0.05, 0.25, size=(61, 1)).astype(np.float32)),
        cp.asarray(node_indexes),
        cp.asarray(order),
        candidates,
        2,
    )

    expected_prefix = np.zeros((2, 3), dtype=np.uint64)
    expected_nan = np.zeros_like(expected_prefix)
    expected_total = np.zeros(2, dtype=np.uint64)
    for node in range(2):
        rows = order[node_indexes[order] == node]
        values = encoded[rows[:, None], features[node][None, :]]
        expected_prefix[node] = (values <= split_bins[node]).sum(axis=0)
        expected_nan[node] = (values == 0).sum(axis=0)
        expected_total[node] = rows.size

    np.testing.assert_array_equal(statistics['prefix_counts'].get(), expected_prefix)
    np.testing.assert_array_equal(statistics['nan_counts'].get(), expected_nan)
    np.testing.assert_array_equal(statistics['counts'].get(), expected_total)
