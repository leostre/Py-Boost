"""GPU regression and property checks for node-local candidate statistics."""

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

pytestmark = pytest.mark.skipif(
    os.environ.get('HASBOOST_GPU_ACCEPTANCE') != '1',
    reason='GPU acceptance is an orx-managed experiment',
)


def _arrays(node_indexes, n_features=8, n_outputs=6):
    import cupy as cp
    from py_boost.gpu.utils import pad_and_move

    node_indexes = np.asarray(node_indexes, dtype=np.int32)
    rows = np.arange(node_indexes.size, dtype=np.float32)[:, np.newaxis]
    outputs = np.arange(n_outputs, dtype=np.float32)[np.newaxis, :]
    encoded = (
        np.arange(node_indexes.size * n_features, dtype=np.uint16)
        .reshape(node_indexes.size, n_features)
        % 32
    ).astype(np.uint8)
    encoded[::7, ::2] = 0
    grad = np.sin(rows * 0.17 + outputs * 0.31).astype(np.float32)
    hess = (0.05 + 0.2 / (1.0 + np.exp(-(rows * 0.03 - outputs * 0.07)))).astype(
        np.float32
    )
    return (
        pad_and_move(encoded),
        cp.asarray(grad),
        cp.asarray(hess),
        cp.asarray(node_indexes),
    )


def _loss(arr, grad, hess, nodes, rows, nnodes, min_data=5):
    import cupy as cp
    from py_boost.gpu.utils import _loss_for_statistics

    return _loss_for_statistics(
        arr,
        grad,
        hess,
        nodes,
        rows,
        cp.arange(arr.shape[1], dtype=cp.uint64),
        nnodes,
        {'max_bin': 256, 'lambda_l2': 1.0, 'min_data_in_leaf': min_data},
    )


def test_full_statistics_ignore_rows_outside_active_nodes_regression():
    """Inactive rows are a sentinel class, not a negative histogram address."""

    import cupy as cp

    node_indexes = np.arange(256, dtype=np.int32) % 4
    node_indexes[node_indexes == 3] = -1
    arr, grad, hess, nodes = _arrays(node_indexes)
    all_rows = cp.arange(len(node_indexes), dtype=cp.uint64)
    active_rows = all_rows[nodes[all_rows] >= 0]
    observed = _loss(arr, grad, hess, nodes, all_rows, 3)
    reference = _loss(arr, grad, hess, nodes, active_rows, 3)
    np.testing.assert_allclose(observed.get(), reference.get(), rtol=5e-5, atol=5e-3)


@st.composite
def _active_node_layouts(draw):
    """Build layouts containing every active node and at least one inactive row."""

    nnodes = draw(st.integers(min_value=1, max_value=4))
    extras = draw(
        st.lists(
            st.integers(min_value=-1, max_value=nnodes - 1),
            min_size=4,
            max_size=40,
        )
    )
    return nnodes, tuple(range(nnodes)) + (-1,) + tuple(extras)


@given(_active_node_layouts())
@settings(max_examples=12, deadline=None)
def test_inactive_row_filter_is_equivalent_to_explicit_active_subset(layout):
    """Adding sentinel rows cannot change statistics of any active node."""

    import cupy as cp

    nnodes, node_indexes = layout
    arr, grad, hess, nodes = _arrays(node_indexes, n_features=4, n_outputs=2)
    all_rows = cp.arange(len(node_indexes), dtype=cp.uint64)
    active_rows = all_rows[nodes[all_rows] >= 0]
    observed = _loss(arr, grad, hess, nodes, all_rows, nnodes, min_data=0)
    reference = _loss(arr, grad, hess, nodes, active_rows, nnodes, min_data=0)
    np.testing.assert_allclose(observed.get(), reference.get(), rtol=5e-5, atol=5e-3)


def test_candidate_intermediates_and_gains_match_full_histogram_with_inactive_rows():
    """Candidate-only accumulation agrees with the full histogram at every stage."""

    import cupy as cp
    from py_boost.gpu.utils import (
        _candidate_gain_details,
        _loss_for_statistics,
        _selected_candidate_statistics,
        count_histogram,
        histogram,
    )

    node_indexes = np.arange(320, dtype=np.int32) % 5
    node_indexes[node_indexes == 4] = -1
    nnodes = 4
    arr, grad, hess, nodes = _arrays(node_indexes)
    all_rows = cp.arange(len(node_indexes), dtype=cp.uint64)
    active_rows = all_rows[nodes[all_rows] >= 0]
    cols = cp.arange(arr.shape[1], dtype=cp.uint64)
    candidates = {
        'features': cp.asarray(
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
            dtype=cp.int64,
        ),
        'split_bins': cp.asarray(
            [[0, 7, 15, 23], [3, 11, 19, 27], [5, 13, 21, 29], [1, 9, 17, 25]],
            dtype=cp.int64,
        ),
        'nan_left': cp.asarray(
            [[True, False, True, False]] * nnodes,
            dtype=cp.bool_,
        ),
    }
    stats = _selected_candidate_statistics(
        arr, grad, hess, nodes, all_rows, candidates, nnodes
    )
    gh = cp.concatenate((grad, hess), axis=1)
    full_hist = histogram(
        arr,
        gh,
        nodes,
        cols,
        active_rows,
        cp.arange(gh.shape[1], dtype=cp.uint64),
        nnodes,
        256,
    )
    full_counts = count_histogram(
        arr, nodes, cols, active_rows, nnodes, 256
    )
    node_positions = cp.arange(nnodes)[:, cp.newaxis]
    feature_positions = candidates['features']
    oracle_prefix = full_hist[
        :, node_positions, feature_positions, candidates['split_bins']
    ]
    oracle_nan = full_hist[:, node_positions, feature_positions, 0]
    oracle_totals = full_hist[:, :, 0, -1]
    oracle_prefix_counts = full_counts[
        node_positions, feature_positions, candidates['split_bins']
    ]
    oracle_nan_counts = full_counts[node_positions, feature_positions, 0]
    expected_counts = np.bincount(
        np.asarray(node_indexes)[np.asarray(node_indexes) >= 0], minlength=nnodes
    )

    np.testing.assert_allclose(stats['prefix'].get(), oracle_prefix.get(), rtol=5e-5, atol=5e-3)
    np.testing.assert_allclose(stats['nan'].get(), oracle_nan.get(), rtol=5e-5, atol=5e-3)
    np.testing.assert_allclose(stats['totals'].get(), oracle_totals.get(), rtol=5e-5, atol=5e-3)
    np.testing.assert_array_equal(stats['counts'].get(), expected_counts)
    np.testing.assert_array_equal(stats['prefix_counts'].get(), oracle_prefix_counts.get())
    np.testing.assert_array_equal(stats['nan_counts'].get(), oracle_nan_counts.get())

    details = _candidate_gain_details(
        stats, candidates, grad.shape[1], lambda_l2=1.0, min_data_in_leaf=5
    )
    full_loss, full_components = _loss_for_statistics(
        arr,
        grad,
        hess,
        nodes,
        all_rows,
        cols,
        nnodes,
        {'max_bin': 256, 'lambda_l2': 1.0, 'min_data_in_leaf': 5},
        return_components=True,
    )
    directions = cp.where(candidates['nan_left'], 0, 1)
    oracle_components = full_components[
        :, node_positions, feature_positions, candidates['split_bins'], directions
    ]
    oracle_gains = oracle_components.sum(axis=0)
    np.testing.assert_allclose(
        details['component_gains'].get(), oracle_components.get(), rtol=5e-5, atol=5e-3
    )
    np.testing.assert_array_equal(details['valid'].get(), (oracle_components != 0).get())
    np.testing.assert_allclose(details['gains'].get(), oracle_gains.get(), rtol=5e-5, atol=5e-3)
    np.testing.assert_allclose(full_loss[
        node_positions, feature_positions, candidates['split_bins'], directions
    ].get(), oracle_gains.get(), rtol=5e-5, atol=5e-3)
