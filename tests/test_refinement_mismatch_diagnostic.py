"""Checks for stage-wise localization of refinement gain mismatches."""

from __future__ import annotations

import os

import numpy as np
import pytest

from experiments.refinement_mismatch_diagnostic import (
    summarize_failure_diagnostics,
    summarize_refinement_correctness,
)
from py_boost.gpu.utils import _comparison_summary


def test_comparison_summary_distinguishes_numeric_and_logical_mismatches():
    numeric = _comparison_summary(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 2.02]),
        rtol=0.0,
        atol=0.01,
    )
    logical = _comparison_summary(
        np.asarray([True, False]),
        np.asarray([True, True]),
    )

    assert numeric['allclose'] is False
    assert numeric['mismatch_count'] == 1
    assert logical['allclose'] is False
    assert logical['mismatch_count'] == 1


def test_summary_requires_the_registered_critical_state():
    record = {
        'node_position': 8,
        'node_id': 23,
        'candidate_index': 0,
        'identity': [103, 20, True],
        'is_refined_winner': False,
        'is_oracle_winner': True,
        'first_divergent_stage': 'prefix',
        'closer_to_float64_gain': 'histogram',
        'candidate_gain': -82.0,
        'histogram_gain': 9.0,
        'original_oracle_gain': 9.0,
        'float64_gain': 9.0,
        'comparisons': {},
        'values': {},
    }
    trace = [{
        'iteration': 1,
        'group': 0,
        'depth': 4,
        'diagnostic_output_budget': 50,
        'failure_diagnostics': {
            'severe_error_threshold': 1.0,
            'severe_node_count': 1,
            'records': [record],
        },
    }]

    summary = summarize_failure_diagnostics(trace)

    assert summary['diagnostic_status'] == 'critical_mismatch_localized'
    assert summary['target_records'] == 1
    assert summary['first_divergent_stage_counts'] == {'prefix': 1}
    assert summary['closer_to_float64_gain_counts'] == {'histogram': 1}


def test_correctness_summary_requires_every_refinement_batch_to_agree():
    trace = [
        {
            'candidate_gains_allclose': True,
            'all_refined_winners_gain_equivalent': True,
            'maximum_candidate_gain_absolute_error': 0.002,
            'maximum_candidate_gain_relative_error': 0.001,
        },
        {
            'candidate_gains_allclose': True,
            'all_refined_winners_gain_equivalent': True,
            'maximum_candidate_gain_absolute_error': 0.004,
            'maximum_candidate_gain_relative_error': 0.003,
        },
    ]

    summary = summarize_refinement_correctness(trace)

    assert summary['batches'] == 2
    assert summary['candidate_gains_allclose'] is True
    assert summary['all_refined_winners_gain_equivalent'] is True
    assert summary['maximum_candidate_gain_absolute_error'] == pytest.approx(0.004)
    assert summary['maximum_candidate_gain_relative_error'] == pytest.approx(0.003)


@pytest.mark.skipif(
    os.environ.get('HASBOOST_GPU_ACCEPTANCE') != '1',
    reason='GPU acceptance is an orx-managed experiment',
)
def test_float64_reference_matches_candidate_statistics_on_gpu():
    import cupy as cp

    from py_boost.gpu.utils import (
        _candidate_gain_details,
        _cpu_float64_candidate_reference,
        _selected_candidate_statistics,
        pad_and_move,
    )

    rng = np.random.default_rng(20260921)
    encoded = rng.integers(0, 32, size=(160, 4), dtype=np.uint8)
    encoded[::11, 2] = 0
    node_indexes = np.arange(160, dtype=np.int32) % 3
    node_indexes[node_indexes == 2] = -1
    grad = rng.normal(size=(160, 5)).astype(np.float32)
    hess = rng.uniform(0.05, 0.25, size=(160, 1)).astype(np.float32)
    arr = pad_and_move(encoded)
    grad_gpu = cp.asarray(grad)
    hess_gpu = cp.asarray(hess)
    nodes_gpu = cp.asarray(node_indexes)
    rows = cp.arange(160, dtype=cp.uint64)
    candidates = {
        'features': cp.asarray([[2], [1]], dtype=cp.int64),
        'split_bins': cp.asarray([[10], [15]], dtype=cp.int64),
        'nan_left': cp.asarray([[True], [False]], dtype=cp.bool_),
    }
    statistics = _selected_candidate_statistics(
        arr, grad_gpu, hess_gpu, nodes_gpu, rows, candidates, 2
    )
    details = _candidate_gain_details(
        statistics,
        candidates,
        n_outputs=5,
        lambda_l2=1.0,
        min_data_in_leaf=5,
    )
    reference = _cpu_float64_candidate_reference(
        arr,
        grad_gpu,
        hess_gpu,
        nodes_gpu,
        rows,
        node_position=0,
        feature=2,
        split_bin=10,
        nan_left=True,
        lambda_l2=1.0,
        min_data_in_leaf=5,
    )

    np.testing.assert_allclose(
        statistics['prefix'][:, 0, 0].get(), reference['prefix'], rtol=5e-5, atol=5e-3
    )
    np.testing.assert_allclose(
        statistics['nan'][:, 0, 0].get(), reference['nan'], rtol=5e-5, atol=5e-3
    )
    np.testing.assert_allclose(
        statistics['totals'][:, 0].get(), reference['totals'], rtol=5e-5, atol=5e-3
    )
    assert int(statistics['prefix_counts'][0, 0].get()) == int(reference['prefix_counts'])
    assert int(statistics['nan_counts'][0, 0].get()) == int(reference['nan_counts'])
    assert int(details['left_count'][0, 0].get()) == int(reference['left_count'])
    np.testing.assert_array_equal(details['valid'][:, 0, 0].get(), reference['valid'])
    np.testing.assert_allclose(
        details['component_gains'][:, 0, 0].get(),
        reference['component_gains'],
        rtol=5e-5,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        details['gains'][0, 0].get(), reference['gain'], rtol=5e-5, atol=5e-3
    )
