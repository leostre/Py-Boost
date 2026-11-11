"""Pure tests for the H-HIST-DETERMINISM-1 primary gate."""

import pytest

from experiments.deterministic_histogram_core import (
    analyze_deterministic_histogram,
    analyze_histogram_perf2_microbenchmark,
    analyze_histogram_perf3_microbenchmark,
    analyze_histogram_runtime,
)


def _entry(iteration, depth, *, diagnostic='d', full='f', candidate='c'):
    return {
        'iteration': iteration,
        'group': 0,
        'depth': depth,
        'state_fingerprints': {
            'full_grad': 'g',
            'full_hess': 'h',
            'node_indexes': 'n',
            'full_rows': 'r',
            'col_indexer': 'cols',
            'diagnostic_loss': diagnostic,
            'full_loss': full,
            'candidate_identities': candidate,
        },
        'nodes': [{
            'node_id': 0,
            'refined_identity': [1, 2, False],
        }],
    }


def test_primary_gate_passes_only_for_complete_exact_histogram_replay():
    first = [_entry(1, 0), _entry(1, 1)]
    second = [_entry(1, 0), _entry(1, 1)]
    result = analyze_deterministic_histogram(first, second, expected_batches=2)
    assert result['primary_gate_passed'] is True
    assert result['all_captured_state_exact'] is True
    assert result['first_refined_identity_mismatch'] is None


def test_primary_gate_fails_on_diagnostic_histogram_drift():
    first = [_entry(1, 0)]
    second = [_entry(1, 0, diagnostic='changed')]
    result = analyze_deterministic_histogram(first, second, expected_batches=1)
    assert result['primary_gate_passed'] is False
    assert result['state_mismatch_counts']['diagnostic_loss'] == 1
    assert result['first_mismatch_by_field']['diagnostic_loss'] == {
        'iteration': 1, 'group': 0, 'depth': 0
    }


def test_primary_gate_rejects_incomplete_frontier():
    result = analyze_deterministic_histogram(
        [_entry(1, 0), _entry(1, 1)],
        [_entry(1, 0)],
        expected_batches=2,
    )
    assert result['primary_gate_passed'] is False
    assert result['trace_key_sets_equal'] is False


def test_runtime_gate_requires_twenty_percent_warm_speedup():
    result = analyze_histogram_runtime(
        optimized_seconds=[16.0, 12.0],
        frozen_histogram_seconds=[20.0, 16.0],
        leaf_baseline_seconds=[14.0, 10.0],
        minimum_warm_speedup_fraction=0.20,
    )
    assert result['first_fit_speedup_fraction'] == pytest.approx(0.20)
    assert result['warm_cache_speedup_fraction'] == pytest.approx(0.25)
    assert result['performance_gate_passed'] is True
    assert result['warm_cache_overhead_vs_leaf'] == pytest.approx(0.20)


def test_runtime_gate_rejects_a_smaller_warm_speedup():
    result = analyze_histogram_runtime(
        optimized_seconds=[16.0, 13.0],
        frozen_histogram_seconds=[20.0, 16.0],
        leaf_baseline_seconds=[14.0, 10.0],
        minimum_warm_speedup_fraction=0.20,
    )
    assert result['performance_gate_passed'] is False


def test_perf2_microbenchmark_accepts_exact_fast_sixteen_node_layout():
    result = analyze_histogram_perf2_microbenchmark(
        [
            {
                'nodes': 8,
                'bitwise_equal': True,
                'perf1_kernel_ms': [10.0, 10.2, 9.8],
                'perf2_kernel_ms': [10.4, 10.1, 10.3],
            },
            {
                'nodes': 16,
                'bitwise_equal': True,
                'perf1_kernel_ms': [20.0, 20.2, 19.8],
                'perf2_kernel_ms': [13.5, 13.7, 13.4],
            },
        ],
        minimum_16_node_speedup_fraction=0.30,
        maximum_8_node_regression_fraction=0.10,
    )

    assert result['correctness_gate_passed'] is True
    assert result['speedup_gate_passed'] is True
    assert result['regression_gate_passed'] is True
    assert result['performance_gate_passed'] is True
    assert result['scientific_status'] == 'confirmed'


def test_perf2_microbenchmark_rejects_speed_at_the_cost_of_correctness():
    result = analyze_histogram_perf2_microbenchmark(
        [
            {
                'nodes': 8,
                'bitwise_equal': True,
                'perf1_kernel_ms': [10.0],
                'perf2_kernel_ms': [10.0],
            },
            {
                'nodes': 16,
                'bitwise_equal': False,
                'perf1_kernel_ms': [20.0],
                'perf2_kernel_ms': [10.0],
            },
        ],
        minimum_16_node_speedup_fraction=0.30,
        maximum_8_node_regression_fraction=0.10,
    )

    assert result['performance_gate_passed'] is True
    assert result['correctness_gate_passed'] is False
    assert result['scientific_status'] == 'correctness_gate_failed'


def test_perf3_microbenchmark_reports_hybrid_timings():
    result = analyze_histogram_perf3_microbenchmark(
        [
            {
                'nodes': 8,
                'bitwise_equal': True,
                'perf1_kernel_ms': [10.0],
                'hybrid_kernel_ms': [10.2],
            },
            {
                'nodes': 16,
                'bitwise_equal': True,
                'perf1_kernel_ms': [20.0],
                'hybrid_kernel_ms': [13.0],
            },
        ],
        minimum_16_node_speedup_fraction=0.30,
        maximum_8_node_regression_fraction=0.05,
    )

    assert result['performance_gate_passed'] is True
    assert 'hybrid_kernel_ms' in result['cases'][0]
    assert 'perf2_kernel_ms' not in result['cases'][0]
