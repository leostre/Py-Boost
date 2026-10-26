"""Pure tests for the H-LEAF-DETERMINISM-1 primary gate."""

from experiments.deterministic_leaf_reduction_core import (
    analyze_deterministic_leaf_reduction,
)


def _entry(iteration, depth, *, grad='g', hess='h', nodes='n', diagnostic='d'):
    return {
        'iteration': iteration,
        'group': 0,
        'depth': depth,
        'state_fingerprints': {
            'full_grad': grad,
            'full_hess': hess,
            'node_indexes': nodes,
            'diagnostic_loss': diagnostic,
            'full_loss': diagnostic,
        },
    }


def test_gate_allows_histogram_drift_when_optimization_state_is_exact():
    first = [_entry(1, 0), _entry(2, 0, diagnostic='d1'), _entry(2, 1)]
    second = [_entry(1, 0), _entry(2, 0, diagnostic='d2'), _entry(2, 1)]
    result = analyze_deterministic_leaf_reduction(first, second)
    assert result['primary_gate_passed'] is True
    assert result['first_mismatch_by_field']['diagnostic_loss'] == {
        'iteration': 2, 'group': 0, 'depth': 0
    }


def test_gate_fails_on_gradient_drift_before_partition_drift():
    first = [_entry(2, 0), _entry(2, 1), _entry(3, 0)]
    second = [
        _entry(2, 0),
        _entry(2, 1, grad='changed'),
        _entry(3, 0, nodes='changed'),
    ]
    result = analyze_deterministic_leaf_reduction(first, second)
    assert result['primary_gate_passed'] is False
    assert result['gate_mismatch_counts']['full_grad'] == 1
    assert result['first_row_partition_divergence'] == {
        'iteration': 3, 'group': 0, 'depth': 0
    }


def test_gate_excludes_state_after_first_partition_drift():
    first = [_entry(2, 0), _entry(2, 1), _entry(3, 0)]
    second = [
        _entry(2, 0),
        _entry(2, 1, nodes='changed'),
        _entry(3, 0, grad='changed', hess='changed', nodes='changed'),
    ]
    result = analyze_deterministic_leaf_reduction(first, second)
    assert result['primary_gate_passed'] is True
    assert result['registered_gate_batches'] == 1


def test_gate_rejects_incomplete_trace_alignment():
    first = [_entry(2, 0), _entry(2, 1)]
    second = [_entry(2, 0)]
    result = analyze_deterministic_leaf_reduction(first, second)
    assert result['primary_gate_passed'] is False
    assert result['trace_key_sets_equal'] is False
