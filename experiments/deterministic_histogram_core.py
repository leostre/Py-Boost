"""Pure paired-trace analysis for H-HIST-DETERMINISM-1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import math
import statistics


_STATE_FIELDS = (
    'full_grad',
    'full_hess',
    'node_indexes',
    'full_rows',
    'col_indexer',
    'diagnostic_loss',
    'full_loss',
    'candidate_identities',
)


def analyze_histogram_perf2_microbenchmark(
    cases: Sequence[Mapping[str, object]],
    minimum_16_node_speedup_fraction: float,
    maximum_8_node_regression_fraction: float,
) -> dict[str, object]:
    """Apply the preregistered H-HIST-PERF-2 kernel-level gates."""

    minimum_speedup = float(minimum_16_node_speedup_fraction)
    maximum_regression = float(maximum_8_node_regression_fraction)
    if not 0.0 <= minimum_speedup < 1.0:
        raise ValueError('minimum 16-node speedup must be in [0, 1)')
    if not 0.0 <= maximum_regression < 1.0:
        raise ValueError('maximum 8-node regression must be in [0, 1)')

    by_nodes = {}
    for raw in cases:
        nodes = int(raw['nodes'])
        if nodes in by_nodes:
            raise ValueError(f'duplicate microbenchmark case: {nodes} nodes')
        reference = [float(value) for value in raw['perf1_kernel_ms']]
        treatment = [float(value) for value in raw['perf2_kernel_ms']]
        if not reference or len(reference) != len(treatment):
            raise ValueError('paired kernel timings must be non-empty and aligned')
        if any(not math.isfinite(value) or value <= 0 for value in reference + treatment):
            raise ValueError('kernel timings must be finite and positive')
        reference_median = statistics.median(reference)
        treatment_median = statistics.median(treatment)
        by_nodes[nodes] = {
            'nodes': nodes,
            'repeats': len(reference),
            'bitwise_equal': bool(raw['bitwise_equal']),
            'perf1_kernel_ms': reference,
            'perf2_kernel_ms': treatment,
            'perf1_kernel_ms_median': reference_median,
            'perf2_kernel_ms_median': treatment_median,
            'speedup_fraction': 1.0 - treatment_median / reference_median,
            'regression_fraction': treatment_median / reference_median - 1.0,
        }

    if set(by_nodes) != {8, 16}:
        raise ValueError('microbenchmark requires exactly 8-node and 16-node cases')
    correctness_gate_passed = all(
        case['bitwise_equal'] for case in by_nodes.values()
    )
    speedup_gate_passed = (
        by_nodes[16]['speedup_fraction'] >= minimum_speedup
    )
    regression_gate_passed = (
        by_nodes[8]['regression_fraction'] <= maximum_regression
    )
    performance_gate_passed = speedup_gate_passed and regression_gate_passed
    return {
        'scientific_status': (
            'confirmed'
            if correctness_gate_passed and performance_gate_passed
            else 'correctness_gate_failed'
            if not correctness_gate_passed
            else 'performance_gate_failed'
        ),
        'cases': [by_nodes[nodes] for nodes in (8, 16)],
        'minimum_16_node_speedup_fraction': minimum_speedup,
        'maximum_8_node_regression_fraction': maximum_regression,
        'correctness_gate_passed': correctness_gate_passed,
        'speedup_gate_passed': speedup_gate_passed,
        'regression_gate_passed': regression_gate_passed,
        'performance_gate_passed': performance_gate_passed,
    }


def analyze_histogram_perf3_microbenchmark(
    cases: Sequence[Mapping[str, object]],
    minimum_16_node_speedup_fraction: float,
    maximum_8_node_regression_fraction: float,
) -> dict[str, object]:
    """Apply the H-HIST-PERF-3 gates and label the treatment as hybrid."""

    normalized = [
        {
            **raw,
            'perf2_kernel_ms': raw['hybrid_kernel_ms'],
        }
        for raw in cases
    ]
    result = analyze_histogram_perf2_microbenchmark(
        normalized,
        minimum_16_node_speedup_fraction,
        maximum_8_node_regression_fraction,
    )
    for case in result['cases']:
        case['hybrid_kernel_ms'] = case.pop('perf2_kernel_ms')
        case['hybrid_kernel_ms_median'] = case.pop(
            'perf2_kernel_ms_median'
        )
    return result


def analyze_histogram_runtime(
    optimized_seconds: Sequence[float],
    frozen_histogram_seconds: Sequence[float],
    leaf_baseline_seconds: Sequence[float],
    minimum_warm_speedup_fraction: float,
) -> dict[str, object]:
    """Evaluate the preregistered H-HIST-PERF-1 runtime gate."""

    series = {
        'optimized_train_seconds': optimized_seconds,
        'frozen_histogram_train_seconds': frozen_histogram_seconds,
        'leaf_baseline_train_seconds': leaf_baseline_seconds,
    }
    if any(len(values) != 2 for values in series.values()):
        raise ValueError('runtime comparisons require first and warm measurements')
    if any(float(value) <= 0 for values in series.values() for value in values):
        raise ValueError('runtime measurements must be positive')
    minimum_speedup = float(minimum_warm_speedup_fraction)
    if not 0.0 <= minimum_speedup < 1.0:
        raise ValueError('minimum warm speedup must be in [0, 1)')

    optimized = [float(value) for value in optimized_seconds]
    frozen = [float(value) for value in frozen_histogram_seconds]
    leaf = [float(value) for value in leaf_baseline_seconds]
    first_speedup = 1.0 - optimized[0] / frozen[0]
    warm_speedup = 1.0 - optimized[1] / frozen[1]
    return {
        'frozen_histogram_train_seconds': frozen,
        'leaf_baseline_train_seconds': leaf,
        'optimized_train_seconds': optimized,
        'first_fit_speedup_fraction': first_speedup,
        'warm_cache_speedup_fraction': warm_speedup,
        'first_fit_overhead_vs_leaf': optimized[0] / leaf[0] - 1.0,
        'warm_cache_overhead_vs_leaf': optimized[1] / leaf[1] - 1.0,
        'minimum_warm_speedup_fraction': minimum_speedup,
        'performance_gate_passed': warm_speedup >= minimum_speedup,
        'comparison_note': (
            'Frozen timings come from H-HIST-DETERMINISM-1 and '
            'H-LEAF-DETERMINISM-1 with the same data, horizon, seeds, '
            'instrumentation, and fixed run command.'
        ),
    }


def _batch_key(entry):
    return (
        int(entry['iteration']),
        int(entry['group']),
        int(entry['depth']),
    )


def _key_record(key):
    if key is None:
        return None
    return {'iteration': key[0], 'group': key[1], 'depth': key[2]}


def _states(trace):
    order = []
    states = {}
    entries = {}
    for entry in trace:
        key = _batch_key(entry)
        if key in states:
            raise ValueError(f'duplicate frontier batch key: {key}')
        state = entry.get('state_fingerprints', {})
        missing = [field for field in _STATE_FIELDS if field not in state]
        if missing:
            raise ValueError(f'missing state fingerprints at {key}: {missing}')
        order.append(key)
        states[key] = state
        entries[key] = entry
    return order, states, entries


def _identity(value):
    return tuple(value) if value is not None else None


def _first_refined_mismatch(common, treatment, repeat):
    for key in common:
        treatment_nodes = {
            int(node['node_id']): node for node in treatment[key]['nodes']
        }
        repeat_nodes = {
            int(node['node_id']): node for node in repeat[key]['nodes']
        }
        for node_id in treatment_nodes:
            if node_id not in repeat_nodes:
                return (*key, node_id)
            if _identity(treatment_nodes[node_id].get('refined_identity')) != _identity(
                repeat_nodes[node_id].get('refined_identity')
            ):
                return (*key, node_id)
    return None


def analyze_deterministic_histogram(
    treatment_trace: Sequence[Mapping[str, object]],
    repeat_trace: Sequence[Mapping[str, object]],
    expected_batches: int = 150,
) -> dict[str, object]:
    """Check exact histogram replay over the registered frontier."""

    treatment_order, treatment_states, treatment_entries = _states(
        treatment_trace
    )
    repeat_order, repeat_states, repeat_entries = _states(repeat_trace)
    trace_keys_equal = treatment_order == repeat_order
    common = [key for key in treatment_order if key in repeat_states]
    mismatch_counts = {
        field: sum(
            treatment_states[key][field] != repeat_states[key][field]
            for key in common
        )
        for field in _STATE_FIELDS
    }
    first_mismatch = {
        field: _key_record(next(
            (
                key for key in common
                if treatment_states[key][field] != repeat_states[key][field]
            ),
            None,
        ))
        for field in _STATE_FIELDS
    }
    primary_gate_passed = (
        trace_keys_equal
        and len(common) == expected_batches
        and mismatch_counts['diagnostic_loss'] == 0
        and mismatch_counts['full_loss'] == 0
    )
    refined_mismatch = _first_refined_mismatch(
        common,
        treatment_entries,
        repeat_entries,
    )
    return {
        'scientific_status': (
            'primary_gate_passed'
            if primary_gate_passed
            else 'primary_gate_failed'
        ),
        'primary_gate_passed': primary_gate_passed,
        'expected_frontier_batches': expected_batches,
        'aligned_frontier_batches': len(common),
        'trace_key_sets_equal': trace_keys_equal,
        'state_mismatch_counts': mismatch_counts,
        'first_mismatch_by_field': first_mismatch,
        'first_refined_identity_mismatch': (
            None if refined_mismatch is None else {
                'iteration': refined_mismatch[0],
                'group': refined_mismatch[1],
                'depth': refined_mismatch[2],
                'node_id': refined_mismatch[3],
            }
        ),
        'all_captured_state_exact': all(
            count == 0 for count in mismatch_counts.values()
        ),
        'interpretation': (
            'Both diagnostic and full histogram-derived loss arrays replay exactly.'
            if primary_gate_passed else
            'At least one registered histogram-derived loss array still drifts.'
        ),
    }
