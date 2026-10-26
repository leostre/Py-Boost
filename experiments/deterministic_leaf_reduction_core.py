"""Pure paired-trace analysis for H-LEAF-DETERMINISM-1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_STATE_FIELDS = (
    'full_grad',
    'full_hess',
    'node_indexes',
    'diagnostic_loss',
    'full_loss',
)


def _key(entry):
    return (
        int(entry['iteration']),
        int(entry['group']),
        int(entry['depth']),
    )


def _states(trace):
    result = {}
    order = []
    for entry in trace:
        key = _key(entry)
        if key in result:
            raise ValueError(f'duplicate frontier batch key: {key}')
        order.append(key)
        state = entry.get('state_fingerprints', {})
        missing = [field for field in _STATE_FIELDS if field not in state]
        if missing:
            raise ValueError(f'missing state fingerprints at {key}: {missing}')
        result[key] = state
    return order, result


def _key_record(key):
    if key is None:
        return None
    return {'iteration': key[0], 'group': key[1], 'depth': key[2]}


def analyze_deterministic_leaf_reduction(
    treatment_trace: Sequence[Mapping[str, object]],
    repeat_trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Test exact optimization-state replay until row partitions diverge."""

    treatment_order, treatment = _states(treatment_trace)
    repeat_order, repeat = _states(repeat_trace)
    trace_keys_equal = treatment_order == repeat_order
    common = [key for key in treatment_order if key in repeat]
    first_partition_mismatch = next(
        (
            key for key in common
            if treatment[key].get('node_indexes')
            != repeat[key].get('node_indexes')
        ),
        None,
    )
    gate_keys = [
        key for key in common
        if key[0] >= 2
        and (
            first_partition_mismatch is None
            or common.index(key) < common.index(first_partition_mismatch)
        )
    ]
    field_first_mismatch = {
        field: _key_record(next(
            (
                key for key in common
                if treatment[key].get(field) != repeat[key].get(field)
            ),
            None,
        ))
        for field in _STATE_FIELDS
    }
    gate_mismatches = {
        field: sum(
            treatment[key].get(field) != repeat[key].get(field)
            for key in gate_keys
        )
        for field in ('full_grad', 'full_hess')
    }
    gate_passed = (
        trace_keys_equal
        and bool(gate_keys)
        and all(count == 0 for count in gate_mismatches.values())
    )
    return {
        'scientific_status': (
            'primary_gate_passed' if gate_passed else 'primary_gate_failed'
        ),
        'primary_gate_passed': gate_passed,
        'aligned_frontier_batches': len(common),
        'registered_gate_batches': len(gate_keys),
        'first_row_partition_divergence': _key_record(
            first_partition_mismatch
        ),
        'gate_mismatch_counts': gate_mismatches,
        'first_mismatch_by_field': field_first_mismatch,
        'trace_key_sets_equal': trace_keys_equal,
        'interpretation': (
            'Exact gradient and hessian replay isolates leaf accumulation from '
            'the still-atomic histogram path.'
            if gate_passed else
            'Leaf accumulation alone did not restore exact optimization-state replay.'
        ),
    }
