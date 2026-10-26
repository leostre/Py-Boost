"""Pure H-DETERMINISM-2 comparison of fully captured split frontiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


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


def _key(entry, node_id):
    return (
        int(entry['iteration']),
        int(entry['group']),
        int(entry['depth']),
        int(node_id),
    )


def _flatten(trace):
    order = []
    nodes = {}
    states = {}
    for entry in trace:
        state = entry.get('state_fingerprints', {})
        for node in entry['nodes']:
            node_key = _key(entry, node['node_id'])
            if node_key in nodes:
                raise ValueError(f'duplicate refinement node key: {node_key}')
            order.append(node_key)
            nodes[node_key] = node
            states[node_key] = state
    return order, nodes, states


def _diagnostic_index(records):
    return {_key(record, record['node_id']): record for record in records}


def _identity(value):
    return tuple(value) if value is not None else None


def _selected_outputs(record):
    if record is None or record.get('selected_output_indices') is None:
        return None
    return tuple(int(value) for value in record['selected_output_indices'])


def _candidate_ids(node):
    return tuple(_identity(candidate['identity']) for candidate in node['candidate_frontier'])


def _candidate_map(node):
    return {
        _identity(candidate['identity']): candidate
        for candidate in node['candidate_frontier']
    }


def _gain_comparison(first, second):
    first_map = _candidate_map(first)
    second_map = _candidate_map(second)
    common = sorted(set(first_map) & set(second_map))
    fields = ('diagnostic_gain', 'refined_gain', 'oracle_gain')
    result = {}
    for field in fields:
        deltas = [
            abs(float(first_map[key][field]) - float(second_map[key][field]))
            for key in common
        ]
        result[field] = {
            'exact': bool(common) and all(delta == 0.0 for delta in deltas),
            'maximum_absolute_delta': max(deltas, default=None),
        }
    return result


def _classify(first, second, first_state, second_state, first_diag, second_diag):
    first_outputs = _selected_outputs(first_diag)
    second_outputs = _selected_outputs(second_diag)
    if first_outputs is None or second_outputs is None:
        return 'missing_output_selection_evidence'
    if first_outputs != second_outputs:
        return 'output_selection_drift'
    for field in ('full_grad', 'full_hess'):
        if first_state.get(field) != second_state.get(field):
            return 'optimization_state_drift'
    if first_state.get('node_indexes') != second_state.get('node_indexes'):
        return 'tree_state_drift'
    if first_state.get('diagnostic_loss') != second_state.get('diagnostic_loss'):
        return 'diagnostic_histogram_drift'
    if first_state.get('full_loss') != second_state.get('full_loss'):
        return 'full_histogram_drift'
    first_ids = _candidate_ids(first)
    second_ids = _candidate_ids(second)
    if set(first_ids) != set(second_ids):
        return 'shortlist_membership_drift'
    if first_ids != second_ids:
        return 'shortlist_order_drift'
    gains = _gain_comparison(first, second)
    if not gains['diagnostic_gain']['exact']:
        return 'diagnostic_gain_drift'
    if not gains['refined_gain']['exact']:
        return 'refined_gain_drift'
    if not gains['oracle_gain']['exact']:
        return 'oracle_gain_drift'
    return 'tie_breaking_drift'


def _top_candidates(record):
    if record is None:
        return None
    return {
        'diagnostic_top_k': record.get('diagnostic_top_k'),
        'full_top_k': record.get('full_top_k'),
    }


def analyze_determinism_frontier(
    treatment_trace: Sequence[Mapping[str, object]],
    repeat_trace: Sequence[Mapping[str, object]],
    treatment_diagnostics: Sequence[Mapping[str, object]],
    repeat_diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Classify the earliest applied-winner divergence from captured evidence."""

    treatment_order, treatment_nodes, treatment_states = _flatten(treatment_trace)
    repeat_order, repeat_nodes, repeat_states = _flatten(repeat_trace)
    treatment_diag = _diagnostic_index(treatment_diagnostics)
    repeat_diag = _diagnostic_index(repeat_diagnostics)
    common_keys = [key for key in treatment_order if key in repeat_nodes]
    first_key = next(
        (
            key for key in common_keys
            if _identity(treatment_nodes[key].get('refined_identity'))
            != _identity(repeat_nodes[key].get('refined_identity'))
        ),
        None,
    )
    if first_key is None:
        return {
            'scientific_status': 'no_divergence_within_registered_horizon',
            'classification': 'no_refined_identity_divergence',
            'aligned_nodes': len(common_keys),
            'first_divergence': None,
            'registered_horizon_trees': 30,
        }

    first = treatment_nodes[first_key]
    second = repeat_nodes[first_key]
    first_state = treatment_states[first_key]
    second_state = repeat_states[first_key]
    first_diag = treatment_diag.get(first_key)
    second_diag = repeat_diag.get(first_key)
    classification = _classify(
        first,
        second,
        first_state,
        second_state,
        first_diag,
        second_diag,
    )
    first_position = common_keys.index(first_key)
    prefix = common_keys[:first_position + 1]
    state_mismatch_counts = {
        field: sum(
            treatment_states[key].get(field) != repeat_states[key].get(field)
            for key in prefix
        )
        for field in _STATE_FIELDS
    }
    first_ids = _candidate_ids(first)
    second_ids = _candidate_ids(second)
    gains = _gain_comparison(first, second)
    result = {
        'scientific_status': 'first_divergence_classified',
        'classification': classification,
        'aligned_nodes': len(common_keys),
        'nodes_before_first_divergence': first_position,
        'state_mismatch_counts_through_first_divergence': state_mismatch_counts,
        'first_divergence': {
            'key': {
                'iteration': first_key[0],
                'group': first_key[1],
                'depth': first_key[2],
                'node_id': first_key[3],
            },
            'selected_outputs_equal': (
                _selected_outputs(first_diag) == _selected_outputs(second_diag)
                if _selected_outputs(first_diag) is not None
                and _selected_outputs(second_diag) is not None
                else None
            ),
            'state_fingerprint_equal': {
                field: first_state.get(field) == second_state.get(field)
                for field in _STATE_FIELDS
            },
            'treatment_refined_identity': first.get('refined_identity'),
            'repeat_refined_identity': second.get('refined_identity'),
            'ordered_shortlist_equal': first_ids == second_ids,
            'shortlist_membership_equal': set(first_ids) == set(second_ids),
            'shared_candidate_count': len(set(first_ids) & set(second_ids)),
            'gain_comparison_by_identity': gains,
            'treatment_margins': first.get('top1_top2_margins'),
            'repeat_margins': second.get('top1_top2_margins'),
            'treatment_search_top_candidates': _top_candidates(first_diag),
            'repeat_search_top_candidates': _top_candidates(second_diag),
        },
    }
    if classification == 'optimization_state_drift':
        result['next_action'] = (
            'Localize the first gradient or hessian fingerprint divergence before '
            'changing shortlist or tie-breaking logic.'
        )
    elif classification in {'diagnostic_histogram_drift', 'full_histogram_drift'}:
        result['next_action'] = (
            'Make the responsible histogram reduction deterministic, then repeat the '
            'same bounded paired protocol.'
        )
    elif classification in {'shortlist_membership_drift', 'shortlist_order_drift'}:
        result['next_action'] = (
            'Define one total candidate order for equal gains and apply it inside the '
            'feature-local and merged top-K stages.'
        )
    elif classification == 'tie_breaking_drift':
        result['next_action'] = (
            'Apply one explicit total order to diagnostic, full-search, and refined '
            'winner selection; no numerical tolerance is required by this evidence.'
        )
    else:
        result['next_action'] = (
            'Repair the earliest reported numeric stage before introducing a tie policy.'
        )
    return result
