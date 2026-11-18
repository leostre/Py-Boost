"""Pure comparison logic for H-DETERMINISM-1 refinement traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_IDENTITY_FIELDS = (
    'refined_identity',
    'diagnostic_identity',
    'oracle_refined_identity',
    'full_best_identity',
)


def _key(entry: Mapping[str, object], node_id: object) -> tuple[int, int, int, int]:
    return (
        int(entry['iteration']),
        int(entry['group']),
        int(entry['depth']),
        int(node_id),
    )


def _identity(value: object) -> tuple[int, int, bool] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 3:
        return None
    return int(value[0]), int(value[1]), bool(value[2])


def _flatten_refinement(
    trace: Sequence[Mapping[str, object]],
) -> tuple[list[tuple[int, int, int, int]], dict[tuple[int, int, int, int], Mapping[str, object]]]:
    order = []
    indexed = {}
    for entry in trace:
        for node in entry['nodes']:
            node_key = _key(entry, node['node_id'])
            if node_key in indexed:
                raise ValueError(f'duplicate refinement node key: {node_key}')
            order.append(node_key)
            indexed[node_key] = node
    return order, indexed


def _index_diagnostics(
    records: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int, int, int], Mapping[str, object]]:
    indexed = {}
    for record in records:
        record_key = _key(record, record['node_id'])
        if record_key in indexed:
            raise ValueError(f'duplicate diagnostic node key: {record_key}')
        indexed[record_key] = record
    return indexed


def _selected_outputs(record: Mapping[str, object] | None) -> tuple[int, ...] | None:
    if record is None:
        return None
    value = record.get('selected_output_indices')
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return tuple(int(index) for index in value)


def _numeric_delta(
    first: Mapping[str, object],
    second: Mapping[str, object],
    field: str,
) -> float | None:
    left = first.get(field)
    right = second.get(field)
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def _classify_first_mismatch(
    first: Mapping[str, object],
    second: Mapping[str, object],
    first_diagnostic: Mapping[str, object] | None,
    second_diagnostic: Mapping[str, object] | None,
) -> str:
    first_outputs = _selected_outputs(first_diagnostic)
    second_outputs = _selected_outputs(second_diagnostic)
    if first_outputs is None or second_outputs is None:
        return 'insufficient_output_selection_evidence'
    if first_outputs != second_outputs:
        return 'output_selection_drift'
    if _identity(first.get('diagnostic_identity')) != _identity(
        second.get('diagnostic_identity')
    ):
        return 'diagnostic_stage_drift'
    if _identity(first.get('full_best_identity')) != _identity(
        second.get('full_best_identity')
    ):
        return 'full_state_drift_before_refinement_decision'
    if _identity(first.get('oracle_refined_identity')) == _identity(
        second.get('oracle_refined_identity')
    ):
        return 'segmented_gain_or_argmax_drift'
    return 'post_diagnostic_ambiguous_shortlist_or_state_drift'


def analyze_refinement_determinism(
    treatment_trace: Sequence[Mapping[str, object]],
    repeat_trace: Sequence[Mapping[str, object]],
    treatment_diagnostics: Sequence[Mapping[str, object]],
    repeat_diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Locate the first applied-winner divergence and state what the traces prove."""

    treatment_order, treatment_nodes = _flatten_refinement(treatment_trace)
    repeat_order, repeat_nodes = _flatten_refinement(repeat_trace)
    treatment_diag = _index_diagnostics(treatment_diagnostics)
    repeat_diag = _index_diagnostics(repeat_diagnostics)
    common_keys = [key for key in treatment_order if key in repeat_nodes]
    unmatched_treatment = [key for key in treatment_order if key not in repeat_nodes]
    unmatched_repeat = [key for key in repeat_order if key not in treatment_nodes]

    mismatch_counts = {field: 0 for field in _IDENTITY_FIELDS}
    selected_output_mismatches = 0
    missing_selected_output_pairs = 0
    maximum_deltas = {
        'refined_full_gain': 0.0,
        'oracle_refined_gain': 0.0,
        'full_best_gain': 0.0,
    }
    first_refined_mismatch_key = None
    for node_key in common_keys:
        first = treatment_nodes[node_key]
        second = repeat_nodes[node_key]
        for field in _IDENTITY_FIELDS:
            mismatch_counts[field] += int(
                _identity(first.get(field)) != _identity(second.get(field))
            )
        for field in maximum_deltas:
            delta = _numeric_delta(first, second, field)
            if delta is not None:
                maximum_deltas[field] = max(maximum_deltas[field], delta)
        first_outputs = _selected_outputs(treatment_diag.get(node_key))
        second_outputs = _selected_outputs(repeat_diag.get(node_key))
        if first_outputs is None or second_outputs is None:
            missing_selected_output_pairs += 1
        else:
            selected_output_mismatches += int(first_outputs != second_outputs)
        if (
            first_refined_mismatch_key is None
            and _identity(first.get('refined_identity'))
            != _identity(second.get('refined_identity'))
        ):
            first_refined_mismatch_key = node_key

    first_mismatch = None
    classification = 'no_refined_identity_divergence'
    if first_refined_mismatch_key is not None:
        first = treatment_nodes[first_refined_mismatch_key]
        second = repeat_nodes[first_refined_mismatch_key]
        first_diagnostic = treatment_diag.get(first_refined_mismatch_key)
        second_diagnostic = repeat_diag.get(first_refined_mismatch_key)
        classification = _classify_first_mismatch(
            first,
            second,
            first_diagnostic,
            second_diagnostic,
        )
        first_mismatch = {
            'key': {
                'iteration': first_refined_mismatch_key[0],
                'group': first_refined_mismatch_key[1],
                'depth': first_refined_mismatch_key[2],
                'node_id': first_refined_mismatch_key[3],
            },
            'classification': classification,
            'selected_outputs_equal': (
                _selected_outputs(first_diagnostic)
                == _selected_outputs(second_diagnostic)
                if _selected_outputs(first_diagnostic) is not None
                and _selected_outputs(second_diagnostic) is not None
                else None
            ),
            'treatment': {
                field: first.get(field) for field in _IDENTITY_FIELDS
            }
            | {
                'refined_full_gain': first.get('refined_full_gain'),
                'oracle_refined_gain': first.get('oracle_refined_gain'),
                'full_best_gain': first.get('full_best_gain'),
            },
            'repeat': {
                field: second.get(field) for field in _IDENTITY_FIELDS
            }
            | {
                'refined_full_gain': second.get('refined_full_gain'),
                'oracle_refined_gain': second.get('oracle_refined_gain'),
                'full_best_gain': second.get('full_best_gain'),
            },
            'absolute_gain_deltas': {
                field: _numeric_delta(first, second, field)
                for field in maximum_deltas
            },
        }

    evidence = {
        'selected_output_indices_saved': missing_selected_output_pairs == 0,
        'diagnostic_top1_saved': True,
        'refined_winner_saved': True,
        'oracle_refined_winner_saved': True,
        'full_search_winner_saved': True,
        'full_k64_candidate_identities_saved': False,
        'full_k64_refined_gains_saved': False,
        'top1_top2_margin_saved': False,
        'raw_gradients_hessians_saved': False,
        'float64_reconstruction_possible': False,
    }
    if classification == 'segmented_gain_or_argmax_drift':
        conclusion = (
            'The same selected outputs, diagnostic winner, full-search winner, and '
            'oracle shortlist winner were recorded; the applied winner changed. The '
            'remaining supported locus is segmented gain evaluation or final argmax.'
        )
    elif classification == 'post_diagnostic_ambiguous_shortlist_or_state_drift':
        conclusion = (
            'The selected outputs and diagnostic/full-search winners agree at the first '
            'applied split divergence, but the saved oracle shortlist winner also changes. '
            'Because full K64 identities, all gains, and raw G/H were not saved, shortlist '
            'membership/order drift cannot be separated from sub-ULP state drift or argmax.'
        )
    elif classification == 'no_refined_identity_divergence':
        conclusion = 'No applied refined-identity divergence was found in aligned nodes.'
    else:
        conclusion = (
            'The first applied split divergence is already visible upstream of the '
            'refinement winner; inspect the reported classification before changing argmax.'
        )

    return {
        'scientific_status': 'localized_with_identifiability_limit',
        'classification': classification,
        'aligned_nodes': len(common_keys),
        'treatment_nodes': len(treatment_order),
        'repeat_nodes': len(repeat_order),
        'unmatched_treatment_nodes': len(unmatched_treatment),
        'unmatched_repeat_nodes': len(unmatched_repeat),
        'identity_mismatch_counts': mismatch_counts,
        'selected_output_mismatches': selected_output_mismatches,
        'missing_selected_output_pairs': missing_selected_output_pairs,
        'maximum_absolute_gain_deltas': maximum_deltas,
        'first_refined_identity_mismatch': first_mismatch,
        'evidence_inventory': evidence,
        'conclusion': conclusion,
        'next_experiment': {
            'name': 'H-DETERMINISM-2 instrumented paired frontier replay',
            'required_capture': [
                'ordered K64 candidate identities for both repeats',
                'diagnostic, segmented-float32, and oracle gain for every candidate',
                'top1/top2 margin and deterministic identity fingerprints',
                'gradient and hessian fingerprints before shortlist construction',
            ],
            'stop_rule': (
                'Stop at the first applied winner divergence; do not run the full '
                '100-tree protocol until the unstable stage is identified.'
            ),
            'policy_rule': (
                'Choose float64 near-tie reranking only if candidate membership and '
                'pre-refinement state are stable; otherwise fix the upstream stage.'
            ),
        },
    }
