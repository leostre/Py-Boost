"""Deterministic diagnosis of K=64 float32 accumulation-order differences."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.refinement_integration_benchmark import _gpu_name, _make_inputs


FLOAT32_UNIT_ROUNDOFF = 2.0 ** -24


def summation_error_bound(values, term_count=None, extra_terms=0):
    """Return Higham's gamma bound for a float32 sum against a float64 sum."""

    values = np.asarray(values, dtype=np.float64)
    n_terms = int(values.size if term_count is None else term_count) + int(extra_terms)
    if n_terms <= 1:
        return 0.0
    product = (n_terms - 1) * FLOAT32_UNIT_ROUNDOFF
    if product >= 1.0:
        return float('inf')
    gamma = product / (1.0 - product)
    return float(gamma * np.abs(values).sum(dtype=np.float64))


def bounded_error_summary(observed, reference, bound):
    """Summarize errors relative to an independently computed numeric bound."""

    observed = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    bound = np.asarray(bound, dtype=np.float64)
    absolute = np.abs(observed - reference)
    allowed = bound + np.finfo(np.float64).tiny
    normalized = np.divide(
        absolute,
        allowed,
        out=np.zeros_like(absolute),
        where=allowed > 0,
    )
    return {
        'shape': list(observed.shape),
        'within_bound': bool(np.all(absolute <= allowed)),
        'exactly_equal': bool(np.array_equal(observed, reference)),
        'maximum_absolute_error': float(absolute.max(initial=0.0)),
        'maximum_bound': float(bound.max(initial=0.0)),
        'maximum_bound_fraction': float(normalized.max(initial=0.0)),
    }


def _columnwise_summation_bounds(values, extra_terms):
    """Vectorized gamma bounds for independent column sums."""

    values = np.asarray(values, dtype=np.float64)
    n_terms = int(values.shape[0]) + int(extra_terms)
    if n_terms <= 1:
        return np.zeros(values.shape[1], dtype=np.float64)
    product = (n_terms - 1) * FLOAT32_UNIT_ROUNDOFF
    if product >= 1.0:
        return np.full(values.shape[1], np.inf, dtype=np.float64)
    gamma = product / (1.0 - product)
    return gamma * np.abs(values).sum(axis=0, dtype=np.float64)


def _exact_summary(observed, reference):
    observed = np.asarray(observed)
    reference = np.asarray(reference)
    mismatch = observed != reference
    return {
        'shape': list(observed.shape),
        'exact': bool(not mismatch.any()),
        'mismatch_count': int(mismatch.sum()),
    }


def _gain_summary(observed, reference, rtol, atol):
    observed = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    absolute = np.abs(observed - reference)
    relative = absolute / np.maximum(np.abs(reference), 1e-12)
    return {
        'allclose': bool(np.allclose(observed, reference, rtol=rtol, atol=atol)),
        'maximum_absolute_error': float(absolute.max(initial=0.0)),
        'maximum_relative_error': float(relative.max(initial=0.0)),
    }


def _winner_equivalence(observed, reference, rtol, atol):
    observed = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    observed_indices = observed.argmax(axis=1)
    reference_indices = reference.argmax(axis=1)
    rows = np.arange(observed.shape[0])
    observed_winners = observed[rows, observed_indices]
    reference_winners = reference[rows, reference_indices]
    return {
        'gain_equivalent': bool(np.allclose(
            observed_winners, reference_winners, rtol=rtol, atol=atol
        )),
        'identity_equal': bool(np.array_equal(observed_indices, reference_indices)),
        'observed_indices': observed_indices.tolist(),
        'reference_indices': reference_indices.tolist(),
        'maximum_winner_gain_error': float(
            np.abs(observed_winners - reference_winners).max(initial=0.0)
        ),
    }


def _cpu_float64_reference(
    encoded,
    grad,
    hess,
    node_indexes,
    features,
    split_bins,
    nan_left,
    min_data_in_leaf,
    lambda_l2,
    roundoff_extra_terms,
):
    """Compute candidate statistics independently with deterministic float64 sums."""

    nnodes, shortlist_size = features.shape
    n_outputs = grad.shape[1]
    statistics = np.concatenate((grad, hess), axis=1).astype(np.float64, copy=False)
    stat_count = statistics.shape[1]
    prefix = np.zeros((stat_count, nnodes, shortlist_size), dtype=np.float64)
    nan_statistics = np.zeros_like(prefix)
    totals = np.zeros((stat_count, nnodes), dtype=np.float64)
    prefix_bounds = np.zeros_like(prefix)
    nan_bounds = np.zeros_like(prefix)
    total_bounds = np.zeros_like(totals)
    counts = np.zeros(nnodes, dtype=np.uint64)
    prefix_counts = np.zeros((nnodes, shortlist_size), dtype=np.uint64)
    nan_counts = np.zeros_like(prefix_counts)

    for node in range(nnodes):
        row_mask = node_indexes == node
        node_statistics = statistics[row_mask]
        node_encoded = encoded[row_mask]
        counts[node] = node_statistics.shape[0]
        totals[:, node] = node_statistics.sum(axis=0, dtype=np.float64)
        total_bounds[:, node] = _columnwise_summation_bounds(
            node_statistics, roundoff_extra_terms
        )
        for candidate in range(shortlist_size):
            values = node_encoded[:, features[node, candidate]]
            prefix_mask = values <= split_bins[node, candidate]
            missing_mask = values == 0
            prefix_counts[node, candidate] = int(prefix_mask.sum())
            nan_counts[node, candidate] = int(missing_mask.sum())
            prefix[:, node, candidate] = node_statistics[prefix_mask].sum(
                axis=0, dtype=np.float64
            )
            nan_statistics[:, node, candidate] = node_statistics[missing_mask].sum(
                axis=0, dtype=np.float64
            )
            prefix_bounds[:, node, candidate] = _columnwise_summation_bounds(
                node_statistics[prefix_mask], roundoff_extra_terms
            )
            nan_bounds[:, node, candidate] = _columnwise_summation_bounds(
                node_statistics[missing_mask], roundoff_extra_terms
            )

    grad_prefix = prefix[:n_outputs]
    hess_prefix = prefix[n_outputs:]
    nan_grad = nan_statistics[:n_outputs]
    nan_hess = nan_statistics[n_outputs:]
    direction = nan_left[np.newaxis, :, :]
    left_grad = np.where(direction, grad_prefix, grad_prefix - nan_grad)
    left_hess = np.where(direction, hess_prefix, hess_prefix - nan_hess)
    left_count = np.where(direction[0], prefix_counts, prefix_counts - nan_counts)
    count_valid = (
        (left_count >= min_data_in_leaf)
        & ((counts[:, np.newaxis] - left_count) >= min_data_in_leaf)
    )
    count_valid &= nan_left | (
        (nan_counts > 0) & (prefix_counts < counts[:, np.newaxis])
    )
    valid = np.broadcast_to(count_valid[np.newaxis, :, :], left_grad.shape)
    total_grad = totals[:n_outputs, :, np.newaxis]
    total_hess = totals[n_outputs:, :, np.newaxis]
    right_grad = total_grad - left_grad
    previous = total_grad * total_grad / (total_hess + lambda_l2)
    components = (
        left_grad * left_grad / (left_hess + lambda_l2)
        + right_grad * right_grad / (total_hess - left_hess + lambda_l2)
        - previous
    )
    components = np.where(valid, components, 0.0)
    return {
        'prefix': prefix,
        'nan': nan_statistics,
        'totals': totals,
        'counts': counts,
        'prefix_counts': prefix_counts,
        'nan_counts': nan_counts,
        'valid': valid,
        'component_gains': components,
        'gains': components.sum(axis=0, dtype=np.float64),
        'bounds': {
            'prefix': prefix_bounds,
            'nan': nan_bounds,
            'totals': total_bounds,
        },
    }


def _forced_segmented_statistics(
    arr, grad, hess, node_indexes, rows, candidates, nnodes
):
    import cupy as cp
    from py_boost.gpu.utils import (
        _active_rows,
        _counts_for_rows,
        _selected_candidate_kernel_statistics,
    )

    active_rows = _active_rows(node_indexes, rows)
    counts = _counts_for_rows(node_indexes, active_rows, nnodes)
    row_nodes = node_indexes[active_rows]
    order = cp.argsort(row_nodes)
    sorted_nodes = row_nodes[order].astype(cp.int64, copy=False)
    sorted_rows = active_rows[order]
    offsets = cp.cumsum(counts, dtype=cp.uint64) - counts
    prefix, nan_statistics, totals, prefix_counts, nan_counts = (
        _selected_candidate_kernel_statistics(
            arr,
            grad,
            hess,
            sorted_rows,
            offsets,
            counts,
            candidates['features'],
            candidates['split_bins'],
            nnodes,
        )
    )
    return {
        'prefix': prefix,
        'nan': nan_statistics,
        'totals': totals,
        'counts': counts,
        'prefix_counts': prefix_counts,
        'nan_counts': nan_counts,
    }


def _candidate_level_full_histogram(full_histogram, candidates):
    gh = full_histogram['gh'].get()
    counts = full_histogram['counts'].get()
    features = candidates['features'].get()
    split_bins = candidates['split_bins'].get()
    node_positions = np.arange(features.shape[0])[:, np.newaxis]
    return {
        'prefix': gh[:, node_positions, features, split_bins],
        'nan': gh[:, node_positions, features, 0],
        'totals': gh[:, node_positions, features, -1],
        'prefix_counts': counts[node_positions, features, split_bins],
        'nan_counts': counts[node_positions, features, 0],
    }


def _valid_from_counts(counts, prefix_counts, nan_counts, nan_left, minimum):
    left = np.where(nan_left, prefix_counts, prefix_counts - nan_counts)
    valid = (left >= minimum) & ((counts[:, np.newaxis] - left) >= minimum)
    valid &= nan_left | ((nan_counts > 0) & (prefix_counts < counts[:, np.newaxis]))
    return valid


def _run_case(arr, grad, hess, encoded, node_count, protocol):
    import cupy as cp
    from py_boost.gpu.utils import (
        _candidate_gain_details,
        _fused_merge_top_k_split_candidate_arrays,
        _loss_for_statistics,
        _selected_candidate_statistics,
        _score_candidate_matrix,
    )

    n_rows = int(protocol['n_rows'])
    n_features = int(protocol['n_features'])
    shortlist_size = int(protocol['shortlist_size'])
    budget = int(protocol['diagnostic_output_budget'])
    rows = cp.arange(n_rows, dtype=cp.uint64)
    columns = cp.arange(n_features, dtype=cp.uint64)
    node_indexes_cpu = np.arange(n_rows, dtype=np.int32) % node_count
    node_indexes_cpu[::int(protocol['inactive_row_period'])] = -1
    node_indexes = cp.asarray(node_indexes_cpu)
    params = {
        'max_bin': int(protocol['max_bin']),
        'lambda_l2': float(protocol['lambda_l2']),
        'min_data_in_leaf': int(protocol['min_data_in_leaf']),
    }
    diagnostic_loss = _loss_for_statistics(
        arr,
        grad[:, :budget],
        hess[:, :budget],
        node_indexes,
        rows,
        columns,
        node_count,
        params,
    )
    candidates = _fused_merge_top_k_split_candidate_arrays(
        diagnostic_loss, columns, shortlist_size
    )
    full_loss, full_components, full_histogram = _loss_for_statistics(
        arr,
        grad,
        hess,
        node_indexes,
        rows,
        columns,
        node_count,
        params,
        return_components=True,
        return_histogram=True,
    )
    production = _selected_candidate_statistics(
        arr, grad, hess, node_indexes, rows, candidates, node_count
    )
    segmented = _forced_segmented_statistics(
        arr, grad, hess, node_indexes, rows, candidates, node_count
    )
    production_details = _candidate_gain_details(
        production,
        candidates,
        int(protocol['n_outputs']),
        params['lambda_l2'],
        params['min_data_in_leaf'],
    )
    segmented_details = _candidate_gain_details(
        segmented,
        candidates,
        int(protocol['n_outputs']),
        params['lambda_l2'],
        params['min_data_in_leaf'],
    )
    candidates_cpu = {name: value.get() for name, value in candidates.items()}
    reference = _cpu_float64_reference(
        encoded,
        grad.get(),
        hess.get(),
        node_indexes_cpu,
        candidates_cpu['features'],
        candidates_cpu['split_bins'],
        candidates_cpu['nan_left'],
        params['min_data_in_leaf'],
        params['lambda_l2'],
        int(protocol['roundoff_extra_terms']),
    )
    histogram = _candidate_level_full_histogram(full_histogram, candidates)
    oracle_gains = _score_candidate_matrix(full_loss, columns, candidates).get()
    node_positions = np.arange(node_count)[:, np.newaxis]
    directions = np.where(candidates_cpu['nan_left'], 0, 1)
    histogram_components = full_components.get()[
        :,
        node_positions,
        candidates_cpu['features'],
        candidates_cpu['split_bins'],
        directions,
    ]

    production_cpu = {name: value.get() for name, value in production.items()}
    segmented_cpu = {name: value.get() for name, value in segmented.items()}
    production_detail_cpu = {
        name: value.get() for name, value in production_details.items()
    }
    segmented_detail_cpu = {
        name: value.get() for name, value in segmented_details.items()
    }
    k = shortlist_size
    reference_total = np.repeat(reference['totals'][:, :, np.newaxis], k, axis=2)
    reference_total_bound = np.repeat(
        reference['bounds']['totals'][:, :, np.newaxis], k, axis=2
    )
    paths = {
        'production_batched': {
            'statistics': production_cpu,
            'details': production_detail_cpu,
            'totals_for_comparison': np.repeat(
                production_cpu['totals'][:, :, np.newaxis], k, axis=2
            ),
        },
        'forced_segmented': {
            'statistics': segmented_cpu,
            'details': segmented_detail_cpu,
            'totals_for_comparison': np.repeat(
                segmented_cpu['totals'][:, :, np.newaxis], k, axis=2
            ),
        },
        'full_histogram': {
            'statistics': histogram,
            'details': {
                'component_gains': histogram_components,
                'gains': oracle_gains,
            },
            'totals_for_comparison': histogram['totals'],
        },
    }

    path_reports = {}
    all_bounds_pass = True
    all_winners_pass = True
    all_counts_pass = True
    any_float_difference = False
    rtol = float(protocol['existing_gain_rtol'])
    atol = float(protocol['existing_gain_atol'])
    reference_valid_candidate = reference['valid'][0]
    for name, path in paths.items():
        statistics = path['statistics']
        prefix_report = bounded_error_summary(
            statistics['prefix'], reference['prefix'], reference['bounds']['prefix']
        )
        nan_report = bounded_error_summary(
            statistics['nan'], reference['nan'], reference['bounds']['nan']
        )
        total_report = bounded_error_summary(
            path['totals_for_comparison'], reference_total, reference_total_bound
        )
        if name == 'full_histogram':
            path_counts = np.bincount(
                node_indexes_cpu[node_indexes_cpu >= 0], minlength=node_count
            ).astype(np.uint64)
            path_valid = _valid_from_counts(
                path_counts,
                statistics['prefix_counts'],
                statistics['nan_counts'],
                candidates_cpu['nan_left'],
                params['min_data_in_leaf'],
            )
        else:
            path_counts = statistics['counts']
            path_valid = path['details']['valid'][0]
        count_reports = {
            'counts': _exact_summary(path_counts, reference['counts']),
            'prefix_counts': _exact_summary(
                statistics['prefix_counts'], reference['prefix_counts']
            ),
            'nan_counts': _exact_summary(
                statistics['nan_counts'], reference['nan_counts']
            ),
            'valid': _exact_summary(path_valid, reference_valid_candidate),
        }
        winner_report = _winner_equivalence(
            path['details']['gains'], reference['gains'], rtol, atol
        )
        path_report = {
            'statistics_vs_float64': {
                'prefix': prefix_report,
                'nan': nan_report,
                'totals': total_report,
            },
            'integer_contract_vs_float64': count_reports,
            'component_gains_vs_float64': _gain_summary(
                path['details']['component_gains'],
                reference['component_gains'],
                rtol,
                atol,
            ),
            'total_gains_vs_float64': _gain_summary(
                path['details']['gains'], reference['gains'], rtol, atol
            ),
            'winner_vs_float64': winner_report,
        }
        path_reports[name] = path_report
        bounds_pass = all(
            report['within_bound']
            for report in path_report['statistics_vs_float64'].values()
        )
        counts_pass = all(
            report['exact'] for report in count_reports.values()
        )
        all_bounds_pass &= bounds_pass
        all_counts_pass &= counts_pass
        all_winners_pass &= winner_report['gain_equivalent']
        any_float_difference |= any(
            not report['exactly_equal']
            for report in path_report['statistics_vs_float64'].values()
        )

    production_vs_histogram = _gain_summary(
        production_detail_cpu['gains'], oracle_gains, rtol, atol
    )
    return {
        'active_nodes': node_count,
        'active_rows': int((node_indexes_cpu >= 0).sum()),
        'shortlist_size': shortlist_size,
        'paths': path_reports,
        'production_vs_full_histogram_gains': production_vs_histogram,
        'production_vs_segmented_gains': _gain_summary(
            production_detail_cpu['gains'],
            segmented_detail_cpu['gains'],
            rtol,
            atol,
        ),
        'integer_contract_exact': bool(all_counts_pass),
        'float_statistics_within_roundoff_bound': bool(all_bounds_pass),
        'winner_gains_equivalent_to_float64': bool(all_winners_pass),
        'float_accumulation_difference_observed': bool(any_float_difference),
        'existing_tolerance_failure_reproduced': bool(
            not production_vs_histogram['allclose']
        ),
        'h_numeric_1_supported': bool(
            all_counts_pass
            and all_bounds_pass
            and all_winners_pass
            and any_float_difference
        ),
    }


def run_refinement_numeric_diagnostic(config):
    import cupy as cp

    protocol = config['refinement_numeric_diagnostic']
    arr, grad, hess = _make_inputs(protocol)
    encoded = arr[:, :int(protocol['n_features'])].get()
    cases = [
        _run_case(arr, grad, hess, encoded, int(node_count), protocol)
        for node_count in protocol['node_counts']
    ]
    gate = {
        'all_integer_contracts_exact': all(
            case['integer_contract_exact'] for case in cases
        ),
        'all_float_statistics_within_roundoff_bound': all(
            case['float_statistics_within_roundoff_bound'] for case in cases
        ),
        'all_winner_gains_equivalent_to_float64': all(
            case['winner_gains_equivalent_to_float64'] for case in cases
        ),
        'float_accumulation_difference_observed': any(
            case['float_accumulation_difference_observed'] for case in cases
        ),
        'existing_tolerance_failure_reproduced': any(
            case['existing_tolerance_failure_reproduced'] for case in cases
        ),
    }
    gate['h_numeric_1_supported'] = bool(
        gate['all_integer_contracts_exact']
        and gate['all_float_statistics_within_roundoff_bound']
        and gate['all_winner_gains_equivalent_to_float64']
        and gate['float_accumulation_difference_observed']
    )
    report = {
        'phase': 'refinement_numeric_diagnostic',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': (
            'numeric_accumulation_hypothesis_supported'
            if gate['h_numeric_1_supported']
            else 'numeric_accumulation_hypothesis_inconclusive'
        ),
        'protocol': protocol,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': _gpu_name(cp),
        },
        'cases': cases,
        'gate': gate,
        'new_boosting_fits': 0,
        'media_mill_fits': 0,
        'scope_limit': (
            'Deterministic compact diagnosis only; saved MediaMill traces do not '
            'contain raw gradient/Hessian tensors and no training is repeated.'
        ),
    }
    output = Path('results/revision/refinement_numeric_diagnostic.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'REFINEMENT_NUMERIC_DIAGNOSTIC': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': report['scientific_status'],
        'gate': gate,
        'new_boosting_fits': 0,
        'media_mill_fits': 0,
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
