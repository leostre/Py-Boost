"""Controlled end-to-end benchmark for exact shortlist refinement shadow mode."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from math import fsum
from pathlib import Path

import numpy as np


def summarize_integration_cases(cases, policies, maximum_runtime_ratio):
    """Aggregate node-count medians independently for each budget policy."""

    per_policy = {}
    for budget, shortlist_size in sorted(
        (int(budget), int(size)) for budget, size in policies.items()
    ):
        selected = [
            case for case in cases
            if int(case['output_budget']) == budget
            and int(case['shortlist_size']) == shortlist_size
        ]
        if not selected:
            raise ValueError(f'no integration cases for budget {budget}')
        two_stage_total = fsum(
            float(case['two_stage_median_ms']) for case in selected
        )
        full_search_total = fsum(
            float(case['full_search_median_ms']) for case in selected
        )
        ratio = two_stage_total / full_search_total
        correctness = all(bool(case['correctness']['passed']) for case in selected)
        per_policy[str(budget)] = {
            'shortlist_size': shortlist_size,
            'node_count_cases': sorted(int(case['active_nodes']) for case in selected),
            'two_stage_weighted_total_ms': two_stage_total,
            'full_search_weighted_total_ms': full_search_total,
            'runtime_ratio': ratio,
            'correctness_passed': correctness,
            'runtime_passed': ratio < float(maximum_runtime_ratio),
            'passed': correctness and ratio < float(maximum_runtime_ratio),
        }
    return {
        'maximum_runtime_ratio': float(maximum_runtime_ratio),
        'aggregation': 'sum of per-node-count medians',
        'per_policy': per_policy,
        'passed': all(item['passed'] for item in per_policy.values()),
    }


def _gpu_name(cp):
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties['name']
    return name.decode() if isinstance(name, bytes) else str(name)


def _make_inputs(protocol):
    import cupy as cp
    from py_boost.gpu.utils import pad_and_move

    n_rows = int(protocol['n_rows'])
    n_features = int(protocol['n_features'])
    n_outputs = int(protocol['n_outputs'])
    max_bin = int(protocol['max_bin'])
    rng = np.random.default_rng(int(protocol['seed']))
    encoded = rng.integers(
        0,
        max_bin,
        size=(n_rows, n_features),
        dtype=np.uint16,
    ).astype(np.uint8)
    encoded[::11, ::7] = 0
    rows = np.arange(n_rows, dtype=np.float32)[:, np.newaxis]
    outputs = np.arange(n_outputs, dtype=np.float32)[np.newaxis, :]
    grad = (
        np.sin(rows * np.float32(0.013) + outputs * np.float32(0.071))
        + np.cos(rows * np.float32(0.0017) - outputs * np.float32(0.037))
    ).astype(np.float32)
    hess = (
        np.float32(0.08)
        + np.float32(0.12)
        / (np.float32(1.0) + np.exp(-(rows * np.float32(0.002) - outputs * np.float32(0.011))))
    ).astype(np.float32)
    return pad_and_move(encoded), cp.asarray(grad), cp.asarray(hess)


def _selector_equivalence(diagnostic_loss, full_loss, columns, shortlist_size):
    import cupy as cp
    from py_boost.gpu.utils import (
        _fused_merge_top_k_split_candidate_arrays,
        _score_candidate_matrix,
        _top_k_split_candidate_arrays,
    )

    baseline = _top_k_split_candidate_arrays(
        diagnostic_loss, columns, shortlist_size
    )
    fused = _fused_merge_top_k_split_candidate_arrays(
        diagnostic_loss, columns, shortlist_size
    )
    cp.cuda.get_current_stream().synchronize()
    baseline_cpu = {name: value.get() for name, value in baseline.items()}
    fused_cpu = {name: value.get() for name, value in fused.items()}
    gain_multiset_equal = bool(np.array_equal(
        baseline_cpu['diagnostic_gains'], fused_cpu['diagnostic_gains']
    ))
    unique_selected_gains = bool(all(
        np.unique(node_gains).size == node_gains.size
        for node_gains in baseline_cpu['diagnostic_gains']
    ))
    identity_fields = ('features', 'split_bins', 'nan_left')
    identities_equal = all(
        np.array_equal(baseline_cpu[name], fused_cpu[name])
        for name in identity_fields
    )

    baseline_oracle = _score_candidate_matrix(full_loss, columns, baseline)
    fused_oracle = _score_candidate_matrix(full_loss, columns, fused)
    baseline_indices = baseline_oracle.argmax(axis=1)
    fused_indices = fused_oracle.argmax(axis=1)
    node_positions = cp.arange(full_loss.shape[0])
    baseline_winner_gains = baseline_oracle[node_positions, baseline_indices].get()
    fused_winner_gains = fused_oracle[node_positions, fused_indices].get()
    refined_winner_gains_equivalent = bool(np.allclose(
        baseline_winner_gains,
        fused_winner_gains,
        rtol=5e-5,
        atol=5e-3,
    ))
    passed = bool(
        gain_multiset_equal
        and (identities_equal or not unique_selected_gains)
        and refined_winner_gains_equivalent
    )
    return {
        'gain_multiset_equal': gain_multiset_equal,
        'unique_selected_gains': unique_selected_gains,
        'identities_equal': identities_equal,
        'refined_winner_gains_equivalent': refined_winner_gains_equivalent,
        'passed': passed,
    }


def _run_once(
    arr,
    full_grad,
    full_hess,
    diagnostic_grad,
    diagnostic_hess,
    node_indexes,
    rows,
    columns,
    unique_nodes,
    budget,
    shortlist_size,
    params,
    full_first,
):
    from py_boost.gpu.utils import (
        _build_refinement_shadow_entry,
        _loss_for_statistics,
        _timed_gpu_call,
        get_best_split,
    )

    diagnostic_args = (
        arr, diagnostic_grad, diagnostic_hess, node_indexes, rows,
        columns, len(unique_nodes), params,
    )
    full_args = (
        arr, full_grad, full_hess, node_indexes, rows,
        columns, len(unique_nodes), params,
    )
    if full_first:
        full_loss, full_statistics_seconds = _timed_gpu_call(
            _loss_for_statistics, *full_args
        )
        diagnostic_loss, diagnostic_statistics_seconds = _timed_gpu_call(
            _loss_for_statistics, *diagnostic_args
        )
    else:
        diagnostic_loss, diagnostic_statistics_seconds = _timed_gpu_call(
            _loss_for_statistics, *diagnostic_args
        )
        full_loss, full_statistics_seconds = _timed_gpu_call(
            _loss_for_statistics, *full_args
        )
    full_best, full_selection_seconds = _timed_gpu_call(
        get_best_split, full_loss, columns
    )
    entry = _build_refinement_shadow_entry(
        arr,
        full_grad,
        full_hess,
        node_indexes,
        rows,
        columns,
        unique_nodes,
        diagnostic_loss,
        diagnostic_statistics_seconds,
        full_loss,
        None,
        full_best,
        full_statistics_seconds,
        full_selection_seconds,
        budget,
        shortlist_size,
        params,
    )
    return entry, diagnostic_loss, full_loss


def _median_milliseconds(entries, field):
    return float(np.median([float(entry[field]) for entry in entries]) * 1_000.0)


def run_refinement_integration_benchmark(config):
    import cupy as cp

    protocol = config['decision_aware_refinement_integration_benchmark']
    policies = {
        int(budget): int(size)
        for budget, size in protocol['shortlist_sizes_by_budget'].items()
    }
    node_counts = tuple(int(value) for value in protocol['node_counts'])
    warmups = int(protocol['warmups'])
    repeats = int(protocol['repeats'])
    params = {
        'max_bin': int(protocol['max_bin']),
        'lambda_l2': float(protocol['lambda_l2']),
        'min_data_in_leaf': int(protocol['min_data_in_leaf']),
        'min_gain_to_split': float(protocol['min_gain_to_split']),
    }
    arr, full_grad, full_hess = _make_inputs(protocol)
    n_rows = int(protocol['n_rows'])
    n_features = int(protocol['n_features'])
    rows = cp.arange(n_rows, dtype=cp.uint64)
    columns = cp.arange(n_features, dtype=cp.uint64)
    cases = []

    for active_nodes in node_counts:
        node_indexes_cpu = np.arange(n_rows, dtype=np.int32) % active_nodes
        node_indexes_cpu[::17] = -1
        node_indexes = cp.asarray(node_indexes_cpu)
        unique_nodes = np.arange(active_nodes, dtype=np.int32)
        for budget, shortlist_size in sorted(policies.items()):
            diagnostic_grad = full_grad[:, :budget]
            diagnostic_hess = full_hess[:, :budget]
            for index in range(warmups):
                _run_once(
                    arr, full_grad, full_hess, diagnostic_grad, diagnostic_hess,
                    node_indexes, rows, columns, unique_nodes, budget,
                    shortlist_size, params, full_first=bool(index % 2),
                )
            cp.cuda.get_current_stream().synchronize()

            entries = []
            diagnostic_loss = full_loss = None
            for index in range(repeats):
                entry, diagnostic_loss, full_loss = _run_once(
                    arr, full_grad, full_hess, diagnostic_grad, diagnostic_hess,
                    node_indexes, rows, columns, unique_nodes, budget,
                    shortlist_size, params, full_first=bool(index % 2),
                )
                entries.append(entry)
            selector = _selector_equivalence(
                diagnostic_loss, full_loss, columns, shortlist_size
            )
            entry_correctness = {
                'candidate_gains_allclose': all(
                    bool(entry['candidate_gains_allclose']) for entry in entries
                ),
                'refined_winners_gain_equivalent': all(
                    bool(entry['all_refined_winners_gain_equivalent'])
                    for entry in entries
                ),
                'selector': selector,
            }
            entry_correctness['passed'] = bool(
                entry_correctness['candidate_gains_allclose']
                and entry_correctness['refined_winners_gain_equivalent']
                and selector['passed']
            )
            two_stage_median = _median_milliseconds(entries, 'two_stage_seconds')
            full_search_median = _median_milliseconds(entries, 'full_search_seconds')
            cases.append({
                'active_nodes': active_nodes,
                'output_budget': budget,
                'shortlist_size': shortlist_size,
                'active_rows': int((node_indexes_cpu >= 0).sum()),
                'candidate_count_per_node': n_features * int(protocol['max_bin']) * 2,
                'initial_statistics_median_ms': _median_milliseconds(
                    entries, 'initial_statistics_seconds'
                ),
                'shortlist_selection_median_ms': _median_milliseconds(
                    entries, 'shortlist_selection_seconds'
                ),
                'refinement_median_ms': _median_milliseconds(
                    entries, 'refinement_seconds'
                ),
                'two_stage_median_ms': two_stage_median,
                'full_statistics_median_ms': _median_milliseconds(
                    entries, 'full_statistics_seconds'
                ),
                'full_selection_median_ms': _median_milliseconds(
                    entries, 'full_selection_seconds'
                ),
                'full_search_median_ms': full_search_median,
                'runtime_ratio': two_stage_median / full_search_median,
                'correctness': entry_correctness,
            })
            del diagnostic_loss, full_loss, entries
        del node_indexes

    gate = summarize_integration_cases(
        cases,
        policies,
        float(protocol['maximum_runtime_ratio']),
    )
    report = {
        'phase': 'decision_aware_refinement_integration_benchmark',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': 'integration_gate_passed' if gate['passed'] else 'integration_gate_failed',
        'protocol': protocol,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': _gpu_name(cp),
        },
        'implementation': {
            'shortlist_selector': 'exact fused feature-list merge RawKernel',
            'refinement': (
                'candidate-only exact full-output statistics with exact uint64 '
                'counts and a segmented candidate kernel through K=64'
            ),
            'tree_application': False,
            'jit_compile_and_warmup_timed': False,
            'workspace_allocation_timed': True,
        },
        'cases': cases,
        'gate': gate,
        'new_benchmark_boosting_fits': 0,
        'acceptance_test_boosting_fits': 2,
        'media_mill_fits': 0,
        'scope_limit': (
            'The controlled benchmark exercises the complete GPU shadow computation '
            'but does not train a tree or apply refined winners to model structure.'
        ),
    }
    output = Path('results/revision/refinement_integration_benchmark.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'REFINEMENT_INTEGRATION_BENCHMARK': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': report['scientific_status'],
        'gate_passed': gate['passed'],
        'per_policy': gate['per_policy'],
        'new_benchmark_boosting_fits': 0,
        'acceptance_test_boosting_fits': 2,
        'media_mill_fits': 0,
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
