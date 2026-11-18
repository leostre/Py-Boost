"""GPU benchmark for exact global and fully fused top-K extraction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np


def _synchronized_milliseconds(operation) -> float:
    import cupy as cp

    cp.cuda.Device().synchronize()
    started = perf_counter()
    operation()
    cp.cuda.Device().synchronize()
    return (perf_counter() - started) * 1_000.0


def _paired_timings(baseline, fused, repeats: int):
    """Alternate call order so launch order cannot favor one implementation."""

    measurements = {'baseline': [], 'fused': []}
    for index in range(repeats):
        order = (
            (('baseline', baseline), ('fused', fused))
            if index % 2 == 0
            else (('fused', fused), ('baseline', baseline))
        )
        for name, operation in order:
            measurements[name].append(_synchronized_milliseconds(operation))
    return measurements


def _cpu_arrays(result):
    return {name: values.get() for name, values in result.items()}


def _correctness(unique_loss, columns, top_k: int) -> dict[str, object]:
    from py_boost.gpu.utils import (
        _fused_merge_top_k_split_candidate_arrays,
        _top_k_split_candidate_arrays,
    )

    baseline = _cpu_arrays(
        _top_k_split_candidate_arrays(unique_loss, columns, top_k)
    )
    fused = _cpu_arrays(
        _fused_merge_top_k_split_candidate_arrays(unique_loss, columns, top_k)
    )
    fields = ('features', 'split_bins', 'nan_left', 'diagnostic_gains')
    exact_fields = {
        name: bool(np.array_equal(baseline[name], fused[name]))
        for name in fields
    }
    return {'passed': all(exact_fields.values()), 'exact_fields': exact_fields}


def _tie_correctness(tied_loss, columns, top_k: int) -> dict[str, object]:
    from py_boost.gpu.utils import (
        _fused_merge_top_k_split_candidate_arrays,
        _top_k_split_candidate_arrays,
    )

    baseline = _top_k_split_candidate_arrays(tied_loss, columns, top_k)
    fused = _fused_merge_top_k_split_candidate_arrays(
        tied_loss, columns, top_k
    )
    gain_multiset_equal = bool(np.array_equal(
        baseline['diagnostic_gains'].get(),
        fused['diagnostic_gains'].get(),
    ))
    return {'passed': gain_multiset_equal, 'gain_multiset_equal': gain_multiset_equal}


def run_top_k_benchmark(config: dict[str, object]) -> dict[str, object]:
    import cupy as cp
    from py_boost.gpu.utils import (
        _fused_merge_top_k_split_candidate_arrays,
        _top_k_split_candidate_arrays,
    )

    protocol = config['decision_aware_fused_merge_top_k_benchmark']
    node_counts = tuple(int(value) for value in protocol['node_counts'])
    shortlist_sizes = tuple(int(value) for value in protocol['shortlist_sizes'])
    nfeatures = int(protocol['n_features'])
    nbins = int(protocol['max_bin'])
    ndirections = int(protocol['nan_directions'])
    repeats = int(protocol['repeats'])
    warmups = int(protocol['warmups'])
    threshold = float(protocol['max_runtime_ratio'])
    columns = cp.arange(nfeatures, dtype=cp.int64)
    cases = []

    for nnodes in node_counts:
        size = nnodes * nfeatures * nbins * ndirections
        rng = np.random.default_rng(int(protocol['seed']) + nnodes)
        unique_cpu = rng.permutation(size).astype(np.float32).reshape(
            nnodes, nfeatures, nbins, ndirections
        )
        unique_loss = cp.asarray(unique_cpu)
        tied_loss = cp.asarray(np.remainder(unique_cpu, 7.0))
        for top_k in shortlist_sizes:
            baseline = lambda: _top_k_split_candidate_arrays(
                unique_loss, columns, top_k
            )
            fused = lambda: _fused_merge_top_k_split_candidate_arrays(
                unique_loss, columns, top_k
            )
            for _ in range(warmups):
                baseline()
                fused()
            cp.cuda.Device().synchronize()

            timings = _paired_timings(baseline, fused, repeats)
            baseline_median = float(np.median(timings['baseline']))
            fused_median = float(np.median(timings['fused']))
            unique_correctness = _correctness(unique_loss, columns, top_k)
            tie_correctness = _tie_correctness(tied_loss, columns, top_k)
            cases.append({
                'active_nodes': nnodes,
                'top_k': top_k,
                'candidate_count_per_node': nfeatures * nbins * ndirections,
                'reduced_candidate_count_per_node': nfeatures * min(
                    top_k, nbins * ndirections
                ),
                'baseline_median_ms': baseline_median,
                'fused_median_ms': fused_median,
                'runtime_ratio': fused_median / baseline_median,
                'correctness': {
                    'passed': unique_correctness['passed'] and tie_correctness['passed'],
                    'unique_candidates': unique_correctness,
                    'tied_candidates': tie_correctness,
                },
            })
        del unique_loss, tied_loss

    per_k = {}
    for top_k in shortlist_sizes:
        selected = [case for case in cases if case['top_k'] == top_k]
        baseline_total = sum(case['baseline_median_ms'] for case in selected)
        fused_total = sum(
            case['fused_median_ms'] for case in selected
        )
        ratio = fused_total / baseline_total
        correct = all(case['correctness']['passed'] for case in selected)
        per_k[str(top_k)] = {
            'baseline_weighted_total_ms': baseline_total,
            'fused_weighted_total_ms': fused_total,
            'runtime_ratio': ratio,
            'correctness_passed': correct,
            'performance_passed': ratio <= threshold,
            'passed': correct and ratio <= threshold,
        }

    gate_passed = all(result['passed'] for result in per_k.values())
    report = {
        'phase': 'decision_aware_fused_merge_top_k_benchmark',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': 'gate_passed' if gate_passed else 'gate_failed',
        'protocol': protocol,
        'implementation': {
            'feature_local_stage': 'one RawKernel block per node-feature pair',
            'feature_local_ordering': 'shared-memory bitonic ordering',
            'global_stage': 'one exact shared-memory k-way merge RawKernel block per node',
            'cupy_partition_or_sort_in_candidate_selection': False,
            'jit_compile_and_warmup_timed': False,
            'workspace_allocation_timed': True,
        },
        'cases': cases,
        'per_k': per_k,
        'gate': {
            'passed': gate_passed,
            'max_runtime_ratio': threshold,
            'aggregation': 'sum of medians with equal weight for each active-node count',
        },
        'new_boosting_fits': 0,
        'media_mill_fits': 0,
        'scope_limit': (
            'This microbenchmark measures exact top-k extraction only. It does not '
            'measure histogram construction, candidate rescoring, or model training.'
        ),
    }
    output = Path('results/revision/fused_merge_top_k_benchmark.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'TOP_K_BENCHMARK': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': report['scientific_status'],
        'gate_passed': gate_passed,
        'per_k': per_k,
        'new_boosting_fits': 0,
        'media_mill_fits': 0,
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
