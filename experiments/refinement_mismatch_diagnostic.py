"""Reproduce and localize the first-tree MediaMill refinement mismatch."""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

from experiments import benchmark_helena as benchmark
from experiments.benchmark_datasets import load_benchmark_dataset, split_benchmark_dataset
from experiments.benchmark_plan import (
    ExperimentCase,
    PlanDefaults,
    PlannedRun,
    Scenario,
    SchedulerMode,
    Variant,
)
from experiments.reproducibility import paired_seed_map


def summarize_failure_diagnostics(trace):
    """Build a compact decision record from detailed refinement batches."""

    batches = [entry for entry in trace if entry.get('failure_diagnostics')]
    records = [
        {
            'iteration': int(entry['iteration']),
            'group': int(entry['group']),
            'depth': int(entry['depth']),
            'budget': int(entry['diagnostic_output_budget']),
            **record,
        }
        for entry in batches
        for record in entry['failure_diagnostics']['records']
    ]
    severe_records = [
        record for record in records
        if abs(float(record['candidate_gain']) - float(record['original_oracle_gain'])) > 1.0
    ]
    target_records = [
        record for record in severe_records
        if record['iteration'] == 1
        and record['group'] == 0
        and record['depth'] == 4
        and record['node_id'] == 23
    ]
    first_stages = Counter(
        str(record['first_divergent_stage']) for record in severe_records
    )
    closer_references = Counter(
        str(record['closer_to_float64_gain']) for record in severe_records
    )
    if target_records and all(record['first_divergent_stage'] for record in target_records):
        status = 'critical_mismatch_localized'
    elif severe_records:
        status = 'severe_mismatch_reproduced_but_not_localized'
    else:
        status = 'critical_mismatch_not_reproduced'
    return {
        'diagnostic_status': status,
        'failure_batches': len(batches),
        'diagnostic_records': len(records),
        'severe_records': len(severe_records),
        'target_records': len(target_records),
        'first_divergent_stage_counts': dict(sorted(first_stages.items())),
        'closer_to_float64_gain_counts': dict(sorted(closer_references.items())),
        'targets': target_records,
    }


def summarize_refinement_correctness(trace):
    """Summarize candidate/full agreement across every recorded refinement batch."""

    batches = [
        entry for entry in trace
        if 'candidate_gains_allclose' in entry
    ]
    return {
        'batches': len(batches),
        'candidate_gains_allclose': bool(batches) and all(
            bool(entry['candidate_gains_allclose']) for entry in batches
        ),
        'all_refined_winners_gain_equivalent': bool(batches) and all(
            bool(entry['all_refined_winners_gain_equivalent']) for entry in batches
        ),
        'maximum_candidate_gain_absolute_error': max(
            (
                float(entry['maximum_candidate_gain_absolute_error'])
                for entry in batches
            ),
            default=0.0,
        ),
        'maximum_candidate_gain_relative_error': max(
            (
                float(entry['maximum_candidate_gain_relative_error'])
                for entry in batches
            ),
            default=0.0,
        ),
    }


def run_refinement_mismatch_diagnostic(config):
    """Fit one tree and capture stage-wise statistics for severe gain errors."""

    protocol = config['decision_aware_refinement_mismatch_diagnostic']
    previous_argv = sys.argv
    sys.argv = ['benchmark_helena']
    try:
        args = benchmark.parse_args()
    finally:
        sys.argv = previous_argv

    args.dataset_seed = int(config['dataset_seed'])
    args.max_samples = int(config['max_samples'])
    args.ntrees = int(config['ntrees'])
    args.max_depth = int(config['max_depth'])
    args.cache_dir = Path(str(config['cache_dir']))
    args.output_dir = Path('results/revision')
    args.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    model_seed = int(protocol['model_seed'])
    split_seed = int(protocol['split_seed'])
    budgets = tuple(int(value) for value in protocol['budgets'])
    shortlists = {
        int(budget): int(size)
        for budget, size in protocol['shortlist_sizes_by_budget'].items()
    }
    args.split_seed_by_model = paired_seed_map([model_seed], [split_seed])
    args.record_selection_indices = True
    args.profile_methods = False
    args.predict_repeats = 1
    args.gain_risk_threshold = 0.10
    args.split_top_k = 1
    args.split_output_attribution = False
    args.split_candidate_output_profiles = False
    args.split_refinement_shadow = True
    args.split_refinement_shortlists = shortlists
    args.split_refinement_failure_diagnostics = True
    args.split_refinement_severe_error_threshold = float(
        protocol['severe_gain_error_threshold']
    )
    args.research_shadow_budgets = budgets[1:]
    args.trajectory = 'none'

    if args.ntrees != 1:
        raise ValueError('mismatch diagnostic must build exactly one tree')
    dataset = load_benchmark_dataset(
        str(protocol['dataset']),
        args.max_samples,
        args.dataset_seed,
        args.cache_dir,
    )
    split = split_benchmark_dataset(dataset, split_seed, args.test_size)
    args.output_min_fraction = min(budgets[0], dataset.n_outputs) / dataset.n_outputs
    defaults = PlanDefaults(
        row_subsample=1.0,
        sketch_outputs=budgets[0],
        min_warmup_rounds=5,
        adaptive_budgets=budgets + (dataset.n_outputs,),
    )
    case = ExperimentCase(
        'refinement_mismatch_diagnostic',
        Variant.HASBOOST_CURRENT_TOPK,
        scheduler_mode=SchedulerMode.NEVER,
        row_subsample=1.0,
        sketch_outputs=budgets[0],
        collect_split_diagnostics=True,
        full_leaf_values=True,
    ).resolve(defaults, dataset.n_outputs)
    run = PlannedRun(Scenario.OBSERVABILITY, case, model_seed)
    record = benchmark.evaluate_run(run, args, dataset, *split)
    refinement_path = Path(record['split_refinement_path'])
    trace = json.loads(refinement_path.read_text(encoding='utf-8'))
    diagnostic = summarize_failure_diagnostics(trace)
    correctness = summarize_refinement_correctness(trace)
    report = {
        'phase': 'decision_aware_refinement_mismatch_diagnostic',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': protocol,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': benchmark._gpu_name(),
        },
        'fit': record,
        'diagnostic': diagnostic,
        'correctness': correctness,
        'scope_limit': (
            'Exactly one tree was fitted. Detailed statistics were captured only '
            'after a severe shadow mismatch; no refined split was applied.'
        ),
    }
    output = args.output_dir / f'refinement_mismatch_diagnostic_{args.run_id}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'REFINEMENT_MISMATCH_DIAGNOSTIC': {
        'phase': report['phase'],
        'protocol': protocol,
        'environment': report['environment'],
        'fit': record,
        'diagnostic': {
            key: value for key, value in diagnostic.items() if key != 'targets'
        },
        'correctness': correctness,
        'target_summaries': [
            {
                key: value for key, value in target.items()
                if key not in ('values', 'comparisons')
            }
            for target in diagnostic['targets']
        ],
        'report': str(output.resolve()),
        'trace': str(refinement_path.resolve()),
    }}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'diagnostic_status': diagnostic['diagnostic_status'],
        'new_boosting_fits': 1,
        'trees_built': record['trees_built'],
        'model_seed': model_seed,
        'split_seed': split_seed,
        'failure_batches': diagnostic['failure_batches'],
        'severe_records': diagnostic['severe_records'],
        'target_records': diagnostic['target_records'],
        'candidate_gains_allclose': correctness['candidate_gains_allclose'],
        'all_refined_winners_gain_equivalent': (
            correctness['all_refined_winners_gain_equivalent']
        ),
        'maximum_candidate_gain_absolute_error': (
            correctness['maximum_candidate_gain_absolute_error']
        ),
        'maximum_candidate_gain_relative_error': (
            correctness['maximum_candidate_gain_relative_error']
        ),
        'first_divergent_stage_counts': diagnostic['first_divergent_stage_counts'],
        'closer_to_float64_gain_counts': diagnostic['closer_to_float64_gain_counts'],
        'report': str(output.resolve()),
        'trace': str(refinement_path.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
