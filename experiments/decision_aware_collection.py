"""Collect one full MediaMill shadow trajectory for shortlist feasibility."""

from __future__ import annotations

import json
import platform
import sys
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
from experiments.decision_aware_feasibility_core import (
    assess_shortlist_recall_gate,
    evaluate_shortlist_collection,
)
from experiments.reproducibility import paired_seed_map
from experiments.split_diagnostic_artifact import load_split_diagnostic_artifact


def run_collection(config: dict[str, object]) -> dict[str, object]:
    """Run the single preregistered fit and audit its compact diagnostics."""

    protocol = config['decision_aware_collection']
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
    args.split_seed_by_model = paired_seed_map([model_seed], [split_seed])
    args.record_selection_indices = True
    args.profile_methods = False
    args.predict_repeats = 3
    args.gain_risk_threshold = 0.10
    args.split_top_k = int(protocol['top_k'])
    args.split_output_attribution = False
    args.split_candidate_output_profiles = True
    args.research_shadow_budgets = tuple(int(value) for value in protocol['budgets'][1:])
    args.trajectory = 'none'

    dataset = load_benchmark_dataset(
        str(protocol['dataset']),
        args.max_samples,
        args.dataset_seed,
        args.cache_dir,
    )
    split = split_benchmark_dataset(dataset, split_seed, args.test_size)
    args.output_min_fraction = min(int(protocol['budgets'][0]), dataset.n_outputs) / dataset.n_outputs
    defaults = PlanDefaults(
        row_subsample=1.0,
        sketch_outputs=int(protocol['budgets'][0]),
        min_warmup_rounds=5,
        adaptive_budgets=tuple(int(value) for value in protocol['budgets']) + (dataset.n_outputs,),
    )
    case = ExperimentCase(
        'full_shadow_collection',
        Variant.HASBOOST_CURRENT_TOPK,
        scheduler_mode=SchedulerMode.NEVER,
        row_subsample=1.0,
        sketch_outputs=int(protocol['budgets'][0]),
        collect_split_diagnostics=True,
        full_leaf_values=True,
    ).resolve(defaults, dataset.n_outputs)
    run = PlannedRun(Scenario.OBSERVABILITY, case, model_seed)
    record = benchmark.evaluate_run(run, args, dataset, *split)
    metadata_path = Path(record['split_diagnostics_path'])
    profile_path = Path(record['split_diagnostics_profiles_path'])
    diagnostics = load_split_diagnostic_artifact(metadata_path, profile_path)
    evaluation = evaluate_shortlist_collection(
        diagnostics,
        tuple(int(value) for value in protocol['shortlist_sizes']),
        tuple(int(value) for value in protocol['budgets']),
        float(protocol['material_gain_epsilon']),
        float(protocol['contribution_tolerance']),
        float(protocol.get('contribution_relative_tolerance', 1e-6)),
    )
    recall_gate = assess_shortlist_recall_gate(
        evaluation,
        tuple(int(value) for value in protocol['shortlist_sizes']),
        tuple(int(value) for value in protocol['budgets']),
        float(protocol['minimum_recall_at_k']),
    )
    report = {
        'phase': 'decision_aware_collection',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': (
            'collection_contract_valid' if evaluation['contract_valid']
            else 'collection_contract_failed'
        ),
        'protocol': protocol,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': benchmark._gpu_name(),
        },
        'fit': record,
        'evaluation': evaluation,
        'shortlist_recall_gate': recall_gate,
        'scope_limit': (
            'This development trajectory evaluates data completeness and shortlist recall only. '
            'It does not train a selector or evaluate regret reduction and runtime cost.'
        ),
    }
    output = args.output_dir / f'decision_aware_collection_{args.run_id}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'DECISION_AWARE_COLLECTION': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': report['scientific_status'],
        'new_boosting_fits': 1,
        'records': len(diagnostics),
        'contract_valid': evaluation['contract_valid'],
        'recall_at_k': {
            budget: values['recall_at_k']
            for budget, values in evaluation['per_budget'].items()
        },
        'joint_minimum_viable_k': recall_gate['joint_minimum_viable_k'],
        'shortlist_recall_gate_passed': recall_gate['passed'],
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
