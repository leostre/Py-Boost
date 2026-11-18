"""Run one preregistered GPU shadow trajectory for exact shortlist refinement."""

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
from experiments.decision_aware_refinement_shadow_core import (
    assess_refinement_shadow_gate,
    evaluate_refinement_shadow,
)
from experiments.reproducibility import paired_seed_map


def run_refinement_shadow(config: dict[str, object]) -> dict[str, object]:
    """Measure exact candidate-only scoring against a full-search oracle."""

    protocol = config['decision_aware_refinement_shadow']
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
    args.predict_repeats = 3
    args.gain_risk_threshold = 0.10
    args.split_top_k = 1
    args.split_output_attribution = False
    args.split_candidate_output_profiles = False
    args.split_refinement_shadow = True
    args.split_refinement_shortlists = shortlists
    args.research_shadow_budgets = budgets[1:]
    args.trajectory = 'none'

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
        'exact_refinement_shadow',
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
    evaluation = evaluate_refinement_shadow(
        trace,
        budgets,
        float(protocol['material_gain_epsilon']),
    )
    gate = assess_refinement_shadow_gate(
        evaluation,
        budgets,
        float(protocol['minimum_weighted_p95_regret_reduction']),
    )
    report = {
        'phase': 'decision_aware_refinement_shadow',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': 'gpu_shadow_gate_passed' if gate['passed'] else 'gpu_shadow_gate_failed',
        'protocol': protocol,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': benchmark._gpu_name(),
        },
        'fit': record,
        'evaluation': evaluation,
        'gate': gate,
        'scope_limit': (
            'Refined candidates were evaluated and timed in shadow mode only. '
            'The resulting split was never applied to the trained tree.'
        ),
    }
    output = args.output_dir / f'decision_aware_refinement_shadow_{args.run_id}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'DECISION_AWARE_REFINEMENT_SHADOW': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': report['scientific_status'],
        'new_boosting_fits': 1,
        'model_seed': model_seed,
        'split_seed': split_seed,
        'gate_passed': gate['passed'],
        'per_budget': evaluation['per_budget'],
        'report': str(output.resolve()),
        'trace': str(refinement_path.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
