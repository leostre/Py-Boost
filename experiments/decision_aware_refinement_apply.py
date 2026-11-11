"""Run the preregistered paired H-APPLY-1 MediaMill experiment."""

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
from experiments.decision_aware_refinement_apply_core import assess_application_gate
from experiments.reproducibility import paired_seed_map


def _load_trace(record: dict[str, object]) -> list[dict[str, object]]:
    path = Path(str(record['split_refinement_path']))
    return json.loads(path.read_text(encoding='utf-8'))


def run_refinement_apply(config: dict[str, object]) -> dict[str, object]:
    """Compare shadow-only and applied refinement on one paired trajectory."""

    protocol = config['decision_aware_refinement_apply']
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
    budget = int(protocol['output_budget'])
    shortlist = int(protocol['shortlist_size'])
    args.split_seed_by_model = paired_seed_map([model_seed], [split_seed])
    args.record_selection_indices = True
    args.profile_methods = False
    args.predict_repeats = 3
    args.gain_risk_threshold = 0.10
    args.split_top_k = 1
    args.split_output_attribution = False
    args.split_candidate_output_profiles = False
    args.split_refinement_shadow = True
    args.split_refinement_shortlists = {budget: shortlist}
    args.research_shadow_budgets = ()
    args.trajectory = 'none'

    dataset = load_benchmark_dataset(
        str(protocol['dataset']),
        args.max_samples,
        args.dataset_seed,
        args.cache_dir,
    )
    split = split_benchmark_dataset(dataset, split_seed, args.test_size)
    args.output_min_fraction = budget / dataset.n_outputs
    defaults = PlanDefaults(
        row_subsample=1.0,
        sketch_outputs=budget,
        min_warmup_rounds=5,
        adaptive_budgets=(budget, dataset.n_outputs),
    )

    records = {}
    traces = {}
    arms = (
        ('baseline', 'refinement_shadow_baseline', False),
        ('treatment', 'refinement_apply', True),
        ('repeat', 'refinement_apply_repeat', True),
    )
    for arm, case_id, apply_refined in arms:
        args.split_refinement_apply = apply_refined
        case = ExperimentCase(
            case_id,
            Variant.HASBOOST_CURRENT_TOPK,
            scheduler_mode=SchedulerMode.NEVER,
            row_subsample=1.0,
            sketch_outputs=budget,
            collect_split_diagnostics=True,
            full_leaf_values=True,
        ).resolve(defaults, dataset.n_outputs)
        run = PlannedRun(Scenario.OBSERVABILITY, case, model_seed)
        record = benchmark.evaluate_run(run, args, dataset, *split)
        records[arm] = record
        traces[arm] = _load_trace(record)

    gate = assess_application_gate(
        records,
        traces,
        expected_trees=args.ntrees,
        expected_budget=budget,
        expected_shortlist=shortlist,
        maximum_log_loss_increase=float(protocol['maximum_log_loss_increase']),
        maximum_f1_weighted_decrease=float(
            protocol['maximum_f1_weighted_decrease']
        ),
        maximum_runtime_ratio=float(protocol['maximum_runtime_ratio']),
    )
    report = {
        'phase': 'decision_aware_refinement_apply',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': (
            'refinement_application_gate_passed'
            if gate['passed']
            else 'refinement_application_gate_failed'
        ),
        'protocol': protocol,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': benchmark._gpu_name(),
        },
        'records': records,
        'gate': gate,
        'scope_limit': (
            'Both arms retained full-search oracle diagnostics. Runtime therefore '
            'measures application safety and overhead parity, not production speedup.'
        ),
    }
    output = args.output_dir / f'decision_aware_refinement_apply_{args.run_id}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'DECISION_AWARE_REFINEMENT_APPLY': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': report['scientific_status'],
        'new_boosting_fits': 3,
        'model_seed': model_seed,
        'split_seed': split_seed,
        'gate_passed': gate['passed'],
        'gate': gate,
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
