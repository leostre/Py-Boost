"""Run the preregistered H-DETERMINISM-2 paired frontier capture."""

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
from experiments.refinement_determinism_frontier_core import (
    analyze_determinism_frontier,
)
from experiments.reproducibility import paired_seed_map


def _load_json(path):
    return json.loads(Path(str(path)).read_text(encoding='utf-8'))


def _capture_complete(trace, expected_shortlist):
    return bool(trace) and all(
        'state_fingerprints' in entry
        and all(
            len(node.get('candidate_frontier', ())) == expected_shortlist
            for node in entry['nodes']
        )
        for entry in trace
    )


def run_refinement_determinism(config: dict[str, object]) -> dict[str, object]:
    """Capture and compare two bounded applied-refinement trajectories."""

    protocol = config['decision_aware_refinement_determinism']
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
    args.predict_repeats = 1
    args.gain_risk_threshold = 0.10
    args.split_top_k = 2
    args.split_output_attribution = False
    args.split_candidate_output_profiles = False
    args.split_refinement_shadow = True
    args.split_refinement_apply = True
    args.split_refinement_determinism_capture = True
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
    diagnostics = {}
    for arm, case_id in (
        ('treatment', 'determinism_frontier'),
        ('repeat', 'determinism_frontier_repeat'),
    ):
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
        traces[arm] = _load_json(record['split_refinement_path'])
        diagnostics[arm] = _load_json(record['split_diagnostics_path'])

    capture_checks = {
        arm: _capture_complete(trace, shortlist)
        for arm, trace in traces.items()
    }
    if not all(capture_checks.values()):
        raise RuntimeError(f'incomplete determinism frontier capture: {capture_checks}')
    analysis = analyze_determinism_frontier(
        traces['treatment'],
        traces['repeat'],
        diagnostics['treatment'],
        diagnostics['repeat'],
    )
    report = {
        'phase': 'decision_aware_refinement_determinism',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': analysis['scientific_status'],
        'protocol': protocol,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': benchmark._gpu_name(),
        },
        'capture_checks': capture_checks,
        'records': records,
        'analysis': analysis,
        'scope_limit': (
            'The run captures and classifies the first divergence within a fixed '
            '30-tree horizon. It does not change tie-breaking or validate a fix.'
        ),
    }
    output = Path('results/revision/decision_aware_refinement_determinism.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(json.dumps({'REFINEMENT_DETERMINISM': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': report['scientific_status'],
        'classification': analysis['classification'],
        'new_boosting_fits': 2,
        'trees_per_fit': args.ntrees,
        'first_divergence': analysis['first_divergence'],
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
