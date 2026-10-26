"""Run the preregistered H-LEAF-DETERMINISM-1 paired experiment."""

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
from experiments.deterministic_leaf_reduction_core import (
    analyze_deterministic_leaf_reduction,
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


def _relative_overhead(current, baseline):
    return current / baseline - 1.0


def _paired_quality(records):
    fields = ('accuracy', 'f1_weighted', 'f1_macro', 'f1_micro', 'log_loss')
    return {
        field: {
            'treatment': records['treatment'][field],
            'repeat': records['repeat'][field],
            'absolute_delta': abs(
                records['treatment'][field] - records['repeat'][field]
            ),
        }
        for field in fields
    }


def run_deterministic_leaf_reduction(config: dict[str, object]) -> dict[str, object]:
    """Compare two fixed-seed trajectories with deterministic leaf sums."""

    protocol = config['deterministic_leaf_reduction']
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
    args.deterministic_leaf_values = True
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
    for arm, case_id in (
        ('treatment', 'deterministic_leaf_values'),
        ('repeat', 'deterministic_leaf_values_repeat'),
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
        if record.get('deterministic_leaf_values') is not True:
            raise RuntimeError('deterministic leaf-value reduction was not enabled')
        records[arm] = record
        traces[arm] = _load_json(record['split_refinement_path'])

    capture_checks = {
        arm: _capture_complete(trace, shortlist)
        for arm, trace in traces.items()
    }
    if not all(capture_checks.values()):
        raise RuntimeError(f'incomplete determinism frontier capture: {capture_checks}')
    analysis = analyze_deterministic_leaf_reduction(
        traces['treatment'],
        traces['repeat'],
    )
    baseline_seconds = [float(value) for value in protocol['baseline_train_seconds']]
    current_seconds = [
        float(records['treatment']['train_seconds']),
        float(records['repeat']['train_seconds']),
    ]
    runtime = {
        'baseline_train_seconds': baseline_seconds,
        'deterministic_train_seconds': current_seconds,
        'first_fit_relative_overhead': _relative_overhead(
            current_seconds[0], baseline_seconds[0]
        ),
        'warm_cache_relative_overhead': _relative_overhead(
            current_seconds[1], baseline_seconds[1]
        ),
        'comparison_note': (
            'Baseline timings come from the frozen H-DETERMINISM-2 parent on '
            'the same dataset, horizon, seeds, and frontier instrumentation.'
        ),
    }
    report = {
        'phase': 'deterministic_leaf_reduction',
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
        'runtime': runtime,
        'quality_repeatability': _paired_quality(records),
        'prediction_fingerprints_equal': (
            records['treatment']['prediction_fingerprint']
            == records['repeat']['prediction_fingerprint']
        ),
        'model_structure_fingerprints_equal': (
            records['treatment']['model_structure_fingerprint']
            == records['repeat']['model_structure_fingerprint']
        ),
        'scope_limit': (
            'The leaf-value reduction is deterministic and opt-in. Histogram '
            'construction and split tie handling remain unchanged.'
        ),
    }
    output = Path('results/revision/deterministic_leaf_reduction.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(json.dumps({'DETERMINISTIC_LEAF_REDUCTION': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': report['scientific_status'],
        'primary_gate_passed': analysis['primary_gate_passed'],
        'new_boosting_fits': 2,
        'trees_per_fit': args.ntrees,
        'runtime': runtime,
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
