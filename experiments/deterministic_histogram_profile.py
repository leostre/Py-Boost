"""Run the preregistered H-HIST-PROFILE-1 diagnostic experiment."""

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
from experiments.deterministic_histogram_profile_core import (
    analyze_histogram_profile,
)
from experiments.reproducibility import paired_seed_map
from py_boost.gpu.utils import (
    start_deterministic_histogram_profile,
    stop_deterministic_histogram_profile,
)


def _load_json(path):
    return json.loads(Path(str(path)).read_text(encoding='utf-8'))


def _capture_complete(trace, expected_shortlist, expected_batches):
    return len(trace) == expected_batches and all(
        'state_fingerprints' in entry
        and all(
            len(node.get('candidate_frontier', ())) == expected_shortlist
            for node in entry['nodes']
        )
        for entry in trace
    )


def run_deterministic_histogram_profile(config: dict[str, object]) -> dict[str, object]:
    """Profile the confirmed deterministic histogram on one fixed trajectory."""

    protocol = config['deterministic_histogram_profile']
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
    args.deterministic_histogram = True
    model_seed = int(protocol['model_seed'])
    split_seed = int(protocol['split_seed'])
    budget = int(protocol['output_budget'])
    shortlist = int(protocol['shortlist_size'])
    expected_batches = int(protocol['expected_frontier_batches'])
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
    case = ExperimentCase(
        'deterministic_histogram_profile',
        Variant.HASBOOST_CURRENT_TOPK,
        scheduler_mode=SchedulerMode.NEVER,
        row_subsample=1.0,
        sketch_outputs=budget,
        collect_split_diagnostics=True,
        full_leaf_values=True,
    ).resolve(defaults, dataset.n_outputs)
    run = PlannedRun(Scenario.OBSERVABILITY, case, model_seed)

    start_deterministic_histogram_profile()
    try:
        record = benchmark.evaluate_run(run, args, dataset, *split)
    finally:
        profile_records = stop_deterministic_histogram_profile()

    if record.get('deterministic_leaf_values') is not True:
        raise RuntimeError('deterministic leaf-value reduction was not enabled')
    if record.get('deterministic_histogram') is not True:
        raise RuntimeError('deterministic histogram reduction was not enabled')

    trace = _load_json(record['split_refinement_path'])
    capture_complete = _capture_complete(
        trace,
        shortlist,
        expected_batches,
    )
    profile_analysis = analyze_histogram_profile(
        profile_records,
        train_seconds=float(record['train_seconds']),
        expected_nodes=tuple(protocol['expected_nodes']),
        dominance_threshold=float(protocol['dominance_threshold']),
    )
    expected_quality = protocol['parent_quality']
    quality_equal = all(
        record[field] == expected_quality[field]
        for field in expected_quality
    )
    prediction_equal = (
        record['prediction_fingerprint']
        == protocol['parent_prediction_fingerprint']
    )
    structure_equal = (
        record['model_structure_fingerprint']
        == protocol['parent_model_structure_fingerprint']
    )
    correctness_gate_passed = (
        capture_complete
        and quality_equal
        and prediction_equal
        and structure_equal
    )
    coverage_gate_passed = bool(profile_analysis['coverage_gate_passed'])
    if not correctness_gate_passed:
        scientific_status = 'correctness_gate_failed'
    elif not coverage_gate_passed:
        scientific_status = 'coverage_gate_failed'
    else:
        scientific_status = 'confirmed'

    report = {
        'phase': 'deterministic_histogram_profile',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': scientific_status,
        'protocol': protocol,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': benchmark._gpu_name(),
        },
        'correctness_gate_passed': correctness_gate_passed,
        'coverage_gate_passed': coverage_gate_passed,
        'capture_complete': capture_complete,
        'prediction_fingerprint_equal_to_parent': prediction_equal,
        'model_structure_fingerprint_equal_to_parent': structure_equal,
        'quality_equal_to_parent': quality_equal,
        'record': record,
        'profile_analysis': profile_analysis,
        'profile_records': profile_records,
        'scope_limit': (
            'Synchronized instrumentation changes runtime and is used only to '
            'attribute cost within the unchanged deterministic histogram path.'
        ),
    }
    output = Path('results/revision/deterministic_histogram_profile.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(json.dumps({'DETERMINISTIC_HISTOGRAM_PROFILE': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': scientific_status,
        'correctness_gate_passed': correctness_gate_passed,
        'coverage_gate_passed': coverage_gate_passed,
        'record_count': profile_analysis['record_count'],
        'profiled_histogram_fraction_of_training': (
            profile_analysis['profiled_histogram_fraction_of_training']
        ),
        'category_shares_of_total': profile_analysis['category_shares_of_total'],
        'dominant_component': profile_analysis['dominant_component'],
        'next_action': profile_analysis['next_action'],
        'new_boosting_fits': 1,
        'trees_per_fit': args.ntrees,
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
