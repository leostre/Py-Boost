"""Run one preregistered Helena arm of H-HIST-XDATA-1."""

from __future__ import annotations

import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

from experiments import benchmark_helena as benchmark
from experiments.benchmark_datasets import (
    load_benchmark_dataset,
    split_benchmark_dataset,
)
from experiments.benchmark_plan import (
    ExperimentCase,
    PlanDefaults,
    PlannedRun,
    Scenario,
    SchedulerMode,
    Variant,
)
from experiments.deterministic_histogram import _capture_complete, _load_json
from experiments.deterministic_histogram_core import analyze_deterministic_histogram
from experiments.reproducibility import paired_seed_map
from py_boost.gpu.utils import (
    start_deterministic_histogram_layout_trace,
    stop_deterministic_histogram_layout_trace,
)


_QUALITY_FIELDS = (
    'accuracy',
    'f1_weighted',
    'f1_macro',
    'f1_micro',
    'log_loss',
)


def _layout_records(counts):
    return [
        {
            'policy': policy,
            'selected_layout': selected,
            'nodes': int(nodes),
            'calls': int(calls),
        }
        for (policy, selected, nodes), calls in sorted(counts.items())
    ]


def _dispatch_gate(layout, records):
    if not records:
        return False
    if any(record['policy'] != layout for record in records):
        return False
    if layout == 'perf1':
        return all(record['selected_layout'] == 'perf1' for record in records)
    selected = {record['selected_layout'] for record in records}
    return (
        selected == {'perf1', 'perf2'}
        and all(
            record['selected_layout'] == ('perf1' if record['nodes'] <= 8 else 'perf2')
            for record in records
        )
    )


def run_deterministic_histogram_cross_dataset(config):
    """Run three exact Helena fits for one committed layout policy."""

    phase = 'deterministic_histogram_cross_dataset'
    protocol = config[phase]
    layout = str(protocol['histogram_layout'])
    fits = int(protocol['fits'])
    if layout not in {'perf1', 'hybrid'}:
        raise ValueError(f'unsupported histogram layout: {layout}')
    if fits < 3:
        raise ValueError('cross-dataset validation requires at least three fits')

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
    args.deterministic_histogram_layout = layout
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
    args.research_shadow_budgets = ()
    args.trajectory = 'none'

    model_seed = int(protocol['model_seed'])
    split_seed = int(protocol['split_seed'])
    budget = int(protocol['output_budget'])
    shortlist = int(protocol['shortlist_size'])
    expected_batches = int(protocol['expected_frontier_batches'])
    args.split_seed_by_model = paired_seed_map([model_seed], [split_seed])
    args.split_refinement_shortlists = {budget: shortlist}

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
    layout_traces = {}
    for fit_index in range(1, fits + 1):
        arm = f'fit_{fit_index}'
        case = ExperimentCase(
            f'{phase}_{layout}_{arm}',
            Variant.HASBOOST_CURRENT_TOPK,
            scheduler_mode=SchedulerMode.NEVER,
            row_subsample=1.0,
            sketch_outputs=budget,
            collect_split_diagnostics=True,
            full_leaf_values=True,
        ).resolve(defaults, dataset.n_outputs)
        run = PlannedRun(Scenario.OBSERVABILITY, case, model_seed)
        start_deterministic_histogram_layout_trace()
        try:
            record = benchmark.evaluate_run(run, args, dataset, *split)
        finally:
            layout_counts = stop_deterministic_histogram_layout_trace()
        if record.get('deterministic_histogram') is not True:
            raise RuntimeError('deterministic histogram reduction was not enabled')
        if record.get('deterministic_histogram_layout') != layout:
            raise RuntimeError('effective deterministic histogram layout drifted')
        records[arm] = record
        traces[arm] = _load_json(record['split_refinement_path'])
        layout_traces[arm] = _layout_records(layout_counts)

    capture_checks = {
        arm: _capture_complete(trace, shortlist, expected_batches)
        for arm, trace in traces.items()
    }
    dispatch_checks = {
        arm: _dispatch_gate(layout, layout_traces[arm])
        for arm in records
    }
    base_arm = 'fit_1'
    replay_analysis = {
        arm: analyze_deterministic_histogram(
            traces[base_arm],
            traces[arm],
            expected_batches=expected_batches,
        )
        for arm in records
        if arm != base_arm
    }
    prediction_fingerprints_equal = len({
        record['prediction_fingerprint'] for record in records.values()
    }) == 1
    model_structure_fingerprints_equal = len({
        record['model_structure_fingerprint'] for record in records.values()
    }) == 1
    quality_repeatability = {
        field: {
            arm: records[arm][field] for arm in records
        }
        for field in _QUALITY_FIELDS
    }
    quality_exact = all(
        len(set(values.values())) == 1
        for values in quality_repeatability.values()
    )
    correctness_gate_passed = (
        all(capture_checks.values())
        and all(dispatch_checks.values())
        and all(
            analysis['primary_gate_passed']
            and analysis['all_captured_state_exact']
            for analysis in replay_analysis.values()
        )
        and prediction_fingerprints_equal
        and model_structure_fingerprints_equal
        and quality_exact
    )
    train_seconds = [
        float(records[f'fit_{index}']['train_seconds'])
        for index in range(1, fits + 1)
    ]
    report = {
        'phase': phase,
        'variant': layout,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': (
            'confirmed' if correctness_gate_passed else 'correctness_gate_failed'
        ),
        'protocol': protocol,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': benchmark._gpu_name(),
        },
        'dataset': {
            'name': dataset.spec.name,
            'task': dataset.spec.task.value,
            'samples': int(dataset.features.shape[0]),
            'features': int(dataset.features.shape[1]),
            'outputs': int(dataset.n_outputs),
        },
        'capture_checks': capture_checks,
        'dispatch_checks': dispatch_checks,
        'layout_traces': layout_traces,
        'records': records,
        'replay_analysis': replay_analysis,
        'prediction_fingerprints_equal': prediction_fingerprints_equal,
        'model_structure_fingerprints_equal': model_structure_fingerprints_equal,
        'quality_repeatability': quality_repeatability,
        'quality_exact': quality_exact,
        'correctness_gate_passed': correctness_gate_passed,
        'train_seconds': train_seconds,
        'warm_train_seconds_median': statistics.median(train_seconds[1:]),
        'comparison_scope': (
            'Compare this warm median only with the sibling Helena arm from '
            'the same preregistered round.'
        ),
    }
    output = Path('results/revision') / f'{phase}_{layout}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(json.dumps({phase.upper(): report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': phase,
        'variant': layout,
        'scientific_status': report['scientific_status'],
        'correctness_gate_passed': correctness_gate_passed,
        'capture_checks': capture_checks,
        'dispatch_checks': dispatch_checks,
        'prediction_fingerprints_equal': prediction_fingerprints_equal,
        'model_structure_fingerprints_equal': model_structure_fingerprints_equal,
        'quality_exact': quality_exact,
        'train_seconds': train_seconds,
        'warm_train_seconds_median': report['warm_train_seconds_median'],
        'new_boosting_fits': fits,
        'trees_per_fit': args.ntrees,
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
