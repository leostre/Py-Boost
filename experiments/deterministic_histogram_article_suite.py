"""Run one preregistered layout arm across the article dataset suite."""

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
    DATASETS,
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
from experiments.deterministic_histogram_article_suite_core import (
    aggregate_suite_status,
    analyze_layout_dispatch,
)
from experiments.deterministic_histogram_core import analyze_deterministic_histogram
from experiments.reproducibility import array_fingerprint, paired_seed_map
from py_boost.gpu.utils import (
    start_deterministic_histogram_layout_trace,
    stop_deterministic_histogram_layout_trace,
)


_QUALITY_FIELDS = (
    "accuracy",
    "f1_weighted",
    "f1_macro",
    "f1_micro",
    "log_loss",
)


def _layout_records(counts):
    return [
        {
            "policy": policy,
            "selected_layout": selected,
            "nodes": int(nodes),
            "calls": int(calls),
        }
        for (policy, selected, nodes), calls in sorted(counts.items())
    ]


def _benchmark_args(config, protocol, layout):
    previous_argv = sys.argv
    sys.argv = ["benchmark_helena"]
    try:
        args = benchmark.parse_args()
    finally:
        sys.argv = previous_argv
    args.dataset_seed = int(config["dataset_seed"])
    args.ntrees = int(config["ntrees"])
    args.max_depth = int(config["max_depth"])
    args.cache_dir = Path(str(config["cache_dir"]))
    args.output_dir = Path("results/revision")
    args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
    args.trajectory = "none"
    args.datasets = list(protocol["datasets"])
    return args


def _dataset_limit(protocol, name):
    value = protocol["max_samples_by_dataset"][name]
    return None if value is None else int(value)


def _run_dataset(name, config, protocol, layout, args):
    fits = int(protocol["fits"])
    model_seed = int(protocol["model_seed"])
    split_seed = int(protocol["split_seed"])
    max_output_budget = int(protocol["maximum_output_budget"])
    shortlist = int(protocol["shortlist_size"])
    expected_batches = int(protocol["expected_frontier_batches"])
    max_samples = _dataset_limit(protocol, name)

    args.max_samples = max_samples
    dataset = load_benchmark_dataset(
        name,
        max_samples,
        args.dataset_seed,
        args.cache_dir,
    )
    split = split_benchmark_dataset(dataset, split_seed, args.test_size)
    budget = min(max_output_budget, dataset.n_outputs)
    args.output_min_fraction = budget / dataset.n_outputs
    args.split_seed_by_model = paired_seed_map([model_seed], [split_seed])
    args.split_refinement_shortlists = {budget: shortlist}
    defaults = PlanDefaults(
        row_subsample=1.0,
        sketch_outputs=budget,
        min_warmup_rounds=5,
        adaptive_budgets=tuple(dict.fromkeys((budget, dataset.n_outputs))),
    )

    records = {}
    traces = {}
    layout_traces = {}
    dispatch_checks = {}
    for fit_index in range(1, fits + 1):
        arm = f"fit_{fit_index}"
        case = ExperimentCase(
            f"deterministic_histogram_article_suite_{name}_{layout}_{arm}",
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
        if record.get("deterministic_histogram") is not True:
            raise RuntimeError("deterministic histogram reduction was not enabled")
        if record.get("deterministic_histogram_layout") != layout:
            raise RuntimeError("effective deterministic histogram layout drifted")
        records[arm] = record
        traces[arm] = _load_json(record["split_refinement_path"])
        layout_traces[arm] = _layout_records(layout_counts)
        dispatch_checks[arm] = analyze_layout_dispatch(
            layout,
            layout_traces[arm],
        )

    capture_checks = {
        arm: _capture_complete(trace, shortlist, expected_batches)
        for arm, trace in traces.items()
    }
    base_arm = "fit_1"
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
        record["prediction_fingerprint"] for record in records.values()
    }) == 1
    model_structure_fingerprints_equal = len({
        record["model_structure_fingerprint"] for record in records.values()
    }) == 1
    quality_repeatability = {
        field: {arm: records[arm][field] for arm in records}
        for field in _QUALITY_FIELDS
    }
    quality_exact = all(
        len(set(values.values())) == 1
        for values in quality_repeatability.values()
    )
    correctness_gate_passed = (
        all(capture_checks.values())
        and all(check["gate_passed"] for check in dispatch_checks.values())
        and all(
            analysis["primary_gate_passed"]
            and analysis["all_captured_state_exact"]
            for analysis in replay_analysis.values()
        )
        and prediction_fingerprints_equal
        and model_structure_fingerprints_equal
        and quality_exact
    )
    train_seconds = [
        float(records[f"fit_{index}"]["train_seconds"])
        for index in range(1, fits + 1)
    ]
    spec = DATASETS[name]
    source_samples, source_features = dataset.source_shape
    article_discrepancies = {
        "samples": (
            None
            if spec.article_reported_samples is None
            or spec.article_reported_samples == source_samples
            else {
                "article": spec.article_reported_samples,
                "available_source": source_samples,
            }
        ),
        "features": (
            None
            if spec.article_reported_features is None
            or spec.article_reported_features == source_features
            else {
                "article": spec.article_reported_features,
                "model_input": source_features,
            }
        ),
    }
    combined_layout_records = [
        record
        for arm in layout_traces.values()
        for record in arm
    ]
    return {
        "scientific_status": (
            "confirmed" if correctness_gate_passed else "correctness_gate_failed"
        ),
        "dataset": {
            "name": name,
            "task": spec.task.value,
            "source": spec.source,
            "representation": spec.representation,
            "source_samples": source_samples,
            "selected_samples": int(dataset.features.shape[0]),
            "model_features": int(dataset.features.shape[1]),
            "outputs": int(dataset.n_outputs),
            "article_reported_samples": spec.article_reported_samples,
            "article_reported_features": spec.article_reported_features,
            "article_metadata_discrepancies": article_discrepancies,
            "selected_sample_fingerprint": array_fingerprint(
                np.asarray(dataset.sample_indices)
            ),
        },
        "effective_output_budget": budget,
        "capture_checks": capture_checks,
        "dispatch_checks": dispatch_checks,
        "dispatch_coverage": analyze_layout_dispatch(
            layout,
            combined_layout_records,
        ),
        "layout_traces": layout_traces,
        "records": records,
        "replay_analysis": replay_analysis,
        "prediction_fingerprints_equal": prediction_fingerprints_equal,
        "model_structure_fingerprints_equal": model_structure_fingerprints_equal,
        "quality_repeatability": quality_repeatability,
        "quality_exact": quality_exact,
        "correctness_gate_passed": correctness_gate_passed,
        "train_seconds": train_seconds,
        "warm_train_seconds_median": statistics.median(train_seconds[1:]),
    }


def run_deterministic_histogram_article_suite(config):
    """Run the preregistered article suite for one committed layout policy."""

    phase = "deterministic_histogram_article_suite"
    protocol = config[phase]
    layout = str(protocol["histogram_layout"])
    fits = int(protocol["fits"])
    if layout not in {"perf1", "hybrid"}:
        raise ValueError(f"unsupported histogram layout: {layout}")
    if fits < 3:
        raise ValueError("article-suite validation requires at least three fits")
    datasets = tuple(protocol["datasets"])
    if set(datasets) != set(protocol["max_samples_by_dataset"]):
        raise ValueError("every article-suite dataset needs one sample limit")

    args = _benchmark_args(config, protocol, layout)
    dataset_reports = {}
    for name in datasets:
        report = _run_dataset(name, config, protocol, layout, args)
        dataset_reports[name] = report
        print(json.dumps({"DATASET_SUMMARY": {
            "dataset": name,
            "variant": layout,
            "scientific_status": report["scientific_status"],
            "warm_train_seconds_median": report["warm_train_seconds_median"],
            "both_layouts_observed": report["dispatch_coverage"][
                "both_layouts_observed"
            ],
            "article_metadata_discrepancies": report["dataset"][
                "article_metadata_discrepancies"
            ],
        }}, ensure_ascii=False), flush=True)

    suite_status = aggregate_suite_status(dataset_reports)
    report = {
        "phase": phase,
        "variant": layout,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "confirmed"
            if suite_status["correctness_gate_passed"]
            else "correctness_gate_failed"
        ),
        "protocol": protocol,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "gpu": benchmark._gpu_name(),
        },
        "suite_status": suite_status,
        "datasets": dataset_reports,
        "comparison_scope": (
            "Compare per-dataset warm medians and exact traces only with the "
            "sibling arm from this preregistered round."
        ),
    }
    output = Path("results/revision") / f"{phase}_{layout}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"FINAL_SUMMARY": {
        "phase": phase,
        "variant": layout,
        "scientific_status": report["scientific_status"],
        "suite_status": suite_status,
        "warm_train_seconds_medians": {
            name: value["warm_train_seconds_median"]
            for name, value in dataset_reports.items()
        },
        "new_boosting_fits": len(datasets) * fits,
        "trees_per_fit": args.ntrees,
        "report": str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
