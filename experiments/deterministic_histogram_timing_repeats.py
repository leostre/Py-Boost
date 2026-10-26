"""Balanced paired timing repeats for frozen deterministic histogram layouts."""

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
from experiments.deterministic_histogram_article_suite_core import analyze_layout_dispatch
from experiments.deterministic_histogram_timing_repeats_core import analyze_paired_runtimes
from experiments.reproducibility import paired_seed_map
from py_boost.gpu.utils import (
    start_deterministic_histogram_layout_trace,
    stop_deterministic_histogram_layout_trace,
)


_QUALITY_FIELDS = ("accuracy", "f1_weighted", "f1_macro", "f1_micro", "log_loss")


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


def _benchmark_args(config, protocol):
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
    args.record_selection_indices = False
    args.profile_methods = False
    args.predict_repeats = 1
    args.gain_risk_threshold = 0.10
    args.split_top_k = 2
    args.split_output_attribution = False
    args.split_candidate_output_profiles = False
    args.split_refinement_shadow = True
    args.split_refinement_apply = True
    args.split_refinement_determinism_capture = False
    args.research_shadow_budgets = ()
    args.trajectory = "none"
    args.datasets = list(protocol["datasets"])
    return args


def _run_one(layout, pair_index, dataset, split, args, defaults, model_seed):
    args.deterministic_histogram_layout = layout
    case = ExperimentCase(
        f"timing_{dataset.spec.name}_{layout}_pair_{pair_index:02d}",
        Variant.HASBOOST_CURRENT_TOPK,
        scheduler_mode=SchedulerMode.NEVER,
        row_subsample=1.0,
        sketch_outputs=defaults.sketch_outputs,
        collect_split_diagnostics=True,
        full_leaf_values=True,
    ).resolve(defaults, dataset.n_outputs)
    run = PlannedRun(Scenario.OBSERVABILITY, case, model_seed)
    start_deterministic_histogram_layout_trace()
    try:
        record = benchmark.evaluate_run(run, args, dataset, *split)
    finally:
        layout_counts = stop_deterministic_histogram_layout_trace()
    dispatch_records = _layout_records(layout_counts)
    dispatch = analyze_layout_dispatch(layout, dispatch_records)
    return record, dispatch_records, dispatch


def _pair_exact(perf1, hybrid):
    quality_exact = all(perf1[field] == hybrid[field] for field in _QUALITY_FIELDS)
    return {
        "prediction_fingerprint_exact": (
            perf1["prediction_fingerprint"] == hybrid["prediction_fingerprint"]
        ),
        "model_structure_fingerprint_exact": (
            perf1["model_structure_fingerprint"]
            == hybrid["model_structure_fingerprint"]
        ),
        "quality_exact": quality_exact,
        "gate_passed": (
            quality_exact
            and perf1["prediction_fingerprint"] == hybrid["prediction_fingerprint"]
            and perf1["model_structure_fingerprint"]
            == hybrid["model_structure_fingerprint"]
        ),
    }


def _run_dataset(name, config, protocol, args):
    model_seed = int(protocol["model_seed"])
    split_seed = int(protocol["split_seed"])
    warmup_pairs = int(protocol["warmup_pairs"])
    measured_pairs = int(protocol["measured_pairs"])
    dataset = load_benchmark_dataset(name, None, args.dataset_seed, args.cache_dir)
    split = split_benchmark_dataset(dataset, split_seed, args.test_size)
    budget = min(int(protocol["maximum_output_budget"]), dataset.n_outputs)
    args.output_min_fraction = budget / dataset.n_outputs
    args.split_seed_by_model = paired_seed_map([model_seed], [split_seed])
    args.split_refinement_shortlists = {budget: int(protocol["shortlist_size"])}
    defaults = PlanDefaults(
        row_subsample=1.0,
        sketch_outputs=budget,
        min_warmup_rounds=5,
        adaptive_budgets=tuple(dict.fromkeys((budget, dataset.n_outputs))),
    )

    pairs = []
    for pair_index in range(warmup_pairs + measured_pairs):
        order = ("perf1", "hybrid") if pair_index % 2 == 0 else ("hybrid", "perf1")
        records = {}
        dispatch_records = {}
        dispatch_checks = {}
        for layout in order:
            record, trace, check = _run_one(
                layout, pair_index, dataset, split, args, defaults, model_seed
            )
            records[layout] = record
            dispatch_records[layout] = trace
            dispatch_checks[layout] = check
        exactness = _pair_exact(records["perf1"], records["hybrid"])
        pair = {
            "pair_index": pair_index,
            "measured": pair_index >= warmup_pairs,
            "execution_order": list(order),
            "records": records,
            "dispatch_records": dispatch_records,
            "dispatch_checks": dispatch_checks,
            "exactness": exactness,
        }
        pairs.append(pair)
        print(json.dumps({"PAIR_SUMMARY": {
            "dataset": name,
            "pair_index": pair_index,
            "measured": pair["measured"],
            "execution_order": pair["execution_order"],
            "perf1_seconds": records["perf1"]["train_seconds"],
            "hybrid_seconds": records["hybrid"]["train_seconds"],
            "exact": exactness["gate_passed"],
        }}, ensure_ascii=False), flush=True)

    measured = [pair for pair in pairs if pair["measured"]]
    timing = analyze_paired_runtimes(
        [pair["records"]["perf1"]["train_seconds"] for pair in measured],
        [pair["records"]["hybrid"]["train_seconds"] for pair in measured],
        maximum_regression_fraction=float(protocol["maximum_regression_fraction"]),
        bootstrap_samples=int(protocol["bootstrap_samples"]),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
    )
    correctness_gate = all(
        pair["exactness"]["gate_passed"]
        and pair["dispatch_checks"]["perf1"]["gate_passed"]
        and pair["dispatch_checks"]["hybrid"]["gate_passed"]
        for pair in pairs
    )
    hybrid_exercised_both = all(
        pair["dispatch_checks"]["hybrid"]["both_layouts_observed"]
        for pair in pairs
    )
    return {
        "dataset": {
            "name": name,
            "samples": int(dataset.features.shape[0]),
            "features": int(dataset.features.shape[1]),
            "outputs": int(dataset.n_outputs),
            "source": dataset.spec.source,
        },
        "effective_output_budget": budget,
        "pairs": pairs,
        "timing_analysis": timing,
        "correctness_gate_passed": correctness_gate,
        "hybrid_exercised_both_layouts": hybrid_exercised_both,
        "dataset_gate_passed": (
            correctness_gate and hybrid_exercised_both and timing["timing_gate_passed"]
        ),
    }


def run_deterministic_histogram_timing_repeats(config):
    phase = "deterministic_histogram_timing_repeats"
    protocol = config[phase]
    if int(protocol["warmup_pairs"]) < 1:
        raise ValueError("at least one warm-up pair is required")
    if int(protocol["measured_pairs"]) < 10:
        raise ValueError("at least ten measured pairs are required")
    args = _benchmark_args(config, protocol)
    reports = {
        name: _run_dataset(name, config, protocol, args)
        for name in protocol["datasets"]
    }
    all_gates_passed = all(report["dataset_gate_passed"] for report in reports.values())
    report = {
        "phase": phase,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "confirmed" if all_gates_passed else "timing_gate_failed",
        "protocol": protocol,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "gpu": benchmark._gpu_name(),
        },
        "datasets": reports,
        "all_gates_passed": all_gates_passed,
    }
    output_path = Path(protocol["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"FINAL_SUMMARY": {
        "phase": phase,
        "scientific_status": report["scientific_status"],
        "all_gates_passed": all_gates_passed,
        "datasets": {
            name: {
                "correctness_gate_passed": value["correctness_gate_passed"],
                "hybrid_exercised_both_layouts": value["hybrid_exercised_both_layouts"],
                "timing_analysis": value["timing_analysis"],
                "dataset_gate_passed": value["dataset_gate_passed"],
            }
            for name, value in reports.items()
        },
        "output_path": str(output_path.resolve()),
    }}, ensure_ascii=False), flush=True)
    if not all_gates_passed:
        raise SystemExit(2)
