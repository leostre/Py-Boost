"""Collect and analyze the preregistered expanded H5-3A replication."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.revision_forecast_replay import file_sha256
from experiments.rmt_residual_expanded_core import (
    nearest_fraction_budget,
    validate_reused_effective_config,
    validate_scalar_grid,
)
from experiments.rmt_residual_signal import _verified_source
from experiments.rmt_residual_signal_core import (
    cross_trajectory_residual_rows,
    summarize_residual_verdict,
)
from experiments.rmt_scalar_reproducibility import _analyze_trajectory


def _legacy_rows(protocol):
    legacy = dict(protocol["legacy_source"])
    verification_protocol = {
        "source_report": legacy["report"],
        "source_report_sha256": legacy["report_sha256"],
        "source_verdict": legacy["verdict"],
        "datasets": legacy["datasets"],
        "model_seeds": legacy["model_seeds"],
        "split_seeds": legacy["split_seeds"],
        "source_endpoints": protocol["endpoints"],
        "partition_repeats": protocol["partition_repeats"],
        "primary_feature": protocol["primary_feature"],
        "secondary_features": protocol["secondary_features"],
    }
    report, provenance = _verified_source(verification_protocol)
    return list(report["rows"]), provenance


def _benchmark_args(config, protocol, run_id):
    from experiments import benchmark_helena as benchmark
    from experiments.reproducibility import paired_seed_map

    previous_argv = sys.argv
    sys.argv = ["benchmark_helena"]
    try:
        args = benchmark.parse_args()
    finally:
        sys.argv = previous_argv
    args.dataset_seed = int(protocol["dataset_seed"])
    args.ntrees = int(protocol["trees"])
    args.max_depth = int(protocol["max_depth"])
    args.cache_dir = Path(config["cache_dir"])
    args.output_dir = Path(protocol["artifact_dir"])
    args.run_id = run_id
    args.split_seed_by_model = paired_seed_map(
        protocol["model_seeds"], protocol["split_seeds"]
    )
    args.record_selection_indices = True
    args.profile_methods = False
    args.predict_repeats = 1
    args.gain_risk_threshold = 0.10
    args.split_top_k = 0
    args.trajectory = "gh"
    args.trajectory_subset_rows = int(protocol["expected_rows"])
    return args


def _artifact_receipt(record):
    paths = {
        name: record.get(name)
        for name in (
            "trajectory_path",
            "split_diagnostics_path",
            "split_indices_path",
            "effective_config_path",
        )
    }
    return {
        "dataset": record["dataset"],
        "model_seed": int(record["model_seed"]),
        "split_seed": int(record["split_seed"]),
        "artifacts": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": file_sha256(Path(path)),
            }
            for name, path in paths.items()
            if path
        },
    }


def _reused_records(protocol):
    reuse = dict(protocol["reuse_source"])
    root = Path(reuse["artifact_root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"reused artifact root does not exist: {root}")
    split_by_model = dict(zip(protocol["model_seeds"], protocol["split_seeds"]))
    records = []
    for dataset in reuse["datasets"]:
        budgets = protocol["budgets"][dataset]
        outputs = int(reuse["outputs_by_dataset"][dataset])
        for raw_seed in protocol["model_seeds"]:
            model_seed = int(raw_seed)
            split_seed = int(split_by_model[model_seed])
            stem = f"{dataset}_adaptive_budget_matched_trajectory_seed{model_seed}"
            paths = {
                "trajectory_path": root / "trajectories" / f"{stem}.npz",
                "split_diagnostics_path": root / "split_diagnostics" / f"{stem}.json",
                "split_indices_path": root / "splits" / f"{stem}.npz",
                "effective_config_path": root / "effective_configs" / f"{stem}.json",
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing reused artifacts: {missing}")
            effective = json.loads(paths["effective_config_path"].read_text(encoding="utf-8"))
            validate_reused_effective_config(
                effective,
                model_seed=model_seed,
                split_seed=split_seed,
                structure_budget=int(budgets["structure"]),
                risk_budget=int(budgets["risk"]),
                outputs=outputs,
                trees=int(protocol["trees"]),
                max_depth=int(protocol["max_depth"]),
                expected_rows=int(protocol["expected_rows"]),
            )
            diagnostics = json.loads(
                paths["split_diagnostics_path"].read_text(encoding="utf-8")
            )
            if not diagnostics:
                raise ValueError("reused split diagnostics are empty")
            if any(
                row.get("dataset") != dataset
                or int(row.get("model_seed", -1)) != model_seed
                or int(row.get("split_seed", -1)) != split_seed
                for row in diagnostics
            ):
                raise ValueError("reused split diagnostics identify another trajectory")
            records.append({
                "dataset": dataset,
                "model_seed": model_seed,
                "split_seed": split_seed,
                "dataset_fingerprint": effective["dataset_fingerprint"],
                **{key: str(path) for key, path in paths.items()},
            })
    if len(records) != int(protocol["reused_boosting_trajectories"]):
        raise ValueError("reused trajectory count differs from the frozen protocol")
    return records


def run_expanded_residual_signal(config):
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

    protocol = dict(config["rmt_residual_signal_expanded"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    legacy_reference_rows, legacy_provenance = _legacy_rows(protocol)
    scalar_rows = []
    args = _benchmark_args(config, protocol, run_id)
    split_by_model = dict(zip(protocol["model_seeds"], protocol["split_seeds"]))
    training_records = []
    reused_records = _reused_records(protocol)
    sources = []
    shapes = {}
    warmup_trainings = 0

    for record in reused_records:
        name = record["dataset"]
        analysis_protocol = dict(protocol)
        analysis_protocol["risk_budget"] = int(protocol["budgets"][name]["risk"])
        source, rows = _analyze_trajectory(record, analysis_protocol, cp)
        source.update(
            structure_budget=int(protocol["budgets"][name]["structure"]),
            risk_budget=int(protocol["budgets"][name]["risk"]),
            dataset_fingerprint=record["dataset_fingerprint"],
            reused=True,
        )
        sources.append(source)
        scalar_rows.extend(rows)
        print(json.dumps({"REUSED_COLLECTION_RESULT": {
            "dataset": name,
            "model_seed": record["model_seed"],
            "trajectory_path": record["trajectory_path"],
            "scalar_rows": len(rows),
        }}), flush=True)

    for name in protocol["collect_datasets"]:
        max_samples = int(protocol["max_samples_by_dataset"][name])
        dataset = load_benchmark_dataset(
            name,
            max_samples=max_samples,
            dataset_seed=args.dataset_seed,
            cache_dir=args.cache_dir,
        )
        outputs = dataset.n_outputs
        structure_budget = nearest_fraction_budget(outputs, 1, 4)
        risk_budget = nearest_fraction_budget(outputs, 1, 2)
        expected_budget = protocol["budgets"][name]
        if expected_budget != {
            "structure": structure_budget,
            "risk": risk_budget,
        }:
            raise ValueError(f"registered budgets differ for {name}")
        shapes[name] = {
            "source_rows": int(dataset.source_shape[0]),
            "source_features": int(dataset.source_shape[1]),
            "selected_rows": int(dataset.features.shape[0]),
            "features": int(dataset.features.shape[1]),
            "outputs": outputs,
            "source": dataset.spec.source,
            "structure_budget": structure_budget,
            "risk_budget": risk_budget,
        }

        defaults = PlanDefaults(
            row_subsample=1.0,
            sketch_outputs=structure_budget,
            min_warmup_rounds=5,
            adaptive_budgets=(risk_budget, outputs),
            scheduler_control_interval=10,
            adaptive_min_dwell_rounds=10,
            adaptive_cooldown_after_increase_rounds=20,
        )
        case = ExperimentCase(
            "matched_trajectory",
            Variant.HASBOOST_CURRENT_TOPK,
            scheduler_mode=SchedulerMode.NEVER,
            row_subsample=1.0,
            sketch_outputs=structure_budget,
            collect_split_diagnostics=True,
        ).resolve(defaults, outputs)
        new_seeds = list(protocol["model_seeds"])
        if not new_seeds:
            continue

        first_seed = int(new_seeds[0])
        first_split = split_benchmark_dataset(
            dataset, int(split_by_model[first_seed]), args.test_size
        )
        args.max_samples = max_samples
        args.output_min_fraction = risk_budget / outputs
        args.research_shadow_budgets = (risk_budget,)
        benchmark.warmup_case(
            case,
            args,
            dataset.spec.task,
            first_seed,
            *first_split,
        )
        warmup_trainings += 1

        for seed in new_seeds:
            seed = int(seed)
            split_seed = int(split_by_model[seed])
            split = (
                first_split
                if seed == first_seed
                else split_benchmark_dataset(dataset, split_seed, args.test_size)
            )
            print(json.dumps({"COLLECTION_BLOCK": {
                "dataset": name,
                "model_seed": seed,
                "split_seed": split_seed,
                "structure_budget": structure_budget,
                "risk_budget": risk_budget,
            }}), flush=True)
            run = PlannedRun(Scenario.ADAPTIVE_BUDGET, case, seed)
            record = benchmark.evaluate_run(run, args, dataset, *split)
            record.update(
                phase="rmt_residual_signal_expanded",
                structure_budget=structure_budget,
                risk_budget=risk_budget,
            )
            if int(record["trees_built"]) != int(protocol["expected_snapshots"]):
                raise ValueError("collected trajectory length differs from protocol")
            analysis_protocol = dict(protocol)
            analysis_protocol["risk_budget"] = risk_budget
            source, rows = _analyze_trajectory(record, analysis_protocol, cp)
            source.update(
                structure_budget=structure_budget,
                risk_budget=risk_budget,
                dataset_fingerprint=record["dataset_fingerprint"],
                reused=False,
            )
            sources.append(source)
            scalar_rows.extend(rows)
            training_records.append(record)
            print(json.dumps({"COLLECTION_RESULT": {
                "dataset": name,
                "model_seed": seed,
                "train_seconds": record["train_seconds"],
                "trajectory_path": record["trajectory_path"],
                "scalar_rows": len(rows),
            }}), flush=True)

    features = [protocol["primary_feature"], *protocol["secondary_features"]]
    if len(training_records) != int(protocol["new_boosting_trainings"]):
        raise ValueError("new boosting fit count differs from the frozen protocol")
    if warmup_trainings != int(protocol["warmup_trainings"]):
        raise ValueError("warm-up fit count differs from the frozen protocol")
    grid = validate_scalar_grid(
        scalar_rows,
        protocol["datasets"],
        protocol["model_seeds"],
        protocol["split_seeds"],
        protocol["endpoints"],
        features,
    )
    residual_rows = cross_trajectory_residual_rows(scalar_rows)
    decision = summarize_residual_verdict(
        residual_rows,
        primary_feature=protocol["primary_feature"],
        secondary_features=protocol["secondary_features"],
        minimum_ccc=protocol["minimum_ccc"],
        minimum_sign_agreement=protocol["minimum_sign_agreement"],
        minimum_risk_rho=protocol["minimum_risk_rho"],
    )
    new_receipts = [_artifact_receipt(record) for record in training_records]
    reused_receipts = [_artifact_receipt(record) for record in reused_records]
    new_logical_decompositions = (
        len(training_records)
        * len(protocol["endpoints"])
        * (2 * int(protocol["partition_repeats"]) + 1)
    )
    if new_logical_decompositions != int(
        protocol["new_logical_spectral_decompositions"]
    ):
        raise ValueError("new spectral decomposition count differs from protocol")
    total_logical_decompositions = (
        (len(training_records) + len(reused_records))
        * len(protocol["endpoints"])
        * (2 * int(protocol["partition_repeats"]) + 1)
    )
    if total_logical_decompositions != int(
        protocol["total_logical_spectral_decompositions"]
    ):
        raise ValueError("total spectral decomposition count differs from protocol")
    report = {
        "hypothesis": "H5-3A-X2 resource-adjusted spectral residual replication",
        "protocol": protocol,
        "provenance": {
            "legacy_reference": {
                **legacy_provenance,
                "reference_rows": len(legacy_reference_rows),
                "used_as_expanded_input": False,
            },
            "dataset_shapes": shapes,
            "reused_source": {
                "run_id": protocol["reuse_source"]["run_id"],
                "experiment_id": protocol["reuse_source"]["experiment_id"],
                "source_commit": protocol["reuse_source"]["source_commit"],
                "source_run_status": "cancelled_after_resource_stop",
                "completed_artifacts_used_for_verdict": True,
                "incomplete_moa_artifacts_used": False,
                "verified_artifacts": reused_receipts,
            },
            "new_artifacts": new_receipts,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "gpu": benchmark._gpu_name(),
        },
        "grid": grid,
        "new_sources": sources,
        "training_records": training_records,
        "scalar_rows": scalar_rows,
        "residual_rows": residual_rows,
        "decision": decision,
        "new_boosting_trainings": len(training_records),
        "reused_boosting_trajectories": len(reused_records),
        "warmup_trainings": warmup_trainings,
        "new_logical_spectral_decompositions": new_logical_decompositions,
        "total_logical_spectral_decompositions": total_logical_decompositions,
        "interpretation_scope": (
            "Offline reproducibility and same-endpoint association after the frozen "
            "leave-one-trajectory-out transform; not a causal controller claim."
        ),
    }
    output_path = Path(protocol["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"FINAL_SUMMARY": {
        "phase": config["phase"],
        "report": str(output_path.resolve()),
        "report_sha256": file_sha256(output_path),
        "trajectories": grid["trajectories"],
        "new_boosting_trainings": len(training_records),
        "reused_boosting_trajectories": len(reused_records),
        "warmup_trainings": warmup_trainings,
        "new_logical_spectral_decompositions": new_logical_decompositions,
        "total_logical_spectral_decompositions": total_logical_decompositions,
        "decision": decision,
    }}), flush=True)
