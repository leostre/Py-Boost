"""Reproducible HASBoost benchmark scenarios on compact reference datasets."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cupy as cp
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from experiments.benchmark_datasets import (
    DATASETS,
    LoadedDataset,
    Task,
    load_benchmark_dataset,
    split_benchmark_dataset,
    split_benchmark_indices,
)
from experiments.benchmark_plan import (
    PlanDefaults,
    PlannedRun,
    ResolvedCase,
    Scenario,
    SchedulerMode,
    Variant,
    build_run_plan,
    cases_for_scenario,
)
from experiments.core.model_timing import time_patch_methods
from experiments.reproducibility import array_fingerprint, paired_seed_map, source_fingerprint
from experiments.diagnostic_results import (
    add_grouped_baseline_ratios,
    aggregate_records,
    summarize_history,
)
from py_boost import GradientBoosting, SketchBoost
from py_boost.callbacks.optimization import OptimizationTrajectoryRecorder
from py_boost.gpu.accumulation.history_callback import (
    CurrentGradientSampling,
    CurrentGradientEnergySampling,
    EmaGradientEnergySampling,
    GradHessHistory,
    SpectralHistorySampling,
    WeightedHistorySampling,
)
from py_boost.gpu.history_boosting import HistoryBasedBoostingModel


OPENML_DATA_ID = DATASETS["helena"].data_id
VARIANTS = tuple(variant.value for variant in Variant)


def load_helena(max_samples: int = 12_000, seed: int = 42):
    """Backward-compatible Helena loader used by older diagnostic scripts."""
    dataset = load_benchmark_dataset(
        "helena",
        max_samples=max_samples,
        dataset_seed=seed,
    )
    return dataset.features, dataset.target


def _history_factory(
    weight_transform: str,
    spectral: bool = False,
    current_gradient: bool = False,
    energy_budget: bool = False,
    ema_energy: bool = False,
):
    def create_history_sketch(**kwargs):
        sketch_class = (
            EmaGradientEnergySampling
            if ema_energy
            else CurrentGradientEnergySampling
            if energy_budget
            else SpectralHistorySampling
            if spectral
            else CurrentGradientSampling
            if current_gradient
            else WeightedHistorySampling
        )
        return sketch_class(weight_transform=weight_transform, **kwargs)

    return create_history_sketch


def _split_diagnostics_enabled(case: ResolvedCase, args: argparse.Namespace) -> bool:
    return bool(
        args.split_diagnostics
        or case.scheduler_mode is SchedulerMode.ADAPTIVE_BUDGET
        or (case.collect_split_diagnostics and not args.disable_case_diagnostics)
    )


def build_model(
    case: ResolvedCase,
    args: argparse.Namespace,
    task: Task,
    seed: int,
):
    """Build one model while keeping shared tree parameters aligned."""
    trajectory_recorder = None
    if args.trajectory != "none":
        recorder_class = OptimizationTrajectoryRecorder
        extra = {}
        if getattr(args, 'trajectory_subset_rows', None):
            from experiments.revision_forecast import SubsetTrajectoryRecorder
            recorder_class = SubsetTrajectoryRecorder
            extra = {'subset_rows': args.trajectory_subset_rows, 'subset_seed': seed + 1000000}
        trajectory_recorder = recorder_class(
            representation=args.trajectory,
            window=args.trajectory_window or None,
            regularization=args.trajectory_lambda,
            every=args.trajectory_every,
            **extra,
        )
    common = {
        "loss": "crossentropy" if task is Task.MULTICLASS else "multilabel",
        "ntrees": args.ntrees,
        "lr": args.learning_rate,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "es": 0,
        "seed": seed,
        "verbose": max(1, args.verbose),
        "use_hess": True,
        "deterministic_leaf_values": bool(
            getattr(args, "deterministic_leaf_values", False)
        ),
        "deterministic_histogram": bool(
            getattr(args, "deterministic_histogram", False)
        ),
        "deterministic_histogram_layout": str(
            getattr(args, "deterministic_histogram_layout", "hybrid")
        ),
        "callbacks": [] if trajectory_recorder is None else [trajectory_recorder],
    }
    if case.variant is Variant.FULL:
        model = GradientBoosting(
            subsample=1.0,
            full_leaf_values=case.full_leaf_values,
            **common,
        )
    elif case.variant is Variant.SKETCHBOOST_TOPK:
        model = SketchBoost(
            sketch_outputs=case.sketch_outputs,
            sketch_method="topk",
            subsample=1.0,
            **common,
        )
    else:
        transform_and_method = {
            Variant.HASBOOST_SIGMOID_RAND: ("sigmoidal", "rand"),
            Variant.HASBOOST_HYPERBOLIC_RAND: ("hyperbolic", "rand"),
            Variant.HASBOOST_SIGMOID_TOPK: ("sigmoidal", "topk"),
            Variant.HASBOOST_SIGMOID_UNIFORM: ("sigmoidal", "uniform"),
            Variant.HASBOOST_CURRENT_TOPK: ("sigmoidal", "topk"),
            Variant.HASBOOST_ENERGY_BUDGET: ("sigmoidal", "topk"),
            Variant.HASBOOST_EMA_ENERGY: ("sigmoidal", "topk"),
            Variant.HASBOOST_SPECTRAL_TOPK: ("sigmoidal", "topk"),
            Variant.HASBOOST_SPECTRAL_DIVERSE: ("sigmoidal", "topk"),
        }
        try:
            weight_transform, sketch_method = transform_and_method[case.variant]
        except KeyError as exc:
            raise ValueError(f"Unsupported model variant: {case.variant.value}") from exc

        model = HistoryBasedBoostingModel(
            multioutput_sketch=_history_factory(
                weight_transform,
                spectral=case.variant in {
                    Variant.HASBOOST_SPECTRAL_TOPK,
                    Variant.HASBOOST_SPECTRAL_DIVERSE,
                },
                current_gradient=case.variant is Variant.HASBOOST_CURRENT_TOPK,
                energy_budget=case.variant is Variant.HASBOOST_ENERGY_BUDGET,
                ema_energy=case.variant is Variant.HASBOOST_EMA_ENERGY,
            ),
            sketch_method=sketch_method,
            subsample=case.row_subsample,
            sketch_outputs=case.sketch_outputs,
            smoothing_alpha=case.smoothing_alpha,
            stabilization_threshold=case.stabilization_threshold,
            destabilization_threshold=case.destabilization_threshold,
            scheduler_mode=case.scheduler_mode.value,
            scheduler_window=case.scheduler_window,
            scheduler_min_observations=case.scheduler_min_observations,
            scheduler_patience_on=case.scheduler_patience_on,
            scheduler_patience_off=case.scheduler_patience_off,
            scheduler_robust_k=case.scheduler_robust_k,
            scheduler_cooldown_rounds=case.scheduler_cooldown_rounds,
            split_diagnostics=_split_diagnostics_enabled(case, args),
            split_diagnostics_top_k=args.split_top_k,
            split_output_attribution=args.split_output_attribution,
            split_diagnostics_control_interval=case.scheduler_control_interval,
            tau=args.tau,
            spectral_window=args.spectral_window,
            spectral_null_permutations=args.spectral_null_permutations,
            spectral_null_rows=args.spectral_null_rows,
            spectral_null_interval=args.spectral_null_interval,
            spectral_beta=args.spectral_beta,
            spectral_tau=args.spectral_tau,
            spectral_diversity_weight=case.spectral_diversity_weight,
            adaptive_budgets=case.adaptive_budgets,
            adaptive_safe_threshold=case.adaptive_safe_threshold,
            adaptive_unsafe_threshold=case.adaptive_unsafe_threshold,
            adaptive_patience_down=case.adaptive_patience_down,
            adaptive_min_dwell_rounds=case.adaptive_min_dwell_rounds,
            adaptive_cooldown_after_increase_rounds=(
                case.adaptive_cooldown_after_increase_rounds
            ),
            min_warmup_rounds=case.min_warmup_rounds,
            record_selection_indices=getattr(args, 'record_selection_indices', False),
            output_energy_fraction=getattr(args, 'output_energy_fraction', 0.90),
            output_min_fraction=getattr(args, 'output_min_fraction', 0.50),
            full_leaf_values=case.full_leaf_values,
            **common,
        )
    if getattr(args, 'research_shadow_budgets', None):
        model.params['multioutput_sketch'].research_shadow_budgets = tuple(args.research_shadow_budgets)
    if getattr(args, 'split_candidate_output_profiles', False):
        model.params['multioutput_sketch'].split_candidate_output_profiles = True
    if getattr(args, 'split_refinement_shadow', False):
        sketch = model.params['multioutput_sketch']
        sketch.split_refinement_shadow = True
        sketch.split_refinement_apply = bool(
            getattr(args, 'split_refinement_apply', False)
        )
        sketch.split_refinement_determinism_capture = bool(
            getattr(args, 'split_refinement_determinism_capture', False)
        )
        sketch.split_refinement_shortlists = {
            int(budget): int(size)
            for budget, size in args.split_refinement_shortlists.items()
        }
        sketch.split_refinement_failure_diagnostics = bool(
            getattr(args, 'split_refinement_failure_diagnostics', False)
        )
        sketch.split_refinement_severe_error_threshold = float(
            getattr(args, 'split_refinement_severe_error_threshold', 1.0)
        )
    model.optimization_trajectory_recorder = trajectory_recorder
    return model


def _synchronize() -> None:
    cp.cuda.get_current_stream().synchronize()


def _cleanup_gpu() -> None:
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


class _DeviceMemorySampler:
    """Sample process-adjacent GPU use while a measured fit is running."""

    def __init__(self, interval_seconds: float = 0.01):
        self.interval_seconds = interval_seconds
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        self.total_bytes = int(total_bytes)
        self.baseline_used_bytes = self.total_bytes - int(free_bytes)
        self.peak_used_bytes = self.baseline_used_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            used_bytes = self.total_bytes - int(free_bytes)
            self.peak_used_bytes = max(self.peak_used_bytes, used_bytes)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        self.peak_used_bytes = max(
            self.peak_used_bytes,
            self.total_bytes - int(free_bytes),
        )


def warmup_case(
    case: ResolvedCase,
    args: argparse.Namespace,
    task: Task,
    seed: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Compile variant-specific GPU kernels outside the measured fit."""
    warmup_args = argparse.Namespace(**vars(args))
    warmup_args.ntrees = max(2, args.min_warmup_rounds + 1)
    warmup_args.verbose = 10_000
    warmup_args.split_diagnostics = False
    warmup_args.split_output_attribution = False
    warmup_args.disable_case_diagnostics = True
    warmup_args.trajectory = "none"
    warmup_args.spectral_window = 2
    warmup_args.spectral_null_permutations = 1
    warmup_args.spectral_null_rows = min(256, args.spectral_null_rows)
    warmup_args.spectral_null_interval = 1
    warmup_case_config = case
    if case.variant.value.startswith("hasboost"):
        warmup_case_config = replace(
            case,
            scheduler_mode=SchedulerMode.FIXED_WARMUP,
            min_warmup_rounds=2,
            collect_split_diagnostics=False,
        )
    model = build_model(warmup_case_config, warmup_args, task, seed)
    model.fit(X_train, y_train, eval_sets=[{"X": X_test, "y": y_test}])
    _synchronize()
    del model
    _cleanup_gpu()


def _timing_fields(timing_data: Dict[str, dict]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for method_name in ("get_weights", "get_indexers"):
        values = timing_data.get(method_name, {})
        total = values.get("total_time")
        calls = int(values.get("call_count", 0))
        fields[f"{method_name}_total_ms"] = total
        fields[f"{method_name}_calls"] = calls
        fields[f"{method_name}_mean_ms"] = total / calls if calls else None
    return fields


def _quality_metrics(
    task: Task,
    target: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    if task is Task.MULTICLASS:
        predictions = probabilities.argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(target, predictions)),
            "f1_weighted": float(f1_score(target, predictions, average="weighted")),
            "f1_macro": float(f1_score(target, predictions, average="macro")),
            "f1_micro": float(f1_score(target, predictions, average="micro")),
            "log_loss": float(
                log_loss(
                    target,
                    probabilities,
                    labels=np.arange(probabilities.shape[1]),
                )
            ),
        }

    predictions = (probabilities >= 0.5).astype(np.uint8)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    binary_crossentropy = -np.mean(
        target * np.log(clipped) + (1 - target) * np.log(1 - clipped)
    )
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "f1_weighted": float(
            f1_score(target, predictions, average="weighted", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(target, predictions, average="macro", zero_division=0)
        ),
        "f1_micro": float(
            f1_score(target, predictions, average="micro", zero_division=0)
        ),
        "log_loss": float(binary_crossentropy),
    }


def _model_structure_fingerprint(model) -> str:
    """Fingerprint the compact inference structure retained after reformatting."""

    structure_arrays = []
    for tree in model.models:
        for attribute in ('test_format', 'test_format_offsets'):
            values = getattr(tree, attribute)
            if values is None:
                raise RuntimeError(
                    f'cannot fingerprint tree structure: {attribute} is unavailable'
                )
            if isinstance(values, cp.ndarray):
                values = values.get()
            structure_arrays.append(np.asarray(values))
    return array_fingerprint(*structure_arrays)


def evaluate_run(
    run: PlannedRun,
    args: argparse.Namespace,
    dataset: LoadedDataset,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Fit, profile, and evaluate one fully resolved planned run."""
    _cleanup_gpu()
    device_pool = cp.get_default_memory_pool()
    pinned_pool = cp.get_default_pinned_memory_pool()
    pool_total_before = int(device_pool.total_bytes())
    pool_used_before = int(device_pool.used_bytes())
    pinned_blocks_before = int(pinned_pool.n_free_blocks())
    model = build_model(run.case, args, dataset.spec.task, run.seed)
    sketch = model.params.get("multioutput_sketch")
    timing_data = {}
    if isinstance(sketch, GradHessHistory) and getattr(args, 'profile_methods', True):
        timed_methods = ["get_indexers"]
        if hasattr(sketch, "get_weights"):
            timed_methods.insert(0, "get_weights")
        timing_data = time_patch_methods(sketch, tuple(timed_methods))

    _synchronize()
    memory_sampler = _DeviceMemorySampler()
    memory_sampler.start()
    started = time.perf_counter()
    try:
        model.fit(X_train, y_train, eval_sets=[{"X": X_test, "y": y_test}])
        _synchronize()
        train_seconds = time.perf_counter() - started
    finally:
        memory_sampler.stop()
    pool_total_after_fit = int(device_pool.total_bytes())
    pool_used_after_fit = int(device_pool.used_bytes())
    pinned_blocks_after_fit = int(pinned_pool.n_free_blocks())

    model.predict(X_test)
    inference_samples: List[float] = []
    probabilities = None
    for _ in range(args.predict_repeats):
        _synchronize()
        started = time.perf_counter()
        probabilities = model.predict(X_test)
        _synchronize()
        inference_samples.append((time.perf_counter() - started) * 1000)

    assert probabilities is not None
    probabilities = np.asarray(probabilities)
    trees_built = len(model.models)
    record: Dict[str, Any] = {
        "dataset": dataset.spec.name,
        "scenario": run.scenario.value,
        "case_id": run.case.case_id,
        "variant": run.case.variant.value,
        "seed": run.seed,
        "model_seed": run.seed,
        "split_seed": getattr(args, 'split_seed_by_model', {}).get(run.seed, run.seed),
        "dataset_fingerprint": array_fingerprint(dataset.features, dataset.target),
        "source_fingerprint": source_fingerprint(Path(__file__).resolve().parents[1]),
        "task": dataset.spec.task.value,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_outputs": dataset.n_outputs,
        "trees_built": trees_built,
        "train_seconds": train_seconds,
        "inference_median_ms": statistics.median(inference_samples),
        "inference_min_ms": min(inference_samples),
        "inference_repeats": args.predict_repeats,
        "row_subsample": run.case.row_subsample,
        "sketch_outputs": run.case.sketch_outputs,
        "smoothing_alpha": run.case.smoothing_alpha,
        "stabilization_threshold": run.case.stabilization_threshold,
        "destabilization_threshold": run.case.destabilization_threshold,
        "min_warmup_rounds": run.case.min_warmup_rounds,
        "scheduler_window": run.case.scheduler_window,
        "scheduler_min_observations": run.case.scheduler_min_observations,
        "scheduler_patience_on": run.case.scheduler_patience_on,
        "scheduler_patience_off": run.case.scheduler_patience_off,
        "scheduler_robust_k": run.case.scheduler_robust_k,
        "scheduler_cooldown_rounds": run.case.scheduler_cooldown_rounds,
        "scheduler_control_interval": run.case.scheduler_control_interval,
        "adaptive_budgets": list(run.case.adaptive_budgets),
        "adaptive_safe_threshold": run.case.adaptive_safe_threshold,
        "adaptive_unsafe_threshold": run.case.adaptive_unsafe_threshold,
        "adaptive_patience_down": run.case.adaptive_patience_down,
        "adaptive_min_dwell_rounds": run.case.adaptive_min_dwell_rounds,
        "adaptive_cooldown_after_increase_rounds": (
            run.case.adaptive_cooldown_after_increase_rounds
        ),
        "scheduler_mode": run.case.scheduler_mode.value,
        "split_diagnostics_enabled": (
            _split_diagnostics_enabled(run.case, args)
        ),
        "split_output_attribution": args.split_output_attribution,
        "split_candidate_output_profiles": bool(
            getattr(sketch, 'split_candidate_output_profiles', False)
        ),
        "split_refinement_shadow": bool(
            getattr(sketch, 'split_refinement_shadow', False)
        ),
        "split_refinement_apply": bool(
            getattr(sketch, 'split_refinement_apply', False)
        ),
        "split_refinement_determinism_capture": bool(
            getattr(sketch, 'split_refinement_determinism_capture', False)
        ),
        "split_diagnostics_top_k": int(args.split_top_k),
        "spectral_window": args.spectral_window,
        "spectral_null_permutations": args.spectral_null_permutations,
        "spectral_null_rows": args.spectral_null_rows,
        "spectral_null_interval": args.spectral_null_interval,
        "spectral_beta": args.spectral_beta,
        "spectral_tau": args.spectral_tau,
        "spectral_diversity_weight": run.case.spectral_diversity_weight,
        "base_row_subsample": model.params["subsample"],
        "history_row_subsample": model.params.get("history_subsample"),
        "full_leaf_values": model.params.get("full_leaf_values", False),
        "deterministic_leaf_values": model.params.get(
            "deterministic_leaf_values", False
        ),
        "deterministic_histogram": model.params.get(
            "deterministic_histogram", False
        ),
        "deterministic_histogram_layout": model.params.get(
            "deterministic_histogram_layout", "hybrid"
        ),
        "cupy_pool_total_before_mib": pool_total_before / 2**20,
        "cupy_pool_used_before_mib": pool_used_before / 2**20,
        "cupy_pool_total_after_fit_mib": pool_total_after_fit / 2**20,
        "cupy_pool_used_after_fit_mib": pool_used_after_fit / 2**20,
        "cupy_pool_growth_mib": (pool_total_after_fit - pool_total_before) / 2**20,
        "cupy_pinned_free_blocks_before": pinned_blocks_before,
        "cupy_pinned_free_blocks_after_fit": pinned_blocks_after_fit,
        "gpu_used_before_fit_mib": memory_sampler.baseline_used_bytes / 2**20,
        "gpu_peak_used_during_fit_mib": memory_sampler.peak_used_bytes / 2**20,
        "gpu_peak_fit_delta_mib": (
            memory_sampler.peak_used_bytes - memory_sampler.baseline_used_bytes
        ) / 2**20,
    }
    record.update(_quality_metrics(dataset.spec.task, y_test, probabilities))
    record['prediction_fingerprint'] = array_fingerprint(probabilities)
    record['model_structure_fingerprint'] = _model_structure_fingerprint(model)
    record.update(_timing_fields(timing_data))
    record['control_timings'] = dict(getattr(sketch, 'control_timings', {}))
    if isinstance(sketch, GradHessHistory):
        record.update(
            summarize_history(
                sketch,
                trees_built,
                gain_risk_threshold=args.gain_risk_threshold,
            )
        )
    else:
        record.update(
            {
                "history_decisions": 0,
                "history_decision_rate": 0.0,
                "activation_iteration": None,
                "mean_volatility": None,
                "last_volatility": None,
                "mean_sampled_rows": None,
                "mean_sampled_outputs": None,
                "all_indices_unique": None,
            }
        )

    artifact_stem = (
        f"{dataset.spec.name}_{run.scenario.value}_{run.case.case_id}_seed{run.seed}"
    )
    if getattr(args, 'record_selection_indices', False):
        split_seed = record['split_seed']
        train_indices, test_indices = split_benchmark_indices(dataset, split_seed, args.test_size)
        split_path = args.output_dir / 'artifacts' / args.run_id / 'splits' / f'{artifact_stem}.npz'
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_path, train_indices=train_indices, test_indices=test_indices,
                            sample_indices=(np.arange(dataset.features.shape[0])
                                            if dataset.sample_indices is None else dataset.sample_indices))
        record['split_indices_path'] = str(split_path.resolve())
        if isinstance(sketch, GradHessHistory):
            selection_path = split_path.parent.parent / 'selections' / f'{artifact_stem}.json'
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            with selection_path.open('w', encoding='utf-8') as stream:
                json.dump(sketch.decision_trace, stream)
            record['selection_indices_path'] = str(selection_path.resolve())
        config_path = split_path.parent.parent / 'effective_configs' / f'{artifact_stem}.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = asdict(run.case)
        config.update(model_seed=run.seed, split_seed=split_seed,
                      parameters=_serialize_parameters(args),
                      source_fingerprint=record['source_fingerprint'],
                      dataset_fingerprint=record['dataset_fingerprint'])
        with config_path.open('w', encoding='utf-8') as stream:
            json.dump(config, stream, indent=2)
        record['effective_config_path'] = str(config_path.resolve())
    trajectory_recorder = getattr(model, "optimization_trajectory_recorder", None)
    if trajectory_recorder is not None:
        trajectory_path = (
            args.output_dir
            / "artifacts"
            / args.run_id
            / "trajectories"
            / f"{artifact_stem}.npz"
        )
        record["trajectory_path"] = str(
            trajectory_recorder.save(trajectory_path).resolve()
        )
        record["trajectory_snapshots"] = len(trajectory_recorder.snapshots)
    else:
        record["trajectory_path"] = None
        record["trajectory_snapshots"] = 0

    if isinstance(sketch, GradHessHistory) and sketch.split_diagnostics_trace:
        from experiments.split_diagnostic_artifact import write_split_diagnostic_artifact

        split_path = (
            args.output_dir
            / "artifacts"
            / args.run_id
            / "split_diagnostics"
            / f"{artifact_stem}.json"
        )
        for diagnostic_record in sketch.split_diagnostics_trace:
            diagnostic_record.update(
                dataset=dataset.spec.name,
                model_seed=run.seed,
                split_seed=record['split_seed'],
                lambda_l2=float(model.params['lambda_l2']),
                min_gain_to_split=float(model.params['min_gain_to_split']),
            )
        profile_path = write_split_diagnostic_artifact(
            sketch.split_diagnostics_trace,
            split_path,
        )
        record["split_diagnostics_path"] = str(split_path.resolve())
        record["split_diagnostics_profiles_path"] = (
            str(profile_path.resolve()) if profile_path is not None else None
        )
        record["split_diagnostic_nodes"] = len(sketch.split_diagnostics_trace)
    else:
        record["split_diagnostics_path"] = None
        record["split_diagnostics_profiles_path"] = None
        record["split_diagnostic_nodes"] = 0

    refinement_trace = getattr(sketch, 'split_refinement_trace', None)
    if refinement_trace:
        refinement_path = (
            args.output_dir
            / 'artifacts'
            / args.run_id
            / 'split_refinement'
            / f'{artifact_stem}.json'
        )
        refinement_path.parent.mkdir(parents=True, exist_ok=True)
        with refinement_path.open('w', encoding='utf-8') as stream:
            json.dump(refinement_trace, stream, indent=2)
        record['split_refinement_path'] = str(refinement_path.resolve())
        record['split_refinement_batches'] = len(refinement_trace)
    else:
        record['split_refinement_path'] = None
        record['split_refinement_batches'] = 0

    scheduler_trace = getattr(sketch, "scheduler_trace", None)
    if scheduler_trace and run.case.scheduler_mode is SchedulerMode.ADAPTIVE_BUDGET:
        scheduler_path = (
            args.output_dir
            / "artifacts"
            / args.run_id
            / "scheduler_trace"
            / f"{artifact_stem}.json"
        )
        scheduler_path.parent.mkdir(parents=True, exist_ok=True)
        with scheduler_path.open("w", encoding="utf-8") as stream:
            json.dump(scheduler_trace, stream, indent=2)
        record["scheduler_trace_path"] = str(scheduler_path.resolve())
    else:
        record["scheduler_trace_path"] = None

    spectral_trace = getattr(sketch, "spectral_trace", None)
    if spectral_trace:
        spectral_path = (
            args.output_dir
            / "artifacts"
            / args.run_id
            / "spectral_trace"
            / f"{artifact_stem}.json"
        )
        spectral_path.parent.mkdir(parents=True, exist_ok=True)
        with spectral_path.open("w", encoding="utf-8") as stream:
            json.dump(spectral_trace, stream, indent=2)
        record["spectral_trace_path"] = str(spectral_path.resolve())
        record["spectral_updates"] = len(spectral_trace)
    else:
        record["spectral_trace_path"] = None
        record["spectral_updates"] = 0

    del probabilities, model
    _cleanup_gpu()
    return record


def _gpu_name() -> str:
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return True


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    records: Iterable[Dict[str, Any]],
    summary: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Persist structured JSON, per-run CSV, and aggregated CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    datasets = "-".join(metadata["datasets"])
    stem = f"{datasets}_{metadata['scenario']}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    runs_path = output_dir / f"{stem}_runs.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    record_list = list(records)
    summary_list = list(summary)

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "metadata": metadata,
                "results": record_list,
                "summary": summary_list,
            },
            stream,
            indent=2,
        )
    _write_csv(runs_path, record_list)
    _write_csv(summary_path, summary_list)
    return {"json": json_path, "runs_csv": runs_path, "summary_csv": summary_path}


def _plan_record(run: PlannedRun, dataset: str) -> Dict[str, Any]:
    record = asdict(run.case)
    record["variant"] = run.case.variant.value
    record["scheduler_mode"] = run.case.scheduler_mode.value
    return {
        "dataset": dataset,
        "scenario": run.scenario.value,
        "seed": run.seed,
        **record,
    }


def _defaults(args: argparse.Namespace, dataset: LoadedDataset | None, name: str):
    spec = DATASETS[name] if dataset is None else dataset.spec
    sketch_outputs = (
        spec.default_sketch_outputs
        if args.sketch_outputs is None
        else args.sketch_outputs
    )
    return PlanDefaults(
        row_subsample=args.row_subsample,
        sketch_outputs=sketch_outputs,
        smoothing_alpha=args.smoothing_alpha,
        stabilization_threshold=args.stabilization_threshold,
        destabilization_threshold=args.destabilization_threshold,
        min_warmup_rounds=args.min_warmup_rounds,
        scheduler_window=args.scheduler_window,
        scheduler_min_observations=args.scheduler_min_observations,
        scheduler_patience_on=args.scheduler_patience_on,
        scheduler_patience_off=args.scheduler_patience_off,
        scheduler_robust_k=args.scheduler_robust_k,
        scheduler_cooldown_rounds=args.scheduler_cooldown_rounds,
        scheduler_control_interval=args.scheduler_control_interval,
        adaptive_budgets=tuple(args.adaptive_budgets),
        adaptive_safe_threshold=args.adaptive_safe_threshold,
        adaptive_unsafe_threshold=args.adaptive_unsafe_threshold,
        adaptive_patience_down=args.adaptive_patience_down,
        adaptive_min_dwell_rounds=args.adaptive_min_dwell_rounds,
        adaptive_cooldown_after_increase_rounds=(
            args.adaptive_cooldown_after_increase_rounds
        ),
        spectral_diversity_weight=args.spectral_diversity_weight,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=[scenario.value for scenario in Scenario],
        default=Scenario.MAIN.value,
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["helena"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--split-seeds", nargs="+", type=int)
    parser.add_argument("--record-selection-indices", action="store_true")
    parser.add_argument("--output-energy-fraction", type=float, default=0.90)
    parser.add_argument("--output-min-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, help="Deprecated single-seed alias")
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--ntrees", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--row-subsample", type=float, default=0.5)
    parser.add_argument("--sketch-outputs", type=int)
    parser.add_argument("--smoothing-alpha", type=float, default=0.9)
    parser.add_argument("--stabilization-threshold", type=float, default=0.2)
    parser.add_argument("--destabilization-threshold", type=float, default=0.2)
    parser.add_argument("--min-warmup-rounds", type=int, default=5)
    parser.add_argument("--scheduler-window", type=int, default=8)
    parser.add_argument("--scheduler-min-observations", type=int, default=4)
    parser.add_argument("--scheduler-patience-on", type=int, default=2)
    parser.add_argument("--scheduler-patience-off", type=int, default=1)
    parser.add_argument("--scheduler-robust-k", type=float, default=1.0)
    parser.add_argument("--scheduler-cooldown-rounds", type=int, default=2)
    parser.add_argument("--scheduler-control-interval", type=int, default=5)
    parser.add_argument(
        "--adaptive-budgets",
        nargs="+",
        type=int,
        default=[25, 50, 75, 100],
    )
    parser.add_argument("--adaptive-safe-threshold", type=float, default=0.10)
    parser.add_argument("--adaptive-unsafe-threshold", type=float, default=0.15)
    parser.add_argument("--adaptive-patience-down", type=int, default=2)
    parser.add_argument("--adaptive-min-dwell-rounds", type=int, default=5)
    parser.add_argument(
        "--adaptive-cooldown-after-increase-rounds",
        type=int,
        default=10,
    )
    parser.add_argument("--split-diagnostics", action="store_true")
    parser.add_argument("--disable-case-diagnostics", action="store_true")
    parser.add_argument("--split-top-k", type=int, default=5)
    parser.add_argument("--split-output-attribution", action="store_true")
    parser.add_argument("--gain-risk-threshold", type=float, default=0.05)
    parser.add_argument(
        "--trajectory",
        choices=("none", "g", "gh", "z", "ghz"),
        default="none",
    )
    parser.add_argument("--trajectory-window", type=int, default=0)
    parser.add_argument("--trajectory-every", type=int, default=1)
    parser.add_argument("--trajectory-lambda", type=float, default=1.0)
    parser.add_argument("--spectral-window", type=int, default=16)
    parser.add_argument("--spectral-null-permutations", type=int, default=5)
    parser.add_argument("--spectral-null-rows", type=int, default=2048)
    parser.add_argument("--spectral-null-interval", type=int, default=5)
    parser.add_argument("--spectral-beta", type=float, default=0.9)
    parser.add_argument("--spectral-tau", type=float, default=1e-6)
    parser.add_argument("--spectral-diversity-weight", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--predict-repeats", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--verbose", type=int, default=10_000)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--dry-plan", action="store_true")
    parser.add_argument(
        "--case-ids",
        nargs="+",
        help="Run only the named cases from the selected scenario.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnostics"))
    return parser.parse_args()


def _serialize_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    parameters = dict(vars(args))
    for key in ("cache_dir", "output_dir"):
        parameters[key] = str(parameters[key])
    return parameters


def _select_cases(cases, requested_ids):
    """Filter scenario cases while rejecting unknown explicit identifiers."""

    if not requested_ids:
        return cases
    requested = tuple(dict.fromkeys(requested_ids))
    available = {case.case_id: case for case in cases}
    unknown = [case_id for case_id in requested if case_id not in available]
    if unknown:
        raise ValueError(f"Unknown case ids: {', '.join(unknown)}")
    return tuple(available[case_id] for case_id in requested)


def main() -> None:
    args = parse_args()
    args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scenario = Scenario(args.scenario)
    seeds = [args.seed] if args.seed is not None else args.seeds
    args.split_seed_by_model = paired_seed_map(seeds, args.split_seeds)

    if args.dry_plan:
        plan = []
        for name in args.datasets:
            spec = DATASETS[name]
            defaults = _defaults(args, None, name)
            cases = _select_cases(
                cases_for_scenario(scenario, spec.expected_outputs),
                args.case_ids,
            )
            runs = build_run_plan(
                scenario,
                cases,
                seeds,
                defaults,
                spec.expected_outputs,
            )
            plan.extend(_plan_record(run, name) for run in runs)
        print(json.dumps(plan, indent=2))
        return

    records: List[Dict[str, Any]] = []
    shapes: Dict[str, Any] = {}
    plan_snapshot: List[Dict[str, Any]] = []
    for name in args.datasets:
        dataset = load_benchmark_dataset(
            name,
            max_samples=args.max_samples,
            dataset_seed=args.dataset_seed,
            cache_dir=args.cache_dir,
        )
        defaults = _defaults(args, dataset, name)
        cases = _select_cases(
            cases_for_scenario(scenario, dataset.n_outputs),
            args.case_ids,
        )
        runs = build_run_plan(scenario, cases, seeds, defaults, dataset.n_outputs)
        plan_snapshot.extend(_plan_record(run, name) for run in runs)
        splits = {
            seed: split_benchmark_dataset(dataset, args.split_seed_by_model[seed], args.test_size)
            for seed in dict.fromkeys(run.seed for run in runs)
        }
        shapes[name] = {
            "samples": int(dataset.features.shape[0]),
            "features": int(dataset.features.shape[1]),
            "outputs": dataset.n_outputs,
            "source": dataset.spec.source,
        }

        warmed_cases = set()
        for run in runs:
            X_train, X_test, y_train, y_test = splits[run.seed]
            if not args.skip_warmup and run.case.case_id not in warmed_cases:
                print(f"Warming {name}/{run.case.case_id}...", flush=True)
                warmup_case(
                    run.case,
                    args,
                    dataset.spec.task,
                    run.seed,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                )
                warmed_cases.add(run.case.case_id)
            print(
                f"Running {name}/{scenario.value}/{run.case.case_id}/seed={run.seed}...",
                flush=True,
            )
            record = evaluate_run(
                run,
                args,
                dataset,
                X_train,
                X_test,
                y_train,
                y_test,
            )
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)

    records = add_grouped_baseline_ratios(records)
    summary = aggregate_records(records)
    metadata = {
        "datasets": args.datasets,
        "scenario": scenario.value,
        "shapes": shapes,
        "plan": plan_snapshot,
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "gpu": _gpu_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "scikit_learn": sklearn.__version__,
        "parameters": _serialize_parameters(args),
    }
    paths = save_results(records, summary, metadata, args.output_dir)
    print(
        json.dumps(
            {key: str(value.resolve()) for key, value in paths.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
