import multiprocessing as mp
import os
import shutil
import tempfile
from typing import Any, Dict, Tuple
import traceback

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from experiments.data.load_data import SPECIAL_LOADERS
from experiments.core.experiment import BaseExperiment, ExperimentContext
from experiments.core.gpu import GPUTimer, initialize_gpu_settings, safe_predict
from experiments.core.metrics import (
    METRICS,
    ROCAUC_SCORE,
    onelabel_postproc,
    optimize_threshold_per_label,
)
from experiments.core.model_timing import log_timing_data, time_patch_methods
from experiments.core.mlflow_utils import log_param_dict, start_run_for_dataset


def _y_for_stratified_cv_split(y, task: str):
    """
    StratifiedKFold only accepts binary / multiclass as a 1D vector of class
    labels. One-hot targets from LabelBinarizer are 2D {0,1} with row-sum 1;
    sklearn treats them as multilabel-indicator and raises.
    """
    y_arr = np.asarray(y)
    if task == "multilabel":
        return y_arr
    if y_arr.ndim == 2:
        if y_arr.shape[1] == 1:
            return y_arr.ravel()
        uniq = np.unique(y_arr)
        if np.all(np.isin(uniq, [0, 1])) and np.allclose(
            y_arr.sum(axis=1), 1.0, rtol=0.0, atol=1e-7
        ):
            return np.argmax(y_arr, axis=1)
    return np.ravel(y_arr)


def _feature_row_groups(X) -> np.ndarray:
    """
    Build stable group ids so identical feature rows stay in the same fold.
    Used to prevent duplicate-row leakage in grouped CV.
    """
    if isinstance(X, pd.DataFrame):
        hashes = pd.util.hash_pandas_object(X, index=False).to_numpy()
    else:
        x_arr = np.asarray(X)
        if x_arr.ndim != 2:
            x_arr = np.atleast_2d(x_arr)
        x_df = pd.DataFrame(x_arr)
        hashes = pd.util.hash_pandas_object(x_df, index=False).to_numpy()
    return hashes


def _to_metric_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def _extract_alpha_array(history: Any):
    if history is None:
        return None
    if isinstance(history, dict):
        if "alpha" in history:
            try:
                arr = np.asarray(history["alpha"])
                if arr.size:
                    return arr
            except Exception:
                return None
        for v in history.values():
            arr = _extract_alpha_array(v)
            if arr is not None:
                return arr
    return None


def _log_alpha_artifact(history: Any, fold: int, output_dir: str) -> None:
    alpha_arr = _extract_alpha_array(history)
    if alpha_arr is None:
        return
    alpha_dir = os.path.join(output_dir, "alpha_artifacts")
    os.makedirs(alpha_dir, exist_ok=True)
    alpha_path = os.path.join(alpha_dir, f"alpha_fold_{fold}.npy")
    np.save(alpha_path, alpha_arr)
    mlflow.log_artifact(alpha_path, artifact_path="alpha_artifacts")


def run_experiment_in_process(
    experiment: BaseExperiment,
    model_factory,
    params: Dict[str, Any],
    X,
    y,
    context: ExperimentContext,
    run_name: str,
    timeout: int | None = None,
):
    """
    Run a single parameter configuration, optionally in a separate process.
    """
    temp_dir = tempfile.mkdtemp(prefix=f"mlflow_{context.dataset_name}_{run_name}_")

    # Allow disabling multiprocessing via environment for constrained runtimes (e.g. Docker)
    if os.getenv("PY_BOOST_DISABLE_MP", "0") == "1":
        return _run_single_experiment_core(
            experiment,
            model_factory,
            params,
            X,
            y,
            context,
            run_name,
            temp_dir,
        )

    if timeout is None:
        timeout = context.timeout

    worker_args = (
        experiment,
        model_factory,
        params,
        X,
        y,
        context,
        run_name,
        temp_dir,
    )
    try:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(target=_process_worker, args=(result_queue, worker_args))
    except (BrokenPipeError, OSError, RuntimeError) as e:
        # Docker / constrained IPC: semaphore registration for mp.Queue can fail.
        print(
            f"[process_runner] Multiprocessing init failed ({type(e).__name__}: {e}); "
            "falling back to in-process execution."
        )
        return _run_single_experiment_core(
            experiment,
            model_factory,
            params,
            X,
            y,
            context,
            run_name,
            temp_dir,
        )

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise TimeoutError(f"Experiment timed out after {timeout} seconds")

    if result_queue.empty():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("Process completed but no result returned")

    status, result = result_queue.get()
    if status == "error":
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Process failed: {result}")
    return result


def _process_worker(result_queue, worker_args):
    try:
        result = _run_single_experiment_core(*worker_args)
        result_queue.put(("success", result))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        result_queue.put(("error", str(e)))


def _run_single_experiment_core(
    experiment: BaseExperiment,
    model_factory,
    params: Dict[str, Any],
    X,
    y,
    context: ExperimentContext,
    run_name: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Core cross-validation and MLflow logic for a single experiment configuration.
    """
    dataset_name = context.dataset_name

    if (X is None or y is None) and dataset_name not in SPECIAL_LOADERS:
        raise RuntimeError("The wrong configuration of dataset")

    os.makedirs(output_dir, exist_ok=True)

    fold_scores: Dict[str, list] = {metric_name: [] for metric_name in METRICS.keys()}
    training_times: list[float] = []
    num_trees_list: list[int] = []

    # Initialize GPU settings for this process
    initialize_gpu_settings()

    results_df = None
    if context.search_keys and experiment.wants_results_dataframe():
        # Results schema mirrors legacy baselines scripts.
        results_df = pd.DataFrame(
            columns=[
                *context.search_keys,
                "fold",
                "num_trees",
                "training_time",
                "metric",
                "score",
            ]
        )

    with start_run_for_dataset(dataset_name, run_name) as run:
        run_id = run.info.run_id

        # Log parameters (all params for now; experiment subclasses can control content)
        log_param_dict(params, allowed_keys=None)
        mlflow.log_param("dataset", dataset_name)

        def after_split_training(
            X_train, y_train, X_test, y_test, fold: int
        ) -> Tuple[Any, Any, Any, Any]:
            model = None
            try:
                X_tr, y_tr, X_te, y_te = experiment.before_fold(
                    context, fold, params, X_train, y_train, X_test, y_test
                )

                model = experiment.make_model(model_factory, params, context)

                timing_methods = experiment.methods_to_time(context)
                timing_data = None
                if timing_methods:
                    timing_data = time_patch_methods(model=model, method_names=timing_methods)

                with GPUTimer() as fit_time:
                    experiment.fit_model(
                        model=model,
                        X_train=X_tr,
                        y_train=y_tr,
                        X_test=X_te,
                        y_test=y_te,
                        context=context,
                        fold_idx=fold,
                    )

                training_time = fit_time.time
                training_times.append(training_time)

                # per-fold structural metrics when applicable
                n_nodes, n_leaves = experiment.estimate_ensemble_structure(model)
                if n_nodes or n_leaves:
                    mlflow.log_metric(f"mean_nodes_fold_{fold}", n_nodes, step=fold)
                    mlflow.log_metric(f"mean_leaves_fold_{fold}", n_leaves, step=fold)

                # log timing metrics
                if timing_data is not None:
                    log_timing_data(timing_data=timing_data, mlflow=mlflow, fold=fold)

                mlflow.log_metric(f"train_time_fold_{fold}", training_time, step=fold)

                # Try to log number of trees for ensemble-based models
                num_trees = getattr(model, "models", None)
                ntrees = None
                if isinstance(num_trees, (list, tuple)):
                    ntrees = len(num_trees)
                    num_trees_list.append(ntrees)
                    mlflow.log_metric(f"ntrees_fold_{fold}", ntrees, step=fold)

                # Inference
                with GPUTimer() as inference_timer:
                    probas = safe_predict(model, X_te, batch_size=1000)
                inference_time = inference_timer.time
                mlflow.log_metric(
                    f"inference_time_fold_{fold}", inference_time, step=fold
                )

                # ROC AUC when compatible
                try:
                    rocauc = ROCAUC_SCORE(y_te, probas)
                    mlflow.log_metric(f"roc_auc_fold_{fold}", rocauc, step=fold)
                except Exception:
                    pass

                # Post-process probabilities into discrete predictions
                if context.task == "multilabel":
                    threshold_applied = 0.0
                    threshold_fallback = 0.0

                    # BCE loss is computed on probabilities (not thresholded)
                    y_true_test = y_te.astype(float)
                    eps = 1e-10
                    y_prob_test = np.clip(probas, eps, 1.0 - eps)
                    bce_loss = -np.mean(
                        y_true_test * np.log(y_prob_test)
                        + (1.0 - y_true_test) * np.log(1.0 - y_prob_test)
                    )
                    mlflow.log_metric(
                        f"bce_loss_fold_{fold}", float(bce_loss), step=fold
                    )

                    thr = getattr(experiment, "pred_thr")
                    is_adaptive = thr is None or (
                        isinstance(thr, str) and thr.lower() == "adaptive"
                    )

                    if is_adaptive:
                        # Adaptive per-label thresholding is valid only for
                        # non-exclusive multilabel binary targets.
                        y_tr_arr = np.asarray(y_tr)
                        uniq = np.unique(y_tr_arr)
                        is_binary_matrix = (
                            y_tr_arr.ndim == 2
                            and y_tr_arr.shape[1] > 1
                            and np.all(np.isin(uniq, [0, 1]))
                        )
                        is_non_exclusive = False
                        if is_binary_matrix:
                            row_sums = y_tr_arr.sum(axis=1)
                            is_non_exclusive = np.any(row_sums > 1) or np.any(row_sums == 0)

                        if not (is_binary_matrix and is_non_exclusive):
                            # Fallback for one-label encodings (e.g. (N,1) class ids
                            # or one-hot multiclass where rows sum to 1).
                            thr = 0.5
                            threshold_fallback = 1.0
                        else:
                            try:
                                with GPUTimer() as adapt_inference_timer:
                                    probas_tr = safe_predict(
                                        model, X_tr, batch_size=1000
                                    )
                                mlflow.log_metric(
                                    f"adaptive_threshold_inference_time_fold_{fold}",
                                    adapt_inference_timer.time,
                                    step=fold,
                                )
                                thr = optimize_threshold_per_label(
                                    y_tr.astype(float), probas_tr, metric="f1"
                                )
                                threshold_applied = 1.0
                            except Exception:
                                # Never fail fold scoring because of threshold tuning.
                                thr = 0.5
                                threshold_fallback = 1.0

                    predictions = (probas > thr).astype(int)

                    # Diagnostics for thresholding behavior.
                    mlflow.log_metric(
                        f"adaptive_threshold_applied_fold_{fold}",
                        threshold_applied,
                        step=fold,
                    )
                    mlflow.log_metric(
                        f"adaptive_threshold_fallback_fold_{fold}",
                        threshold_fallback,
                        step=fold,
                    )

                    if np.isscalar(thr):
                        mlflow.log_metric(
                            f"threshold_value_fold_{fold}",
                            float(thr),
                            step=fold,
                        )
                        mlflow.log_metric(
                            f"threshold_is_vector_fold_{fold}",
                            0.0,
                            step=fold,
                        )
                    else:
                        thr_arr = np.asarray(thr, dtype=float)
                        mlflow.log_metric(
                            f"threshold_is_vector_fold_{fold}",
                            1.0,
                            step=fold,
                        )
                        mlflow.log_metric(
                            f"threshold_value_mean_fold_{fold}",
                            float(np.mean(thr_arr)),
                            step=fold,
                        )
                        mlflow.log_metric(
                            f"threshold_value_min_fold_{fold}",
                            float(np.min(thr_arr)),
                            step=fold,
                        )
                        mlflow.log_metric(
                            f"threshold_value_max_fold_{fold}",
                            float(np.max(thr_arr)),
                            step=fold,
                        )
                else:
                    # Onelabel task: log multiclass log-loss on probabilities.
                    try:
                        y_true_arr = np.asarray(y_te)
                        if y_true_arr.ndim == 2 and y_true_arr.shape[1] > 1:
                            y_true_logloss = np.argmax(y_true_arr, axis=1)
                        else:
                            y_true_logloss = y_true_arr.reshape(-1)

                        probs_arr = np.asarray(probas, dtype=float)
                        eps = 1e-10
                        if probs_arr.ndim == 1:
                            probs_arr = np.column_stack([1.0 - probs_arr, probs_arr])
                        elif probs_arr.ndim == 2 and probs_arr.shape[1] == 1:
                            p = probs_arr.reshape(-1)
                            probs_arr = np.column_stack([1.0 - p, p])
                        probs_arr = np.clip(probs_arr, eps, 1.0 - eps)
                        probs_arr = probs_arr / probs_arr.sum(axis=1, keepdims=True)

                        multiclass_logloss = log_loss(
                            y_true_logloss,
                            probs_arr,
                            labels=list(range(probs_arr.shape[1])),
                        )
                        mlflow.log_metric(
                            f"multiclass_logloss_fold_{fold}",
                            float(multiclass_logloss),
                            step=fold,
                        )
                    except Exception:
                        pass

                    n_cls = context.n_classes
                    yt = np.asarray(y_te)
                    if yt.ndim == 2 and yt.shape[1] > 1:
                        n_cls = int(yt.shape[1])
                    predictions = onelabel_postproc(probas, n_classes=n_cls)

                # Standard metrics
                fold_metric_values: Dict[str, float] = {}
                for metric_name, metric_func in METRICS.items():
                    score = metric_func(y_te, predictions)
                    fold_scores[metric_name].append(score)
                    fold_metric_values[metric_name] = score
                    mlflow.log_metric(f"{metric_name}_fold_{fold}", score, step=fold)

                    if results_df is not None and context.search_keys:
                        row = {k: params.get(k) for k in context.search_keys}
                        row.update(
                            {
                                "fold": fold,
                                "num_trees": ntrees if ntrees is not None else np.nan,
                                "training_time": training_time,
                                "metric": metric_name,
                                "score": score,
                            }
                        )
                        results_df.loc[len(results_df)] = row

                experiment.after_fold(
                    context=context,
                    fold_idx=fold,
                    model=model,
                    y_test=y_te,
                    probas=probas,
                    predictions=predictions,
                    fold_metrics=fold_metric_values,
                )

                # Store history as MLflow metrics (after hooks may populate it).
                history = getattr(model, "history", None)
                _log_alpha_artifact(history=history, fold=fold, output_dir=output_dir)

            except Exception as e:
                err_text = traceback.format_exc()
                mlflow.log_param(f"fold_{fold}_error", err_text)
                # CUDA illegal-address corrupts the current CUDA context.
                # Continuing to next folds only yields secondary failures (malloc/Event/etc).
                err_lower = err_text.lower()
                if "cudaerrorillegaladdress" in err_lower or "cuda_error_illegal_address" in err_lower:
                    raise RuntimeError(
                        f"Fatal CUDA illegal-address in fold {fold}. "
                        "Aborting remaining folds for this run to avoid cascading errors."
                    ) from e
            finally:
                # Explicitly drop references to the model
                del model

        if dataset_name not in SPECIAL_LOADERS:
            y_split = _y_for_stratified_cv_split(y, context.task)
            splitter = context.cv
            split_kwargs = {}

            # rt_iot2022 has duplicate rows crossing random stratified splits.
            # Keep identical feature rows in the same fold while preserving class stratification.
            if dataset_name == "rt_iot2022" and context.task == "onelabel":
                n_splits = getattr(context.cv, "n_splits", 3)
                shuffle = getattr(context.cv, "shuffle", True)
                random_state = getattr(context.cv, "random_state", 42) if shuffle else None
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=shuffle,
                    random_state=random_state,
                )
                split_kwargs["groups"] = _feature_row_groups(X)

            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(X, y_split, **split_kwargs)
            ):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                after_split_training(X_train, y_train, X_test, y_test, fold)
        else:
            loader = SPECIAL_LOADERS[dataset_name]["loader"]()
            for data in loader:
                # loaders yield (X_train, y_train, X_test, y_test, fold)
                after_split_training(*data)

        results_file = None
        if results_df is not None and not results_df.empty:
            results_file = os.path.join(
                output_dir, f"results_{dataset_name}_{run_name}.csv"
            )
            results_df.to_csv(results_file, index=False)
            mlflow.log_artifact(results_file)

    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "dataset_name": dataset_name,
        "run_name": run_name,
        "success": True,
        "artifact_files": {
            "results": results_file,
            "histories": None,
        },
    }

