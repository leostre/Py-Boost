import multiprocessing as mp
import os
import shutil
import tempfile
from typing import Any, Dict, Tuple
import pickle

import mlflow
import numpy as np
import pandas as pd

from experiments.data.load_data import SPECIAL_LOADERS
from experiments.core.experiment import BaseExperiment, ExperimentContext
from experiments.core.gpu import GPUTimer, initialize_gpu_settings, safe_predict
from experiments.core.metrics import METRICS, ROCAUC_SCORE, onelabel_postproc
from experiments.core.model_timing import log_timing_data, time_patch_methods
from experiments.core.mlflow_utils import log_param_dict, start_run_for_dataset


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
    Run a single parameter configuration in a separate process.
    """
    if timeout is None:
        timeout = context.timeout

    temp_dir = tempfile.mkdtemp(prefix=f"mlflow_{context.dataset_name}_{run_name}_")

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
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_process_worker, args=(result_queue, worker_args))

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

    histories = []
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
                    thr = getattr(experiment, "pred_thr", 0.5)
                    predictions = (probas > thr).astype(int)
                else:
                    predictions = onelabel_postproc(probas)

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

                # Store history if available
                history = getattr(model, "history", None)
                if history is not None:
                    histories.append(history)

                experiment.after_fold(
                    context=context,
                    fold_idx=fold,
                    model=model,
                    y_test=y_te,
                    probas=probas,
                    predictions=predictions,
                    fold_metrics=fold_metric_values,
                )

            except Exception as e:
                mlflow.log_param(f"fold_{fold}_error", str(e))
            finally:
                # Explicitly drop references to the model
                del model

        if dataset_name not in SPECIAL_LOADERS:
            for fold, (train_idx, test_idx) in enumerate(context.cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                after_split_training(X_train, y_train, X_test, y_test, fold)
        else:
            loader = SPECIAL_LOADERS[dataset_name]["loader"]()
            for data in loader:
                # loaders yield (X_train, y_train, X_test, y_test, fold)
                after_split_training(*data)

        histories_file = None
        results_file = None
        if results_df is not None and not results_df.empty:
            results_file = os.path.join(
                output_dir, f"results_{dataset_name}_{run_name}.csv"
            )
            results_df.to_csv(results_file, index=False)
            mlflow.log_artifact(results_file)

        if histories:
            histories_file = os.path.join(
                output_dir, f"histories_{dataset_name}_{run_name}.pkl"
            )
            with open(histories_file, "wb") as file:
                pickle.dump(histories, file)
            mlflow.log_artifact(histories_file)

    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "dataset_name": dataset_name,
        "run_name": run_name,
        "success": True,
        "artifact_files": {
            "results": results_file,
            "histories": histories_file,
        },
    }

