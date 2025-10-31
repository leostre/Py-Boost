import pickle
from functools import partial
from itertools import product
import time
import numpy as np
import pandas as pd
import mlflow
import gc
import cupy as cp
import multiprocessing as mp
import os
import tempfile
import shutil

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from py_boost.multioutput.sketching import *
from py_boost.multioutput.target_splitter import *
from py_boost.gpu.history_boosting import HistoryBasedBoostingModel

from experiments.data.load_data import DATASETS, load_and_preprocess_datasets
from experiments.constants import SKETCH_METHODS, LR, SAMPLE_RATIO, DEFAULTS

# Configuration for evaluation metrics
AVERAGE = 'weighted'
ZERO_DIVISION = np.nan
METRICS = {
    'f1': partial(f1_score, average=AVERAGE, zero_division=ZERO_DIVISION),
    'accuracy': accuracy_score,
    'precision': partial(precision_score, average=AVERAGE, zero_division=ZERO_DIVISION),
    'recall': partial(recall_score, average=AVERAGE, zero_division=ZERO_DIVISION)
}
PRED_THR = .5

# Configuration for cross-validation
RANDOM_STATE = 42
FOLDS = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Default model parameters
DEFAULTS['es'] = 15
DEFAULTS['ntrees'] = 10000
DEFAULTS['loss'] = 'multilabel'

sketch_methods = [m for m in SKETCH_METHODS if m not in ('svd', None)]
SEARCH_SPACE = {
    'sketch_method': sketch_methods,
    'lr': LR,
    'sketch_outputs': None,
    'subsample': SAMPLE_RATIO
}

def initialize_gpu_settings():
    if cp.is_available():
        try:
            cp.cuda.set_allocator(None)
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=512*1024**2)
        except:
            pass

def nuclear_cleanup():
    gc.collect()
    if cp.is_available():
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        except:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def calculate_aggregated_metrics(fold_scores):
    aggregated = {}
    for metric_name, scores in fold_scores.items():
        if scores:
            scores_array = np.array(scores)
            aggregated.update({
                f'mean_{metric_name}': np.mean(scores_array),
                f'std_{metric_name}': np.std(scores_array),
                f'min_{metric_name}': np.min(scores_array),
                f'max_{metric_name}': np.max(scores_array),
                f'median_{metric_name}': np.median(scores_array)
            })
    return aggregated

def safe_predict(model, X_test, batch_size=1000):
    predictions = []
    n = len(X_test)
    for i in range(0, n, batch_size):
        batch = X_test[i:min(i + batch_size, n)]
        pred_batch = model.predict(batch)
        predictions.append(pred_batch)
    return np.concatenate(predictions)

def run_single_experiment_core(args):
    model_generator, params, X, y, cv, dataset_name, run_name, output_dir = args
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = pd.DataFrame(columns=[*SEARCH_SPACE.keys(), 'fold', 'num_trees', 'training_time', 'metric', 'score'])
    histories = []
    
    fold_scores = {metric_name: [] for metric_name in METRICS.keys()}
    training_times = []
    num_trees_list = []

    # Initialize GPU settings for this process
    initialize_gpu_settings()

    # Create MLflow run
    with mlflow.start_run(run_name=f"{dataset_name}_{run_name}") as run:
        run_id = run.info.run_id
        
        mlflow.log_params({k: v for k, v in params.items() if k in SEARCH_SPACE})
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("n_splits", cv.n_splits)
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = None
            try:
                model = model_generator(**params)
                start_time = time.time()
                model.fit(
                    X_train, y_train,
                    eval_sets=[{'X': X_train, 'y': y_train}, {'X': X_test, 'y': y_test}]
                )
                training_time = time.time() - start_time
                num_trees = len(model.models)
                
                training_times.append(training_time)
                num_trees_list.append(num_trees)
                
                mlflow.log_metric("fold_training_time", training_time, step=fold)
                mlflow.log_metric("fold_num_trees", num_trees, step=fold)

                predictions = (safe_predict(model, X_test, 1000) > PRED_THR).astype(int)
                
                for metric_name, metric_func in METRICS.items():
                    score = metric_func(y_test, predictions)
                    fold_scores[metric_name].append(score)
                    mlflow.log_metric(f"fold_{metric_name}", score, step=fold)
                    mlflow.log_metric(f"{metric_name}_fold_{fold}", score)
                    
                    results.loc[len(results)] = {
                        **{k: v for k, v in params.items() if k in SEARCH_SPACE},
                        'fold': fold,
                        'num_trees': num_trees,
                        'training_time': training_time,
                        'metric': metric_name,
                        'score': score
                    }

                histories.append(model.history)
                
            except Exception as e:
                mlflow.log_param(f"fold_{fold}_error", str(e))
                continue
            finally:
                if model is not None:
                    try:
                        del model
                    except:
                        pass

        # Calculate and log aggregated metrics
        aggregated_metrics = calculate_aggregated_metrics(fold_scores)

        if training_times:
            aggregated_metrics.update({
                'mean_training_time': np.mean(training_times),
                'std_training_time': np.std(training_times),
                'total_training_time': np.sum(training_times),
                'mean_num_trees': np.mean(num_trees_list),
                'std_num_trees': np.std(num_trees_list)
            })

        mlflow.log_metrics(aggregated_metrics)

        mlflow.log_params({
            'mean_accuracy': aggregated_metrics.get('mean_accuracy', np.nan),
            'mean_f1': aggregated_metrics.get('mean_f1', np.nan),
            'n_successful_folds': len(training_times)
        })

        results_file = os.path.join(output_dir, f"results_{dataset_name}_{run_name}.csv")
        results.to_csv(results_file, index=False)
        mlflow.log_artifact(results_file)

        fold_summary = pd.DataFrame({
            'fold': list(range(len(training_times))),
            'training_time': training_times,
            'num_trees': num_trees_list,
            **{f'{metric}_scores': scores for metric, scores in fold_scores.items()}
        })
        fold_summary_file = os.path.join(output_dir, f"fold_summary_{dataset_name}_{run_name}.csv")
        fold_summary.to_csv(fold_summary_file, index=False)
        mlflow.log_artifact(fold_summary_file)

        if histories:
            histories_file = os.path.join(output_dir, f"histories_{dataset_name}_{run_name}.pkl")
            with open(histories_file, 'wb') as file:
                pickle.dump(histories, file)
            mlflow.log_artifact(histories_file)

    return {
        'run_id': run_id,
        'output_dir': output_dir,
        'dataset_name': dataset_name,
        'run_name': run_name,
        'success': True,
        'artifact_files': {
            'results': results_file,
            'fold_summary': fold_summary_file,
            'histories': histories_file if histories else None
        }
    }

def run_experiment_in_process(model_generator, params, X, y, cv, dataset_name, run_name, timeout=3600):
    temp_dir = tempfile.mkdtemp(prefix=f"mlflow_{dataset_name}_{run_name}_")

    worker_args = (model_generator, params, X, y, cv, dataset_name, run_name, temp_dir)
    ctx = mp.get_context('spawn')
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
    if status == 'error':
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Process failed: {result}")
    return result

def _process_worker(result_queue, worker_args):
    try:
        result = run_single_experiment_core(worker_args)
        result_queue.put(('success', result))
    except Exception as e:
        result_queue.put(('error', str(e)))

def run_experiments_silent(model_generator, datasets, run_name='sketchboost_baselines', skip=None, skip_first=None):
    skip_first = skip_first or {}
    skip = skip or {}

    # Set MLflow experiment
    mlflow.set_experiment(run_name)

    dataset_gen = load_and_preprocess_datasets(datasets)

    for dataset_name, dataset in dataset_gen:
        if dataset_name in skip:
            continue

        nuclear_cleanup()

        # Load dataset
        X = dataset['features']
        y = dataset['target']
        n_classes = y.shape[1] if len(y.shape) > 1 else len(np.unique(y))

        SEARCH_SPACE['sketch_outputs'] = [max(1, int(n_classes * ratio)) for ratio in SAMPLE_RATIO]

        all_results = []
        all_histories = []

        param_combinations = list(product(*SEARCH_SPACE.values()))
        for idx, params in enumerate(param_combinations):
            if idx < skip_first.get(dataset_name, 0):
                continue
            try:
                param_dict = dict(zip(SEARCH_SPACE.keys(), params))
                final_params = {**DEFAULTS, **param_dict}

                run_name_suffix = f"run_{idx}_params_{'_'.join([str(v) for v in param_dict.values()])}"

                print(f"Running {dataset_name} - {run_name_suffix}")
                result_info = run_experiment_in_process(
                    model_generator, final_params, X, y, FOLDS, dataset_name, run_name_suffix
                )

                if result_info['success']:
                    artifact_files = result_info['artifact_files']

                    if os.path.exists(artifact_files['results']):
                        results_df = pd.read_csv(artifact_files['results'])
                        all_results.append(results_df)

                    if artifact_files['histories'] and os.path.exists(artifact_files['histories']):
                        with open(artifact_files['histories'], 'rb') as f:
                            histories = pickle.load(f)
                        all_histories.extend(histories)

                    try:
                        if os.path.exists(result_info['output_dir']):
                            shutil.rmtree(result_info['output_dir'], ignore_errors=True)
                    except:
                        pass

                nuclear_cleanup()

            except Exception as e:
                print(f"Failed: {dataset_name} - {param_dict}: {e}")
                with mlflow.start_run(run_name=f"{dataset_name}_failed_{idx}") as run:
                    mlflow.log_params({k: v for k, v in param_dict.items() if k in SEARCH_SPACE})
                    mlflow.log_param("dataset", dataset_name)
                    mlflow.log_param("error", str(e))
                    mlflow.set_tag("status", "FAILED")
                continue

        # Save combined results for the dataset
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            final_results_file = f'baselines_{run_name}_{dataset_name}.csv'
            final_results.to_csv(final_results_file, index=False)

            # Log dataset-level summary
            with mlflow.start_run(run_name=f"{dataset_name}_summary"):
                mlflow.log_param("dataset", dataset_name)
                mlflow.log_param("total_runs", len(param_combinations))
                mlflow.log_param("successful_runs", len(all_results))
                mlflow.log_artifact(final_results_file)

        if all_histories:
            final_histories_file = f'histories_{run_name}_{dataset_name}.pkl'
            with open(final_histories_file, 'wb') as file:
                pickle.dump(all_histories, file)

            # Log combined histories
            with mlflow.start_run(run_name=f"{dataset_name}_summary"):
                mlflow.log_artifact(final_histories_file)

        nuclear_cleanup()
        print(f"Completed dataset: {dataset_name}")

if __name__ == '__main__':
    run_experiments_silent(
        HistoryBasedBoostingModel, 
        DATASETS, 
        skip=('rt_iot2022'), 
        run_name='hyperbolic_fixed', 
        skip_first={}
    )

    print("All experiments completed successfully!")