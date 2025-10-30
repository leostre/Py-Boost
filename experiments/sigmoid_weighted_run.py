import pickle
from functools import partial
from itertools import product
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import mlflow
import gc
import threading
import cupy as cp
from contextlib import contextmanager

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from py_boost.multioutput.sketching import *
from py_boost.multioutput.target_splitter import *
from py_boost.gpu.accumulation.history_callback import GradHessHistory
from py_boost.gpu.history_boosting import HistoryBasedBoostingModel

from data.load_data import DATASETS, load_and_preprocess_datasets
from constants import SKETCH_METHODS, LR, SAMPLE_RATIO, DEFAULTS

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
DEFAULTS['es'] = 15  # Early stopping rounds
DEFAULTS['ntrees'] = 10000  # Maximum number of trees
DEFAULTS['loss'] = 'multilabel'

sketch_methods = [m for m in SKETCH_METHODS if m not in ('svd', None)]
# Parameter search space for experimentation
SEARCH_SPACE = {
    'sketch_method': sketch_methods,
    'lr': LR,
    'sketch_outputs': None,  # To be set based on dataset
    'subsample': SAMPLE_RATIO
}

@contextmanager
def gpu_memory_context():
    try:
        yield
    finally:
        gc.collect()
        if cp.is_available():
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

def run_single_experiment_threaded(model_generator, params, X, y, cv, dataset_name, run_name):
    result_container = {}
    exception_container = {}
    
    def _run_experiment():
        try:
            with gpu_memory_context():
                result = _run_single_experiment_impl(model_generator, params, X, y, cv, dataset_name, run_name)
                result_container['result'] = result
        except Exception as e:
            exception_container['exception'] = e
            print(f"Error in thread for {run_name}: {e}")

    thread = threading.Thread(target=_run_experiment)
    thread.start()
    thread.join()

    if 'exception' in exception_container:
        raise exception_container['exception']
    
    return result_container['result']

def _run_single_experiment_impl(model_generator, params, X, y, cv, dataset_name, run_name):
    results = pd.DataFrame(columns=[*SEARCH_SPACE.keys(), 'fold', 'num_trees', 'training_time', 'metric', 'score'])
    histories = []
    
    # Initialize dictionaries to store fold scores for aggregation
    fold_scores = {metric_name: [] for metric_name in METRICS.keys()}
    training_times = []
    num_trees_list = []
    histories = []
    global grad_histories, hess_histories, raw_grads
    grad_histories = []
    hess_histories = []

    def patched_before_iteration(self, build_info):
        self._current_iteration = build_info['num_iter'] + 1
        train = build_info['data']['train']
        grad: cp.ndarray = train.get('grad')
        hess: cp.ndarray = train.get('hess')
        # accumulate to history when grads and hesses are available
        if grad is not None and hess is not None:
            self._update_history(grad, hess)
            # check if we should apply approximation based on history
            self.use_approximation = self._scheduler()
            if params['lr'] == 0.05 and fold == 0:
                if self._current_iteration % (2 * self.history_period) == 0:
                    grad_histories.append(self._hist_grad.copy())
                    hess_histories.append(self._hist_hess.copy())

    GradHessHistory.before_iteration = patched_before_iteration

    with mlflow.start_run(run_name=f"{dataset_name}_{run_name}"):
        # Log parameters
        mlflow.log_params({k: v for k, v in params.items() if k in SEARCH_SPACE})
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("n_splits", cv.n_splits)
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Clear GPU memory before each fold
            with gpu_memory_context():
                # Split data into training and test sets
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Initialize and train model
                model = model_generator(**params)
                start_time = time.time()
                model.fit(
                    X_train, y_train,
                    eval_sets=[{'X': X_train, 'y': y_train}, {'X': X_test, 'y': y_test}]
                )
                training_time = time.time() - start_time
                num_trees = len(model.models)
                
                # Store fold-level information for aggregation
                training_times.append(training_time)
                num_trees_list.append(num_trees)
                
                # Log fold-level metrics
                mlflow.log_metric("fold_training_time", training_time, step=fold)
                mlflow.log_metric("fold_num_trees", num_trees, step=fold)

                # Predict and evaluate
                predictions = (safe_predict(model, X_test) > PRED_THR).astype(int)
                
                for metric_name, metric_func in METRICS.items():
                    score = metric_func(y_test, predictions)
                    
                    # Store for aggregation
                    fold_scores[metric_name].append(score)
                    
                    # Log individual fold metrics with step
                    mlflow.log_metric(f"fold_{metric_name}", score, step=fold)
                    
                    # Also log with explicit fold number for clarity
                    mlflow.log_metric(f"{metric_name}_fold_{fold}", score)
                    
                    # Store in results dataframe
                    results.loc[len(results)] = {
                        **{k: v for k, v in params.items() if k in SEARCH_SPACE},
                        'fold': fold,
                        'num_trees': num_trees,
                        'training_time': training_time,
                        'metric': metric_name,
                        'score': score
                    }

                histories.append(model.history)

                del model, predictions
                if cp.is_available():
                    cp.get_default_memory_pool().free_all_blocks()

        # Calculate and log aggregated metrics
        aggregated_metrics = calculate_aggregated_metrics(fold_scores)

        # Add training time and num_trees aggregations
        if training_times:
            aggregated_metrics.update({
                'mean_training_time': np.mean(training_times),
                'std_training_time': np.std(training_times),
                'total_training_time': np.sum(training_times),
                'mean_num_trees': np.mean(num_trees_list),
                'std_num_trees': np.std(num_trees_list)
            })

        # Log all aggregated metrics
        mlflow.log_metrics(aggregated_metrics)

        # Log additional aggregated information as parameters for easy filtering
        mlflow.log_params({
            'mean_accuracy': aggregated_metrics.get('mean_accuracy', np.nan),
            'mean_f1': aggregated_metrics.get('mean_f1', np.nan),
            'n_successful_folds': len(training_times)
        })

        # Log results DataFrame as an artifact
        results_file = f"results_{dataset_name}_{run_name}.csv"
        results.to_csv(results_file, index=False)
        mlflow.log_artifact(results_file)

        # Log fold scores summary as artifact
        fold_summary = pd.DataFrame({
            'fold': list(range(len(training_times))),
            'training_time': training_times,
            'num_trees': num_trees_list,
            **{f'{metric}_scores': scores for metric, scores in fold_scores.items()}
        })
        fold_summary_file = f"fold_summary_{dataset_name}_{run_name}.csv"
        fold_summary.to_csv(fold_summary_file, index=False)
        mlflow.log_artifact(fold_summary_file)

        # Log histories as an artifact
        histories_file = f"histories_{dataset_name}_{run_name}.pkl"
        with open(histories_file, 'wb') as file:
            pickle.dump(histories, file)
        mlflow.log_artifact(histories_file)

    return results, histories

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

def check_gpu_memory():
    try:
        mempool = cp.get_default_memory_pool()
        used_gb = mempool.used_bytes() / 1e9
        total_gb = mempool.total_bytes() / 1e9
        print(f"GPU Memory - Used: {used_gb:.2f} GB, Total: {total_gb:.2f} GB")
        return used_gb, total_gb
    except:
        return None, None

def safe_predict(model, X_test, batch_size=10000):
    predictions = []
    n = len(X_test)
    for i in range(0, n, batch_size):
        with gpu_memory_context():
            batch = X_test[i:min(i + batch_size, n)]
            pred_batch = model.predict(batch)
            predictions.append(pred_batch)
            del batch, pred_batch
    
    return np.concatenate(predictions)

def run_experiments(model_generator, datasets, run_name='sketchboost_baselines', skip=None, skip_first=None):
    skip_first = skip_first or {}
    skip = skip or {}

    # Set MLflow experiment
    mlflow.set_experiment(run_name)

    dataset_gen = load_and_preprocess_datasets(datasets)

    for dataset_name, dataset in dataset_gen:
        if dataset_name in skip:
            continue

        # Clear GPU memory before processing dataset
        with gpu_memory_context():
            pass

        # Check available memory before processing large dataset
        used, total = check_gpu_memory()
        if total and used / total > 0.8:  # If >80% memory used
            print(f"High GPU memory usage ({used/total*100:.1f}%), clearing...")
            with gpu_memory_context():
                pass

        # Load dataset
        X = dataset['features']
        y = dataset['target']
        n_classes = len(np.unique(y))

        # Update sketch_outputs in search space based on number of classes
        SEARCH_SPACE['sketch_outputs'] = [max(1, int(n_classes * ratio)) for ratio in SAMPLE_RATIO]

        all_results = []
        all_histories = []

        # Iterate over all parameter combinations
        param_combinations = list(product(*SEARCH_SPACE.values()))
        for idx, params in enumerate(tqdm(param_combinations, desc=f'Processing {dataset_name}')):
            if idx < skip_first.get(dataset_name, 0):
                continue
            try:
                # Combine default and overridden parameters
                param_dict = dict(zip(SEARCH_SPACE.keys(), params))
                final_params = {**DEFAULTS, **param_dict}

                # Create a unique run name
                run_name_suffix = f"run_{idx}_params_{'_'.join([str(v) for v in param_dict.values()])}"
                
                # Run experiment in thread with memory management
                results, histories = run_single_experiment_threaded(
                    model_generator, final_params, X, y, FOLDS, dataset_name, run_name_suffix
                )
                all_results.append(results)
                all_histories.extend(histories)

                # Force cleanup between parameter combinations
                with gpu_memory_context():
                    pass

            except Exception as e:
                print(f"Error with parameters {param_dict}: {e}")
                # Log failed run
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

        # Save combined histories for the dataset
        if all_histories:
            final_histories_file = f'histories_{run_name}_{dataset_name}.pkl'
            with open(final_histories_file, 'wb') as file:
                pickle.dump(all_histories, file)


if __name__ == '__main__':
    run_experiments(HistoryBasedBoostingModel, DATASETS, 
                    skip=('rt_iot2022'), 
                    run_name='weight_sigmoid', 
                    skip_first={})
