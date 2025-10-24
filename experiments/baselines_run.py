import pickle
from functools import partial
from itertools import product
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from py_boost import GradientBoosting, SketchBoost
from py_boost.gpu.accumulation.history_callback import GradHessHistory
from py_boost.gpu.history_boosting import HistoryBasedBoostingModel
from py_boost.gpu.tree import DepthwiseTreeBuilder
from py_boost.multioutput.sketching import *
from py_boost.multioutput.target_splitter import *

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

# Parameter search space for experimentation
SEARCH_SPACE = {
    'sketch_method': SKETCH_METHODS,
    'lr': LR,
    'sketch_outputs': None,  # To be set based on dataset
    'subsample': SAMPLE_RATIO
}

def run_single_experiment(model_generator, params, X, y, cv, dataset_name, run_name):
    """
    Run a single experiment with given parameters on a dataset using cross-validation.
    
    Args:
        model_generator: Function to create the model (e.g., SketchBoost)
        params: Dictionary of model parameters
        X: Feature matrix
        y: Target labels
        cv: Cross-validation strategy
        dataset_name: Name of the dataset
        run_name: Name for the MLflow run
    
    Returns:
        DataFrame with experiment results and list of model training histories
    """
    results = pd.DataFrame(columns=[*SEARCH_SPACE.keys(), 'fold', 'num_trees', 'training_time', 'metric', 'score'])
    histories = []
    global grad_histories, hess_histories
    grad_histories = []
    hess_histories = []

    def patched_before_iteration(self, build_info):
        global grad_histories, hess_histories
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
                    grad_histories.append(self._hist_grad)
                    hess_histories.append(self._hess_grad)

    with mlflow.start_run(run_name=f"{dataset_name}_{run_name}"):
        # Log parameters
        mlflow.log_params({k: v for k, v in params.items() if k in SEARCH_SPACE})
        grad_histories.clear()
        hess_histories.clear()

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Split data into training and test sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Initialize and train model
            model = model_generator(**params)

            GradHessHistory.before_iteration = patched_before_iteration

            start_time = time.time()
            model.fit(
                X_train, y_train,
                eval_sets=[{'X': X_train, 'y': y_train}, {'X': X_test, 'y': y_test}]
            )
            training_time = time.time() - start_time
            num_trees = len(model.models)

            # Predict and evaluate
            predictions = (model.predict(X_test) > PRED_THR).astype(int)
            for metric_name, metric_func in METRICS.items():
                score = metric_func(y_test, predictions)
                results.loc[len(results)] = {
                    **{k: v for k, v in params.items() if k in SEARCH_SPACE},
                    'fold': fold,
                    'num_trees': num_trees,
                    'training_time': training_time,
                    'metric': metric_name,
                    'score': score
                }
                # Log metrics to MLflow
                mlflow.log_metric(f"{metric_name}_fold_{fold}", score)

            histories.append(model.history)

        # Log results DataFrame as an artifact
        results_file = f"results_{dataset_name}_{run_name}.csv"
        results.to_csv(results_file, index=False)
        mlflow.log_artifact(results_file)

        if grad_histories:
            grad_stacked = cp.stack(grad_histories, axis=0)
            grad_stacked_numpy = cp.asnumpy(grad_stacked)
            grad_histories_file = f"grad_histories_{dataset_name}_{run_name}.npy"
            np.save(grad_histories_file, grad_stacked_numpy)
            mlflow.log_artifact(grad_histories_file)

        if hess_histories:
            hess_stacked = cp.stack(hess_histories, axis=0)
            hess_stacked_numpy = cp.asnumpy(hess_stacked)
            hess_histories_file = f"hess_histories_{dataset_name}_{run_name}.npy"
            np.save(hess_histories_file, hess_stacked_numpy)
            mlflow.log_artifact(hess_histories_file)

    grad_histories.clear()
    hess_histories.clear()
    return results, histories

def run_experiments(model_generator, datasets):
    """
    Run experiments across all datasets and parameter combinations with MLflow tracking.
    
    Args:
        model_generator: Function to create the model (e.g., SketchBoost)
        datasets: Dictionary of dataset names and loading functions
    """
    # Set MLflow experiment
    dataset_gen = load_and_preprocess_datasets(datasets)

    for dataset_name, dataset in dataset_gen:
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
            try:
                # Combine default and overridden parameters
                param_dict = dict(zip(SEARCH_SPACE.keys(), params))
                final_params = {**DEFAULTS, **param_dict}

                # Create a unique run name
                run_name = f"run_{idx}_params_{'_'.join([str(v) for v in param_dict.values()])}"

                # Run experiment for current parameter combination
                results, histories = run_single_experiment(
                    model_generator, final_params, X, y, FOLDS, dataset_name, run_name
                )
                all_results.append(results)
                all_histories.extend(histories)
            except Exception as e:
                print(f"Error with parameters {param_dict}: {e}")
                continue

        # Save combined results for the dataset
        final_results = pd.concat(all_results, ignore_index=True)
        final_results_file = f'baselines_{dataset_name}.csv'
        final_results.to_csv(final_results_file, index=False)

        # Save combined histories for the dataset
        final_histories_file = f'histories_{dataset_name}.pkl'
        with open(final_histories_file, 'wb') as file:
            pickle.dump(all_histories, file)

if __name__ == '__main__':
    datasets = {'genbase': {'id': 'genbase', 'source': 'openml', 'version': 2},
    'birds': {'id': 'birds', 'source': 'openml', 'version': 3},}
    SEARCH_SPACE = {
        'sketch_method': ['topk'],
        'lr': LR,
        'sketch_outputs': None,  # To be set based on dataset
        'subsample': SAMPLE_RATIO
    }
    run_experiments(HistoryBasedBoostingModel, datasets)
