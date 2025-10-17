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

from py_boost import GradientBoosting, SketchBoost
from py_boost.multioutput.sketching import *
from py_boost.multioutput.target_splitter import *

from dataset_loading import DATASETS
from setups import SKETCH_METHODS, LR, SAMPLE_RATIO, DEFAULTS

# Configuration for evaluation metrics
AVERAGE = 'weighted'
METRICS = {
    'f1': partial(f1_score, average=AVERAGE),
    'accuracy': accuracy_score,
    'precision': partial(precision_score, average=AVERAGE),
    'recall': partial(recall_score, average=AVERAGE)
}

# Configuration for cross-validation
RANDOM_STATE = 42
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Default model parameters
DEFAULTS['es'] = 30  # Early stopping rounds
DEFAULTS['ntrees'] = 10000  # Maximum number of trees

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

    with mlflow.start_run(run_name=f"{dataset_name}_{run_name}"):
        # Log parameters
        mlflow.log_params({k: v for k, v in params.items() if k in SEARCH_SPACE})
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
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

            # Predict and evaluate
            predictions = np.argmax(model.predict(X_test), axis=-1)
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

        # Log histories as an artifact
        histories_file = f"histories_{dataset_name}_{run_name}.pkl"
        with open(histories_file, 'wb') as file:
            pickle.dump(histories, file)
        mlflow.log_artifact(histories_file)

    return results, histories

def run_experiments(model_generator, datasets):
    """
    Run experiments across all datasets and parameter combinations with MLflow tracking.
    
    Args:
        model_generator: Function to create the model (e.g., SketchBoost)
        datasets: Dictionary of dataset names and loading functions
    """
    # Set MLflow experiment
    mlflow.set_experiment("SketchBoost_Experiments")

    for dataset_name, dataset_loader in datasets:
        # Load dataset
        X, y, n_classes = dataset_loader()
        
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
        mlflow.log_artifact(final_results_file)

        # Save combined histories for the dataset
        final_histories_file = f'histories_{dataset_name}.pkl'
        with open(final_histories_file, 'wb') as file:
            pickle.dump(all_histories, file)
        mlflow.log_artifact(final_histories_file)

if __name__ == '__main__':
    run_experiments(SketchBoost, DATASETS.items())