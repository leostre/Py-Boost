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

import hydra
from  omegaconf import DictConfig

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from py_boost.multioutput.sketching import *
from py_boost.multioutput.target_splitter import *

from data.load_data import DATASETS, load_and_preprocess_datasets, SPECIAL_LOADERS
from constants import SKETCH_METHODS, LR, SAMPLE_RATIO, DEFAULTS

from experimental_fundamentals import (
    GPUTimer, initialize_gpu_settings, 
    nuclear_cleanup, estimate_ensemble_structure, 
    safe_predict, multilabel_postprocess, onelabel_postproc,
    PRED_THR, AVERAGE, ZERO_DIVISION, METRICS, RANDOM_STATE, ROCAUC_SCORE, FOLDS
)

# Default model parameters
DEFAULTS['es'] = 15
DEFAULTS['ntrees'] = 10000
DEFAULTS['loss'] = 'multilabel'

SEARCH_SPACE = {
    'sketch_method': ['topk'],
    'lr': LR,
    'sketch_outputs': None,
    'subsample': SAMPLE_RATIO,
    'label_corruption': [0.05, 0.1, 0.2]
}

        
class LabelCorruptor:
    def __init__(self, corruption_level: float):
        self.corruption_level = corruption_level

    def _flatten(self, y): 
        assert len(y.shape) == 2, 'Assumed y is binarized'
        shape = y.shape
        y = np.argmax(y, axis=-1)
        return y, shape
    
    def _onelabel(self, y: np.array):
        y, shape = self._flatten(y)
        counts = np.unique_counts(y)
        preproba = np.zeros((y.shape[0], len(counts.values)))
        preproba[:, counts.values] = counts.counts
        preproba[np.arange(len(y)), y] = 0
        preproba = preproba / preproba.sum(axis=-1)[..., None]
        new_labels = np.array([np.random.choice(counts.values, p=pp) for pp in preproba])
        output = np.zeros(shape)
        output[np.arange(shape[0]), new_labels] = 1
        return output 
    
    def one_label_corruption(self, target):
        y, shape = self._flatten(target)
        counts = np.unique_counts(y).counts
        proba = counts[y]
        proba = proba / proba.sum()
        selection = np.random.choice(np.arange(len(y)), size=max(1, int(self.corruption_level * len(y))), p=proba)
        corrupted_labels = target.copy()
        corrupted_labels[selection] = self._onelabel(target[selection])
        return corrupted_labels
    
    def multilabel_corruption(self, target):
        assert len(target.shape) == 2, 'Assumed y is binarized'
        mask = np.random.random(target.shape) < self.corruption_level
        target = target.astype(bool)
        target[mask] = ~target[mask]
        return target.astype(int)
    
    @classmethod
    def check_type(cls, y):
        if len(y.shape) == 1:
            return 'one'
        if (y.sum(-1) > 1).any():
            return 'multi'
        return 'one'
    
    def corrupt(self, target):
        if self.check_type(target) == 'one':
            return self.one_label_corruption(target)
        else:
            return self.multilabel_corruption(target)


def run_single_experiment_core(args):
    model_generator, params, X, y, cv, dataset_name, run_name, output_dir, task = args
    
    if (X is None or y is None) and dataset_name not in SPECIAL_LOADERS:
        raise RuntimeError('The wrong configuration of dataset')
    os.makedirs(output_dir, exist_ok=True)

    current_postproc_func = onelabel_postproc if task != 'multilabel' else multilabel_postprocess
    
    # results = pd.DataFrame(columns=[*SEARCH_SPACE.keys(), 'fold', 'num_trees', 'training_time', 'metric', 'score'])
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
        # mlflow.log_param("n_splits", cv.n_splits)
        

        def after_split_training(X_train, y_train, X_test, y_test, fold):
            # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, '='* 80)
            label_corruptor = LabelCorruptor(params['corruption_level'])
            y_train = label_corruptor.corrupt(y_train)
            model = None
            try:
                model = model_generator(**params)
                with GPUTimer() as fit_time:
                    model.fit(
                        X_train, y_train,
                        eval_sets=[{'X': X_train, 'y': y_train}, {'X': X_test, 'y': y_test}]
                    )
                training_time = fit_time.time
                mlflow.log_metric(f'train_time_fold_{fold}', training_time, step=fold)
                num_trees = len(model.models)
                mlflow.log_metric(f'ntrees_fold_{fold}', num_trees)
                n_nodes, n_leaves = estimate_ensemble_structure(model)
                mlflow.log_metric(f'mean_nodes_fold_{fold}', n_nodes)
                mlflow.log_metric(f'mean_leaves_fold_{fold}', n_leaves)
                
                training_times.append(training_time)
                num_trees_list.append(num_trees)
                
                with GPUTimer() as inference_timer:
                    probas = safe_predict(model, X_test, 1000)
                rocauc = ROCAUC_SCORE(y_test, probas)
                mlflow.log_metric(f'roc_auc_fold_{fold}', rocauc)      
                predictions = current_postproc_func(probas)
                inference_time = inference_timer.time
                mlflow.log_metric(f'inference_time_fold_{fold}', inference_time, step=fold)                
                for metric_name, metric_func in METRICS.items():
                    score = metric_func(y_test, predictions)
                    fold_scores[metric_name].append(score)
                    # mlflow.log_metric(f"fold_{metric_name}", score, step=fold)
                    mlflow.log_metric(f"{metric_name}_fold_{fold}", score, step=fold)


                histories.append(model.history)
            except Exception as e:
                mlflow.log_param(f"fold_{fold}_error", str(e))
            finally:
                if model is not None:
                    try:
                        del model
                    except:
                        pass
        if not dataset_name in SPECIAL_LOADERS:
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                after_split_training(X_train, y_train, X_test, y_test, fold)
        else:
            print('NOT IN SPECIAL')
            loader = SPECIAL_LOADERS[dataset_name]['loader']()
            for data in loader:
                after_split_training(*data)

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
            # 'results': results_file,
            # 'fold_summary': fold_summary_file,
            'histories': histories_file if histories else None
        }
    }

def run_experiment_in_process(model_generator, params, X, y, cv, dataset_name, run_name, task, timeout=3600):
    temp_dir = tempfile.mkdtemp(prefix=f"mlflow_{dataset_name}_{run_name}_")

    worker_args = (model_generator, params, X, y, cv, dataset_name, run_name, temp_dir, task)
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
    except KeyboardInterrupt:
            raise
    except Exception as e:
        result_queue.put(('error', str(e)))

def run_experiments_silent(model_generator, datasets, run_name='sketchboost_baselines', skip=None, skip_first=None):
    skip_first = skip_first or {}
    skip = skip or {}

    # Set MLflow experiment
    mlflow.set_experiment(run_name)


    dataset_gen = load_and_preprocess_datasets(datasets)

    for dataset_name, dataset, n_classes in dataset_gen:
        print(dataset_name.center(100, '*'))
        
        nuclear_cleanup()

        if dataset_name in SPECIAL_LOADERS:
            X = y = None
            n_classes = SPECIAL_LOADERS[dataset_name]['n_classes']
            task = SPECIAL_LOADERS[dataset_name]['task']

        else:
            # Load dataset
            X = dataset['features']
            y = dataset['target']
            is_multilabel = len(y.shape) > 1
            n_classes = y.shape[1] if is_multilabel else len(np.unique(y))
            task = 'multilabel' if is_multilabel else 'onelabel'

        SEARCH_SPACE['sketch_outputs'] = [max(1, int(n_classes * ratio)) for ratio in SAMPLE_RATIO]

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
                    model_generator, final_params, X, y, FOLDS, dataset_name, run_name_suffix, task
                )
                print(result_info)
                nuclear_cleanup()
            except KeyboardInterrupt:
                raise
            except Exception as x:
                print(f'Dataset `{dataset_name}` failed due to: {x}')

        nuclear_cleanup()
        print(f"Completed dataset: {dataset_name}")

@hydra.main(version_base=None, config_path="config", config_name="experiment")
def main(cfg: DictConfig):
    
    # Instantiate the model
    # cfg = hydra.utils.instantiate(cfg)
    experiment = hydra.utils.instantiate(cfg.experiment)
    datasets = {name: d for name, d in cfg.datasets.items() if name not in experiment.skip_datasets}
    run_experiments_silent(
        model_generator=hydra.utils.instantiate(cfg.model),  # or pass the instance if supported
        datasets=datasets,
        skip=experiment.skip_datasets,
        run_name=experiment.run_name,
        skip_first=dict(experiment.skip_first),
    )
    
    print("All experiments completed successfully!")

if __name__ == '__main__':
    main()
