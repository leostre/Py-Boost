from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from experiments.data.load_data import load_and_preprocess_datasets, SPECIAL_LOADERS
from experiments.core.cv import FOLDS
from experiments.core.gpu import nuclear_cleanup


@dataclass
class ExperimentContext:
    dataset_name: str
    task: str  # 'multilabel' | 'onelabel'
    n_classes: int
    cv: Any = FOLDS
    timeout: int = 3600
    # Names of hyperparameters that define the search space for this dataset.
    search_keys: Optional[Sequence[str]] = None

@dataclass
class BaseExperiment:
    """
    Base class for all experiments.

    Concrete experiments should override at least ``build_search_space`` and
    optionally ``build_default_params`` / hooks for per-fold customization.
    """

    # Human‑readable name for logging / config
    name: str = "base"
    # MLflow experiment/run name prefix
    run_name: str = "experiment"
    # Datasets to skip completely
    skip_datasets: Tuple[str, ...] = ()
    # Number of initial param configs to skip per dataset
    skip_first: Dict[str, int] = field(default_factory=dict)
    # Default timeout (seconds) for a single parameter configuration
    timeout: int = 3600

    def build_search_space(self, context: ExperimentContext) -> Mapping[str, Sequence[Any]]:
        """
        Return a mapping from hyperparameter name to a sequence of values.
        """
        raise NotImplementedError

    def build_default_params(self, context: ExperimentContext) -> Dict[str, Any]:
        """
        Return default model parameters that will be merged with each search-space
        configuration.
        """
        return {}

    def make_model(self, model_factory, params: Dict[str, Any], context: ExperimentContext):
        """
        Instantiate a model. Default implementation simply calls the factory.
        """
        return model_factory(**params)

    def fit_model(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        context: ExperimentContext,
        fold_idx: int,
    ):
        """
        Train the model on a single fold.

        Default implementation matches the GPU boosting experiments, which
        expect ``eval_sets`` with train and validation data.
        """
        model.fit(
            X_train,
            y_train,
            eval_sets=[
                {"X": X_train, "y": y_train},
                {"X": X_test, "y": y_test},
            ],
        )
        return model

    def methods_to_time(self, context: ExperimentContext) -> Sequence[str]:
        """
        Return a sequence of method names to time using GPU timers.

        By default, no methods are timed. Experiments that care about
        low-level GPU timings can override this.
        """
        return ()

    def wants_results_dataframe(self) -> bool:
        """
        Whether this experiment wants a per-configuration results DataFrame
        to be materialized and returned as an artifact.

        Baseline-style experiments that perform dataset-level aggregation
        should override this to return True.
        """
        return False

    def estimate_ensemble_structure(self, model) -> Tuple[float, float]:
        """
        Return (mean_nodes, mean_leaves) for the ensemble, if applicable.

        Default implementation returns zeros and can be overridden by
        ensemble-based experiments.
        """
        return 0.0, 0.0

    # Hooks
    def before_fold(
        self,
        context: ExperimentContext,
        fold_idx: int,
        params: Dict[str, Any],
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        """
        Hook to modify data before training on a fold (e.g. label corruption).

        The default implementation returns inputs unchanged. Implementations
        may use ``params`` to apply param-dependent preprocessing.
        """
        return X_train, y_train, X_test, y_test

    def after_fold(
        self,
        context: ExperimentContext,
        fold_idx: int,
        model,
        y_test,
        probas,
        predictions,
        fold_metrics: Dict[str, float],
    ) -> None:
        """
        Hook called after a fold has been trained and evaluated.
        """
        # Default implementation does nothing.
        return None

    def after_dataset(
        self,
        context: ExperimentContext,
        all_results: Iterable[Dict[str, Any]],
    ) -> None:
        """
        Optional hook called after all parameter configurations for a dataset
        have been processed.
        """
        return None


class ExperimentRunner:
    """
    High-level orchestrator for running experiments across datasets and
    hyperparameter grids.
    """

    def run(
        self,
        experiment: BaseExperiment,
        model_factory,
        datasets_config: Mapping[str, Any],
        run_name: Optional[str] = None,
    ):
        from experiments.core.process_runner import run_experiment_in_process

        dataset_gen = load_and_preprocess_datasets(datasets_config)

        for dataset_name, dataset, n_classes in dataset_gen:
            if dataset_name in (experiment.skip_datasets or ()):
                continue

            print(dataset_name.center(100, "*"))

            nuclear_cleanup()

            if dataset_name in SPECIAL_LOADERS:
                X = y = None
                info = SPECIAL_LOADERS[dataset_name]
                n_classes = info.get("n_classes", n_classes or 0)
                task = info.get("task", "onelabel")
            else:
                if dataset is None:
                    # Defensive: skip datasets that failed preprocessing.
                    continue
                X = dataset["features"]
                y = dataset["target"]
                is_multilabel = len(y.shape) > 1
                n_classes = y.shape[1] if is_multilabel else len(np.unique(y))
                task = "multilabel" if is_multilabel else "onelabel"

            context = ExperimentContext(
                dataset_name=dataset_name,
                task=task,
                n_classes=n_classes or 0,
                cv=FOLDS,
                timeout=experiment.timeout,
            )

            search_space = experiment.build_search_space(context)
            default_params = experiment.build_default_params(context)

            # Deterministic ordering of search keys for reproducible run names.
            keys = list(search_space.keys())
            context.search_keys = tuple(keys)
            param_combinations = list(product(*[search_space[k] for k in keys]))

            all_results = []

            for idx, values in enumerate(param_combinations):
                if idx < (experiment.skip_first or {}).get(dataset_name, 0):
                    continue

                param_dict = dict(zip(keys, values))
                final_params = {**default_params, **param_dict}

                run_name_suffix = f"run_{idx}_params_{'_'.join([str(v) for v in param_dict.values()])}"

                print(f"Running {dataset_name} - {run_name_suffix}")
                result_info = run_experiment_in_process(
                    experiment=experiment,
                    model_factory=model_factory,
                    params=final_params,
                    X=X,
                    y=y,
                    context=context,
                    run_name=run_name_suffix if run_name is None else f"{run_name}_{run_name_suffix}",
                )
                all_results.append(result_info)
                nuclear_cleanup()

            experiment.after_dataset(context, all_results)

