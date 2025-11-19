import cupy as cp 
from py_boost.gpu.accumulation.history_callback import WeightedHistorySampling

class HyperbolicWeightedHistorySampling(WeightedHistorySampling):

        def get_weights(self, grad: cp.ndarray, hess: cp.ndarray) -> cp.ndarray:
            """
            Compute weights based on differences between current and historical gradients and hessians.
            
            The weight for each element (i,j) is computed as:
                $$w_{i,j} = -(g_{i,j} - \\mu_{g_{i,j}}) \\cdot (h_{i,j} - \\mu_{h_{i,j}})$$
            
            where:
                - $g_{i,j}$: current gradient at position (i,j)
                - $\\mu_{g_{i,j}}$: historical mean gradient at position (i,j) 
                - $h_{i,j}$: current hessian at position (i,j)
                - $\\mu_{h_{i,j}}$: historical mean hessian at position (i,j)
            
            Args:
                grad: Current gradient tensor of shape (n, m).
                hess: Current hessian tensor of shape (n, m).

            Returns:
                weights: Weight tensor of shape (n, m).
            """

            mean_grad = cp.mean(self._hist_grad, axis=0)
            mean_hess = cp.mean(self._hist_hess, axis=0)

            weights = -((grad - mean_grad) * (hess - mean_hess))

            return weights
        

from functools import partial 
from py_boost.gpu.history_boosting import HistoryBasedBoostingModel
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier

def PreparedLGBM(**kwargs):
    return MultiOutputClassifier(
        LGBMClassifier(**{
        'random_state': 42,
        'n_estimators': 500,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'subsample_freq': 1,
        }, **kwargs
        )
    )

HyperbolicHistoryBoost = partial(HistoryBasedBoostingModel, multioutput_sketch=HyperbolicWeightedHistorySampling)
PreparedCatBoost = partial(CatBoostClassifier, od_wait=15, random_seed=42, iterations=10000, loss_function='MultiCrossEntropy')