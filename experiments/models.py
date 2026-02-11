import cupy as cp

from functools import partial

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from py_boost.gpu.accumulation.history_callback import WeightedHistorySampling
from py_boost.gpu.history_boosting import HistoryBasedBoostingModel
from sklearn.multioutput import MultiOutputClassifier


class HyperbolicWeightedHistorySampling(WeightedHistorySampling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_transform = "hyperbolic"


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