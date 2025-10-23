from py_boost import GradientBoosting
from py_boost.gpu.accumulation.history_callback import WeightedHistorySampling
from py_boost.gpu.tree import DepthwiseTreeBuilder


class HistoryBasedBoostingModel(GradientBoosting):
    def __init__(self, **kwargs):
        self.industrial_strategy = kwargs.get('industrial_strategy', {})
        if len(self.industrial_strategy) != 0:
            del kwargs['industrial_strategy']
        super().__init__(**kwargs)
        self.sketch_params = self.industrial_strategy.get('sketch_params', {})
        self.use_hess = self.industrial_strategy.get('use_hess', False)
        self.history_period = int(self.industrial_strategy.get('history_period', 10))
        self.history_callback = self.industrial_strategy.get('history_callback', WeightedHistorySampling)

        self.sketch_method = self.history_callback(history_period=self.history_period,
                                                   **self.sketch_params)

        self.params['use_hess'] = self.use_hess
        self.params['multioutput_sketch'] = self.sketch_method

    def _fit(self, builder: DepthwiseTreeBuilder, build_info: dict) -> None:
        # from py_boost.callbacks.callback import CallbackPipeline
        # try:
        #     if hasattr(self, 'callbacks') and getattr(self, 'callbacks') is not None:
        #         existing = list(self.callbacks.callbacks)
        #         existing.append(self.sketch_method)
        #         self.callbacks = CallbackPipeline(*existing)
        # except Exception:
        #     pass
        # finally:
        super()._fit(builder, build_info) 
