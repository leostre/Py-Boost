from py_boost import GradientBoosting
from py_boost.gpu.accumulation.history_callback import WeightedHistorySampling
from py_boost.gpu.tree import DepthwiseTreeBuilder


class HistoryBasedBoostingModel(GradientBoosting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_hess = kwargs.get('use_hess', False)
        self.history_period = int(kwargs.get('history_period', 10))
        self.multioutput_sketch = kwargs.get('multioutput_sketch', WeightedHistorySampling)

        self.params['use_hess'] = self.use_hess
        self.params['multioutput_sketch'] = self.multioutput_sketch(history_period=self.history_period,
                                                                    **kwargs)

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


# class HistoryBasedCallbackPipeline(CallbackPipeline):
#     ...
# TODO: callback communication (history passing)
