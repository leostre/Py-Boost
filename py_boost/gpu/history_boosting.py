from py_boost import GradientBoosting
from py_boost.gpu.accumulation.history_callback import WeightedHistorySampling
from py_boost.gpu.tree import DepthwiseTreeBuilder


class HistoryBasedBoostingModel(GradientBoosting):
    def __init__(self, **kwargs):
        history_subsample = kwargs.get('subsample', 0.5)
        sketch_factory = kwargs.get('multioutput_sketch', WeightedHistorySampling)
        sketch = sketch_factory(**kwargs)

        boosting_kwargs = dict(kwargs)
        boosting_kwargs['subsample'] = 1.0
        boosting_kwargs['multioutput_sketch'] = sketch
        boosting_kwargs['full_leaf_values'] = True
        super().__init__(**boosting_kwargs)

        self.params['history_subsample'] = history_subsample

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
