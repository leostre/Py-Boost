from inspect import signature
from numbers import Real
import warnings

from py_boost import GradientBoosting
from py_boost.gpu.accumulation.history_callback import GradHessHistory, WeightedHistorySampling


class HistoryBasedBoostingModel(GradientBoosting):
    """Use one history decision for structure search and all rows for leaf values."""

    def __init__(self, *, history_subsample=None, subsample=None,
                 multioutput_sketch=WeightedHistorySampling, **kwargs):
        """Route history_subsample to the sketch factory and subsample to the builder.

        A numeric subsample alone is a deprecated alias for history_subsample.
        With both parameters set, the builder accepts only subsample=1.0.
        """
        if history_subsample is None and isinstance(subsample, Real):
            warnings.warn(
                'Use history_subsample instead of subsample for HASBoost row selection',
                FutureWarning,
                stacklevel=2,
            )
            history_subsample, subsample = subsample, None

        model_parameters = signature(GradientBoosting.__init__).parameters
        boosting_kwargs = {key: value for key, value in kwargs.items() if key in model_parameters}
        if multioutput_sketch is GradHessHistory or multioutput_sketch is WeightedHistorySampling:
            # Later history callbacks use the model seed for their diagnostic draws.
            sketch_kwargs = {
                key: value for key, value in kwargs.items()
                if key not in model_parameters or key == 'seed'
            }
        else:
            # Third-party factories may also use model settings such as seed.
            sketch_kwargs = dict(kwargs)
        if history_subsample is not None:
            sketch_kwargs['subsample'] = history_subsample
        sketch = multioutput_sketch(**sketch_kwargs)
        if not callable(getattr(sketch, 'make_decision', None)):
            raise TypeError('HistoryBasedBoostingModel requires a sketch with make_decision')

        boosting_kwargs['full_leaf_values'] = True
        super().__init__(subsample=subsample, multioutput_sketch=sketch, **boosting_kwargs)

        nested_params = sketch_kwargs.get('sketch_params') or {}
        requested_fraction = sketch_kwargs.get('subsample', nested_params.get('subsample', 0.5))
        self.params['history_subsample'] = getattr(sketch, 'subsample', requested_fraction)
