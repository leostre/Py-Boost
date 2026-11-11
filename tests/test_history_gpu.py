"""GPU regression checks for row ownership and the sketch result contract."""

import warnings

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if not cp.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from py_boost import GradientBoosting
from py_boost.gpu import tree as tree_module
from py_boost.gpu.accumulation.history_callback import GradHessHistory, WeightedHistorySampling
from py_boost.gpu.accumulation.sketches.sketch_methods import SUPPORTED_SKETCH_METHODS
from py_boost.gpu.history_boosting import HistoryBasedBoostingModel
from py_boost.multioutput.sketch_decision import SketchDecision
from py_boost.sampling.bagging import BaseSampler


@pytest.mark.parametrize("method", ["topk", "rand"])
def test_signed_gradient_configuration_is_rejected_before_fit(method):
    with pytest.raises(ValueError, match="requires non-negative weights"):
        GradHessHistory(sketch_method=method)
    assert GradHessHistory(sketch_method="svd").sketch_method == "svd"

    sketch = WeightedHistorySampling(sketch_method=method)
    sketch.use_approximation = True
    grad = cp.array([[-1.0, 2.0], [3.0, -4.0]])
    decision = sketch.make_decision(grad, cp.ones_like(grad))
    assert isinstance(decision, SketchDecision)
    assert decision.row_indices.size == decision.output_indices.size == 1


def test_row_fraction_conflicts_and_effective_configuration():
    with pytest.raises(ValueError, match="requires subsample=None or 1.0"):
        GradientBoosting(loss="mse", subsample=0.4, multioutput_sketch=GradHessHistory())
    with pytest.raises(ValueError, match="requires subsample=None or 1.0"):
        HistoryBasedBoostingModel(loss="mse", history_subsample=0.6, subsample=0.4)
    with pytest.raises(ValueError, match="Conflicting values for 'subsample'"):
        HistoryBasedBoostingModel(loss="mse", history_subsample=0.4, sketch_params={"subsample": 0.75})

    nested = {"subsample": 0.75, "sketch_outputs": 2}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = HistoryBasedBoostingModel(
            loss="mse", sketch_params=nested, seed=17,
            scheduler_mode="adaptive_budget", adaptive_budgets=(1, 2),
            split_diagnostics_control_interval=2,
        )
    sketch = model.params["multioutput_sketch"]
    assert model.params["history_subsample"] == sketch.subsample == 0.75
    assert model.params["subsample"] is None
    assert sketch.sketch_outputs == 2
    assert sketch.shadow_seed == 17
    assert sketch.split_diagnostics_enabled
    sketch.sketch_outputs = 1
    sketch.before_train({})
    assert sketch.sketch_outputs == 2
    assert nested == {"subsample": 0.75, "sketch_outputs": 2}
    with pytest.warns(UserWarning, match="stabilization_window"):
        HistoryBasedBoostingModel(loss="mse", stabilization_window=10)
    with pytest.warns(UserWarning, match="subsampl"):
        GradientBoosting(loss="mse", subsampl=0.5)


@pytest.mark.parametrize("method", list(SUPPORTED_SKETCH_METHODS))
def test_builtin_methods_return_one_result_type(method):
    tensor = cp.arange(1, 33, dtype=cp.float32).reshape(8, 4)
    select = SUPPORTED_SKETCH_METHODS[method]
    decision = select(tensor, subsample=0.5, sketch_outputs=2)
    assert isinstance(decision, SketchDecision)
    rows, outputs = decision
    assert cp.unique(rows).size == 4
    assert cp.unique(outputs).size == 2
    assert decision.row_probabilities is None
    assert decision.output_probabilities is None
    if method in ("topk", "rand", "uniform"):
        recorded = select(tensor, subsample=0.5, sketch_outputs=2, record_probabilities=True)
        assert float(recorded.row_probabilities.sum()) == pytest.approx(1)
        assert float(recorded.output_probabilities.sum()) == pytest.approx(1)


def test_external_tuple_is_adapted_to_a_decision():
    sketch = GradHessHistory()
    sketch.use_approximation = True
    rows = cp.array([0, 2], dtype=cp.uint64)
    outputs = cp.array([1], dtype=cp.uint64)
    sketch.sketch = lambda tensor, **kwargs: (rows, outputs)
    grad = cp.ones((3, 2))
    decision = sketch.make_decision(grad, grad)
    assert isinstance(decision, SketchDecision)
    assert decision.row_indices is rows and decision.output_indices is outputs


@pytest.mark.parametrize("n_outputs", [1, 3])
def test_fit_uses_selected_structure_rows_and_full_leaf_statistics(monkeypatch, n_outputs):
    rng = np.random.default_rng(14)
    x = rng.normal(size=(192, 8)).astype(np.float32)
    if n_outputs == 1:
        loss, y = "mse", x[:, 0] - x[:, 2]
    else:
        loss, y = "crossentropy", x[:, :n_outputs].argmax(axis=1)

    structure_calls = []
    leaf_calls = []
    feature_samples = []
    grow_tree = tree_module.depthwise_grow_tree
    calc_values = tree_module.calc_node_values
    sample_indices = BaseSampler.__call__

    def observe_features(sampler):
        indices = sample_indices(sampler)
        if sampler.axis == 1:
            feature_samples.append(indices.get())
        return indices

    def observe_structure(tree, group, x_gpu, grad, hess, rows, features, *args, **kwargs):
        structure_calls.append((rows.get(), features.get(), grad.shape[1], hess.shape[1]))
        return grow_tree(tree, group, x_gpu, grad, hess, rows, features, *args, **kwargs)

    def observe_leaf_values(grad, hess, leaves, rows, *args, **kwargs):
        leaf_calls.append((rows.get(), grad.shape[1], hess.shape[1]))
        return calc_values(grad, hess, leaves, rows, *args, **kwargs)

    monkeypatch.setattr(BaseSampler, "__call__", observe_features)
    monkeypatch.setattr(tree_module, "depthwise_grow_tree", observe_structure)
    monkeypatch.setattr(tree_module, "calc_node_values", observe_leaf_values)
    model = HistoryBasedBoostingModel(
        loss=loss, history_subsample=0.5, colsample=0.5, ntrees=3, max_depth=2,
        min_data_in_leaf=4, max_bin=16, sketch_method="topk", sketch_outputs=2,
        stabilization_threshold=1e6, use_hess=True, verbose=100,
    ).fit(x, y)

    assert len(structure_calls) == len(leaf_calls) == len(feature_samples) == 3
    assert model.params["multioutput_sketch"].decisions_applied == 2
    for iteration, ((rows, features, grad_width, hess_width), sampled_features) in enumerate(
        zip(structure_calls, feature_samples)
    ):
        assert len(rows) == (len(x) if iteration == 0 else len(x) // 2)
        assert len(np.unique(rows)) == len(rows)
        assert len(features) == x.shape[1] // 2
        np.testing.assert_array_equal(features, sampled_features)
        expected_width = n_outputs if iteration == 0 else min(n_outputs, 2)
        assert grad_width == hess_width == expected_width
    for rows, grad_width, hess_width in leaf_calls:
        np.testing.assert_array_equal(rows, np.arange(len(x)))
        assert grad_width == hess_width == n_outputs
    assert model.predict(x).shape == (len(x), n_outputs)
