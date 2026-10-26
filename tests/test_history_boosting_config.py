"""Contract tests for HASBoost-specific model configuration."""

import importlib.util
import sys
import types
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py_boost"
    / "gpu"
    / "history_boosting.py"
)


def test_history_subsample_is_not_applied_as_unconditional_bagging(monkeypatch):
    class GradientBoostingStub:
        def __init__(self, **kwargs):
            self.params = kwargs

        def _fit(self, builder, build_info):
            return None

    class WeightedHistorySamplingStub:
        def __init__(self, **kwargs):
            self.config = kwargs

    py_boost_stub = types.ModuleType("py_boost")
    py_boost_stub.GradientBoosting = GradientBoostingStub
    history_stub = types.ModuleType("py_boost.gpu.accumulation.history_callback")
    history_stub.WeightedHistorySampling = WeightedHistorySamplingStub
    tree_stub = types.ModuleType("py_boost.gpu.tree")
    tree_stub.DepthwiseTreeBuilder = object

    monkeypatch.setitem(sys.modules, "py_boost", py_boost_stub)
    monkeypatch.setitem(
        sys.modules,
        "py_boost.gpu.accumulation.history_callback",
        history_stub,
    )
    monkeypatch.setitem(sys.modules, "py_boost.gpu.tree", tree_stub)

    spec = importlib.util.spec_from_file_location("history_boosting_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.HistoryBasedBoostingModel(loss="crossentropy", subsample=0.4)

    assert model.params["subsample"] == 1.0
    assert model.params["history_subsample"] == 0.4
    assert model.params["full_leaf_values"] is True
    assert model.params["multioutput_sketch"].config["subsample"] == 0.4


def test_sampled_leaf_ablation_can_be_enabled(monkeypatch):
    class GradientBoostingStub:
        def __init__(self, **kwargs):
            self.params = kwargs

        def _fit(self, builder, build_info):
            return None

    class WeightedHistorySamplingStub:
        def __init__(self, **kwargs):
            self.config = kwargs

    py_boost_stub = types.ModuleType("py_boost")
    py_boost_stub.GradientBoosting = GradientBoostingStub
    history_stub = types.ModuleType("py_boost.gpu.accumulation.history_callback")
    history_stub.WeightedHistorySampling = WeightedHistorySamplingStub
    tree_stub = types.ModuleType("py_boost.gpu.tree")
    tree_stub.DepthwiseTreeBuilder = object

    monkeypatch.setitem(sys.modules, "py_boost", py_boost_stub)
    monkeypatch.setitem(
        sys.modules,
        "py_boost.gpu.accumulation.history_callback",
        history_stub,
    )
    monkeypatch.setitem(sys.modules, "py_boost.gpu.tree", tree_stub)

    spec = importlib.util.spec_from_file_location("history_boosting_ablation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.HistoryBasedBoostingModel(
        loss="crossentropy",
        subsample=0.4,
        full_leaf_values=False,
    )

    assert model.params["subsample"] == 1.0
    assert model.params["full_leaf_values"] is False
