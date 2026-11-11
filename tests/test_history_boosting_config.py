"""Contract tests for HASBoost-specific model configuration."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py_boost"
    / "gpu"
    / "history_boosting.py"
)


@pytest.fixture
def history_boosting_module(monkeypatch):
    class GradientBoostingStub:
        def __init__(self, loss, full_leaf_values=False, **kwargs):
            self.params = dict(kwargs, loss=loss, full_leaf_values=full_leaf_values)

    class WeightedHistorySamplingStub:
        def __init__(self, **kwargs):
            self.config = kwargs

        def make_decision(self, grad, hess):
            return None

    py_boost_stub = types.ModuleType("py_boost")
    py_boost_stub.GradientBoosting = GradientBoostingStub
    history_stub = types.ModuleType("py_boost.gpu.accumulation.history_callback")
    history_stub.GradHessHistory = type("GradHessHistory", (), {})
    history_stub.WeightedHistorySampling = WeightedHistorySamplingStub

    monkeypatch.setitem(sys.modules, "py_boost", py_boost_stub)
    monkeypatch.setitem(
        sys.modules,
        "py_boost.gpu.accumulation.history_callback",
        history_stub,
    )

    spec = importlib.util.spec_from_file_location("history_boosting_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_history_subsample_is_not_applied_as_unconditional_bagging(history_boosting_module):
    with pytest.warns(FutureWarning, match="history_subsample"):
        model = history_boosting_module.HistoryBasedBoostingModel(loss="crossentropy", subsample=0.4)

    assert model.params["subsample"] is None
    assert model.params["history_subsample"] == 0.4
    assert model.params["full_leaf_values"] is True
    assert model.params["multioutput_sketch"].config["subsample"] == 0.4
    assert model.params["loss"] == "crossentropy"
    assert "loss" not in model.params["multioutput_sketch"].config


def test_sampled_leaf_ablation_can_be_enabled(history_boosting_module):
    model = history_boosting_module.HistoryBasedBoostingModel(
        loss="crossentropy",
        history_subsample=0.4,
        full_leaf_values=False,
    )

    assert model.params["subsample"] is None
    assert model.params["history_subsample"] == 0.4
    assert model.params["full_leaf_values"] is False
