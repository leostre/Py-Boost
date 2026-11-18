"""Tests for optimization-history representations and bounded capture."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py_boost"
    / "callbacks"
    / "optimization.py"
)
SPEC = importlib.util.spec_from_file_location("optimization", MODULE_PATH)
optimization = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = optimization
SPEC.loader.exec_module(optimization)

OptimizationTrajectoryRecorder = optimization.OptimizationTrajectoryRecorder
normalized_newton_signal = optimization.normalized_newton_signal


def test_normalized_signal_supports_a_shared_hessian_column():
    gradient = np.array([[2.0, 4.0], [6.0, 8.0]])
    hessian = np.array([[3.0], [15.0]])

    signal = normalized_newton_signal(gradient, hessian, regularization=1.0)

    np.testing.assert_allclose(signal, [[1.0, 2.0], [1.5, 2.0]])


def test_trajectory_window_keeps_only_recent_snapshots():
    recorder = OptimizationTrajectoryRecorder(representation='z', window=2)
    recorder.before_train({})
    for zero_based_iteration in range(3):
        recorder.before_iteration(
            {
                'num_iter': zero_based_iteration,
                'data': {
                    'train': {
                        'grad': np.ones((2, 3)) * (zero_based_iteration + 1),
                        'hess': np.ones((2, 3)),
                    }
                },
            }
        )

    assert [item['iteration'] for item in recorder.snapshots] == [2, 3]
    assert set(recorder.snapshots[-1]) == {'iteration', 'normalized'}
