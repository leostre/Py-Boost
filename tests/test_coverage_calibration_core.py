"""Independent grid/count oracles, partition conservation and bounded repair."""
from itertools import combinations
from types import SimpleNamespace
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.coverage_calibration_core import (Partition, SupportStatus, repair_support,
    support_objective, fit_thresholds, decisions, decision_series, label_counts)
from experiments.output_calibration_core import fit_decision_shifts, calibrated_quality
from experiments.stable_budget_core import bootstrap_design, paired_indices


@settings(max_examples=35, deadline=None)
@given(n=st.integers(2, 20), c=st.integers(2, 6), seed=st.integers(0, 9999))
def test_grid_fit_and_explicit_decision_bootstrap_match_inherited_oracles(n, c, seed):
    rng = np.random.default_rng(seed)
    raw, y = rng.normal(size=(n, c))*3, rng.integers(0, 2, (n, c))
    grid = [0, .25, -.25, .5, -.5, 1, -1, 2, -2]
    shifts = fit_thresholds(raw, y, grid)
    np.testing.assert_array_equal(shifts, fit_decision_shifts(raw, y, 'multilabel', grid))
    idx = paired_indices(n, 6, seed)
    series = decision_series(bootstrap_design(y, 'multilabel', c, idx), raw, raw >= shifts)
    for k, i in enumerate(idx):
        expected = calibrated_quality(y[i], raw[i], 'multilabel', shifts)
        for field, value in expected.items():
            assert getattr(series, field)[k] == pytest.approx(value, abs=1e-12)


def test_single_positive_cannot_cover_both_folds_and_not_falsely_repaired():
    y = np.array([[1], [0], [0], [0]])
    p = Partition(np.array([0, 1]), np.array([2, 3]))
    result = repair_support(y, p)
    assert result.status == SupportStatus.NOOP
    assert result.partition is p
    assert not result.solver_statuses
    with pytest.raises(ValueError):
        Partition(np.array([0, 1]), np.array([1, 3]))


def test_repair_matches_full_independent_enumeration_on_small_matrix():
    y = np.array([[1, 1], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
    p = Partition(np.array([0, 1, 2]), np.array([3, 4, 5]))
    result = repair_support(y, p, quota=1, seconds=10)
    def oracle(a):
        b = sorted(set(range(6))-set(a))
        missing = sum(int(sum(y[list(fold), j]) == 0) for fold in (a, b) for j in range(2))
        missing += sum(int(sum(1-y[list(fold), j]) == 0) for fold in (a, b) for j in range(2))
        dev = sum(abs(2*sum(y[list(a), j])-sum(y[:, j])) for j in range(2))
        moves = len(set(a).symmetric_difference(set(p.a)))
        return missing, missing, dev, moves
    assert result.status == SupportStatus.IMPROVED and result.optimal
    assert result.after == min(oracle(a) for a in combinations(range(6), 3))
    np.testing.assert_array_equal(y, np.array([[1, 1], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]]))


def test_solver_timeout_is_unresolved_not_infeasibility():
    p = Partition(np.array([0, 1]), np.array([2, 3]))
    y = np.array([[1], [1], [0], [0]])
    result = repair_support(y, p, solver=lambda *a, **kw: SimpleNamespace(status=1, x=None))
    assert result.status == SupportStatus.UNRESOLVED and not result.optimal
    assert result.partition is p and result.before == result.after


@settings(max_examples=15, deadline=None)
@given(bits=st.lists(st.lists(st.integers(0, 1), min_size=3, max_size=3), min_size=6, max_size=6))
def test_repair_never_degrades_objective_or_loses_objects(bits):
    y = np.array(bits)
    p = Partition(np.array([0, 1, 2]), np.array([3, 4, 5]))
    before = y.copy()
    result = repair_support(y, p, quota=1, seconds=5)
    assert result.after <= result.before
    assert len(result.partition.a) == len(result.partition.b) == 3
    np.testing.assert_array_equal(np.sort(np.r_[result.partition.a, result.partition.b]), np.arange(6))
    np.testing.assert_array_equal(y, before)


def test_oof_uses_only_opposite_fold_thresholds_and_pool_counts_not_mean_f1():
    p = Partition(np.array([0, 1]), np.array([2, 3]))
    raw = np.array([[.5], [1.5], [.5], [1.5]])
    a, b = np.array([0.]), np.array([1.])
    pred = decisions(raw, p, a, b, 1., True)
    np.testing.assert_array_equal(pred.ravel(), [False, True, True, True])
    np.testing.assert_array_equal(decisions(raw, p, a, b, 1., False).ravel(), [True, True])
    y = np.array([[1], [1], [0], [0]])
    pooled = label_counts(y, pred)[3][0]
    separate = (label_counts(y[:2], pred[:2])[3][0]+label_counts(y[2:], pred[2:])[3][0])/2
    assert pooled != separate


def test_own_fold_label_change_cannot_change_its_oof_prediction():
    raw = np.array([[-1.], [.5], [.1], [.2]])
    y = np.array([[1], [0], [1], [0]])
    changed = y.copy()
    changed[:2] = 1-y[:2]
    p = Partition(np.array([0, 1]), np.array([2, 3]))
    def calculate(target):
        return decisions(raw, p, fit_thresholds(raw[p.a], target[p.a], [0, -1, 1]),
                         fit_thresholds(raw[p.b], target[p.b], [0, -1, 1]), 1., True)
    np.testing.assert_array_equal(calculate(y)[p.a], calculate(changed)[p.a])


def test_no_probability_loss_change_when_only_decision_thresholds_change():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    raw = np.array([[-1., 2.], [2., -1.], [.2, .3]])
    design = bootstrap_design(y, 'multilabel', 2, np.array([[0, 1, 2]]))
    first = decision_series(design, raw, raw >= 0)
    second = decision_series(design, raw, raw >= -2)
    np.testing.assert_array_equal(first.loss, second.loss)
    assert first.f1_macro[0] != second.f1_macro[0]
