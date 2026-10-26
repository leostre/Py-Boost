import numpy as np
import pytest
from hypothesis import given, strategies as st
from sklearn.metrics import f1_score
from experiments.label_budget_diagnostics import per_label, compare_labels, training_profile


@given(seed=st.integers(0, 10000), task=st.sampled_from(['multilabel', 'multiclass']))
def test_count_based_f1_has_independent_sklearn_oracle_including_missing_classes(seed, task):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, 12) if task == 'multiclass' else rng.integers(0, 2, (12, 8))
    pred = rng.integers(0, 7, 12) if task == 'multiclass' else rng.integers(0, 2, (12, 8))
    result = per_label(y, pred, task, 8)
    assert result['macro'] == pytest.approx(f1_score(y, pred, average='macro', zero_division=0))
    assert result['weighted'] == pytest.approx(f1_score(y, pred, average='weighted', zero_division=0))


def test_rare_group_uses_training_only_and_does_not_hide_test_missing_labels():
    train = np.zeros((200, 3), int)
    train[:2, 0] = 1
    train[:3, 1] = 1
    y = np.array([[1, 0, 0], [0, 1, 0]])
    full = y.copy()
    candidate = y.copy()
    candidate[0, 0] = 0
    rows, summary = compare_labels(y, full, candidate, train, 'multilabel', 3)
    assert [r['rare_train'] for r in rows] == [True, False, True]
    assert summary['label_f1_losses'] == 1
    assert rows[2]['test_positive'] == 0 and rows[2]['full_f1'] == 0
    assert summary['rare_full_macro_fixed_domain'] == .5
    assert summary['rare_macro_retained'] == 0


def test_prediction_rank_is_invariant_to_softmax_gauge_and_constant_columns_are_explicit():
    rng = np.random.default_rng(910)
    x = np.column_stack((np.zeros(30), np.arange(30))).astype(np.float32)
    raw = rng.normal(size=(30, 4))
    y = np.arange(30)%4
    a = training_profile(x, y, raw, 'multiclass')
    b = training_profile(x, y, raw+rng.normal(size=(30, 1))*100, 'multiclass')
    assert a['train_constant_features'] == 1 and a['prediction_rank_95'] <= 3
    assert a['prediction_effective_rank'] == pytest.approx(b['prediction_effective_rank'])
    assert a['prediction_stable_rank'] == pytest.approx(b['prediction_stable_rank'])
