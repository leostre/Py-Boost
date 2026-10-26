from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from experiments.closed_budget_study import combine_actual, empty_model, choose_final_checkpoints
from experiments.shared_leaf_basis import compress, probabilities
from experiments.uniform_threshold_core import ThresholdModel
from experiments.sparse_leaf_core import Budget
from experiments.history_basis_study import raw
from test_sparse_leaf_core import teacher


@pytest.mark.parametrize('task', ['multilabel', 'multiclass'])
def test_next_block_receives_actual_decoded_checkpoint_once_not_original_full_teacher(task):
    dense = teacher(task, c=8)
    base = compress(dense, np.zeros(8), np.eye(8)[:, :2])
    checkpoint = ThresholdModel(base, np.array([0], np.int32), np.array([5], np.int32),
                                np.array([.3], np.float32), np.ones(8, np.float32))
    increment = empty_model(replace(dense, bias=np.zeros(8, np.float32)))
    combined = combine_actual(checkpoint, increment, dense.bias)
    x = np.array([[-1], [1], [np.nan]], np.float32)
    expected = raw(checkpoint, x)+raw(increment, x)
    np.testing.assert_allclose(raw(combined, x), expected, atol=3e-7, rtol=0)
    assert np.max(np.abs(raw(combined, x)-(raw(empty_model(dense), x)+raw(increment, x)))) > .1
    assert combined.base.n_trees == 4
    assert np.all(combined.decision_shifts == 0)  # shifts are decisions, not a logit update


def test_first_block_prior_is_added_once_and_not_owned_by_increment():
    dense = teacher('multilabel', c=8)
    inc = empty_model(replace(dense, bias=np.zeros(8, np.float32)))
    prior = np.linspace(-1, 1, 8).astype(np.float32)
    combined = combine_actual(None, inc, prior)
    np.testing.assert_array_equal(combined.base.bias, prior)
    assert not np.shares_memory(combined.base.bias, prior)
    assert np.all(inc.base.bias == 0)


def test_final_empty_and_failed_quality_have_explicit_status_and_no_test_input(tmp_path):
    dense = teacher('multilabel', c=8)
    model = empty_model(compress(dense, np.zeros(8), np.eye(8)[:, :2]))
    x = np.array([[-1], [1]]*6, np.float32)
    y = np.arange(96).reshape(12, 8)%2
    roles = dict(calibration=np.arange(6), selection=np.arange(6, 12))
    cfg = dict(decision_grid=[0], decision_shrinkage=[1.])
    reference = dict(loss=1e-8, f1_macro=1., f1_weighted=1.)
    budget = Budget('joint', .75, 10000, 40000)
    empty = choose_final_checkpoints([], x, y, roles, reference, budget, cfg, tmp_path, tmp_path)
    assert empty['state'] == 'no_accepted_checkpoint' and empty['candidate'] is None
    result = choose_final_checkpoints([(0, model)], x, y, roles, reference, budget, cfg, tmp_path, tmp_path)
    assert result['state'] == 'no_quality_feasible'
    assert (tmp_path/result['artifact']).is_file()
    with pytest.raises(RuntimeError, match='resource cap'):
        choose_final_checkpoints([(0, model)], x, y, roles, reference,
                                 Budget('joint', .75, 1, 4), cfg, tmp_path, tmp_path)


def test_continuation_pool_guard_is_opt_in_and_uses_nested_allocator():
    source = (Path(__file__).parents[1]/'experiments/continuation_leaf_boosting.py').read_text()
    assert 'pool_limit_bytes=None' in source
    assert "raise MemoryError('continuation pool exceeded explicit resource cap')" in source
    assert 'with cp.cuda.using_allocator(tracked)' in source and 'set_allocator(' not in source
