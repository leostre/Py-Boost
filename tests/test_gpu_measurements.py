"""Small GPU acceptance checks; enabled only by the tracked orx runner."""

import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('HASBOOST_GPU_ACCEPTANCE') != '1',
                                reason='GPU acceptance is an orx-managed experiment')


def _data():
    rng = np.random.RandomState(17)
    X = rng.normal(size=(512, 8)).astype(np.float32)
    y = np.argmax(X[:, :6], axis=1).astype(np.int32)
    return X, y


def _common():
    return dict(loss='crossentropy', ntrees=6, max_depth=3, min_data_in_leaf=5,
                lr=0.1, es=0, seed=17, verbose=10000, use_hess=True)


def test_full_budget_matches_original_model():
    from py_boost import GradientBoosting
    from py_boost.gpu.history_boosting import HistoryBasedBoostingModel
    from py_boost.gpu.accumulation.history_callback import CurrentGradientSampling
    X, y = _data()
    reference = GradientBoosting(subsample=1., full_leaf_values=True, **_common()).fit(X, y)
    history = HistoryBasedBoostingModel(multioutput_sketch=CurrentGradientSampling,
        sketch_method='topk', subsample=1., sketch_outputs=6,
        scheduler_mode='fixed_warmup', min_warmup_rounds=1, **_common()).fit(X, y)
    np.testing.assert_allclose(history.predict(X), reference.predict(X), rtol=1e-6, atol=1e-7)


def test_leaf_recalculation_receives_all_rows_and_outputs(monkeypatch):
    import py_boost.gpu.tree as tree_module
    from py_boost.gpu.history_boosting import HistoryBasedBoostingModel
    from py_boost.gpu.accumulation.history_callback import CurrentGradientSampling
    original = tree_module.calc_node_values
    observed = []
    def checked(grad, hess, leaves, row_indices, *args, **kwargs):
        observed.append((grad.shape, hess.shape, row_indices.get()))
        return original(grad, hess, leaves, row_indices, *args, **kwargs)
    monkeypatch.setattr(tree_module, 'calc_node_values', checked)
    X, y = _data()
    model = HistoryBasedBoostingModel(multioutput_sketch=CurrentGradientSampling,
        sketch_method='topk', subsample=.5, sketch_outputs=3,
        scheduler_mode='fixed_warmup', min_warmup_rounds=1,
        full_leaf_values=True, **_common()).fit(X, y)
    assert model.params['multioutput_sketch'].last_decision_shape == (256, 3)
    assert len(observed) == 6
    for grad_shape, hess_shape, indices in observed:
        assert grad_shape == hess_shape == (512, 6)
        np.testing.assert_array_equal(indices, np.arange(512))


def test_adaptive_trace_first_transition_has_safe_patience():
    from py_boost.gpu.history_boosting import HistoryBasedBoostingModel
    from py_boost.gpu.accumulation.history_callback import CurrentGradientSampling
    X, y = _data()
    model = HistoryBasedBoostingModel(multioutput_sketch=CurrentGradientSampling,
        sketch_method='topk', subsample=1., sketch_outputs=3,
        scheduler_mode='adaptive_budget', adaptive_budgets=(3, 6),
        adaptive_safe_threshold=10., adaptive_unsafe_threshold=20.,
        adaptive_patience_down=2, min_warmup_rounds=2,
        adaptive_min_dwell_rounds=0, split_diagnostics_control_interval=2, **_common()).fit(X, y)
    trace = model.params['multioutput_sketch'].scheduler_trace
    first = next(item for item in trace if item['output_budget'] < 6)
    assert first['iteration'] == 5
    assert first['budget_transitioned'] is True
    assert first['reason'] == 'safe_patience_satisfied'
    assert first['observations'] == 2


def test_multiple_shadow_budgets_share_same_tree_and_do_not_change_structure():
    from py_boost import GradientBoosting
    from py_boost.gpu.history_boosting import HistoryBasedBoostingModel
    from py_boost.gpu.accumulation.history_callback import CurrentGradientSampling
    X, y = _data()
    reference = GradientBoosting(subsample=1., full_leaf_values=True, **_common()).fit(X, y)
    model = HistoryBasedBoostingModel(multioutput_sketch=CurrentGradientSampling,
        sketch_method='topk', subsample=1., sketch_outputs=2, scheduler_mode='never',
        split_diagnostics=True, split_diagnostics_top_k=0, **_common())
    sketch = model.params['multioutput_sketch']
    sketch.research_shadow_budgets = (3, 4)
    model.fit(X, y)
    assert not sketch.decision_trace
    np.testing.assert_allclose(model.predict(X), reference.predict(X), rtol=1e-5, atol=1e-6)
    by_node = {}
    for record in sketch.split_diagnostics_trace:
        key = (record['iteration'], record['group'], record['depth'], record['node_id'])
        by_node.setdefault(key, []).append(record)
    assert by_node
    for records in by_node.values():
        assert {r['diagnostic_output_budget'] for r in records} == {2, 3, 4}
        assert len({r['full_best_gain'] for r in records}) == 1
