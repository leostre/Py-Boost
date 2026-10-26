import inspect
import numpy as np
import pytest
from experiments.prefix_budget_study import prefix_grid
from experiments.output_calibration_core import calibrated_quality
from experiments.sparse_leaf_core import leaf_design
from test_sparse_leaf_core import teacher


@pytest.mark.parametrize('task', ['multilabel', 'multiclass'])
def test_prefix_grid_keeps_prior_once_native_tree_values_and_exact_storage(task):
    dense = teacher(task, c=8)
    x = np.array([[-1], [1], [np.nan], [1]], np.float32)
    y = np.arange(4)%3 if task == 'multiclass' else np.arange(32).reshape(4, 8)%2
    cfg = dict(prefix_checkpoints=[1, 2], decision_grid=[0, -1, 1], decision_shrinkage=[.5, 1.])
    rows, models = prefix_grid({5: dense}, x, y, x, y, cfg)
    assert len(rows) == 6
    for r in rows:
        model = models[(5, r['trees'])]
        assert r['parameters'] == model.parameter_count and r['payload_bytes'] == model.payload_bytes
        assert model.base.n_trees == r['trees']
        assert len(model.base.coefficients) == (2 if r['trees'] == 1 else 3)
        model.decision_shifts[:] = r['decision_shifts']
        design = leaf_design(model.base.leaf_indices(x), len(model.base.coefficients))
        q = calibrated_quality(y, model.raw_from_design(design), task, model.decision_shifts)
        assert q == r['validation']
        if r['trees'] == 2:
            np.testing.assert_allclose(model.predict(x), dense.predict(x), atol=2e-7, rtol=0)
    assert not any('test' in p for p in inspect.signature(prefix_grid).parameters)
