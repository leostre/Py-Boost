import numpy as np
from experiments.shared_leaf_basis import LeafEnsemble
from experiments.sparse_leaf_core import Cost, leaf_design
from experiments.leaf_geometry_core import tree_centered_leaf_basis
from experiments.leaf_geometry_study import add_exact_anchor_rows
from experiments.uniform_threshold_study import uniform_plan, materialize_uniform
from experiments.output_calibration_core import calibrated_quality


def test_exact_anchor_allocation_is_materialized_with_equal_real_bytes_not_zero_padding():
    rng = np.random.default_rng(9012)
    dense = LeafEnsemble(np.array([[1, 0, -1, -2], [2, 0, -1, -2]], np.float32),
        np.array([0, 1, 2], np.int64), np.array([0, 2, 4], np.int64),
        rng.normal(size=(4, 8)).astype(np.float32), np.empty((0, 0), np.float32),
        np.zeros(8, np.float32), 2, 'multilabel')
    x = np.tile(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], np.float32), (5, 1))
    y = rng.integers(0, 2, (20, 8))
    roles = dict(train=np.arange(12), calibration=np.arange(12, 16), selection=np.arange(16, 20))
    leaves = {k: dense.leaf_indices(x[v]) for k, v in roles.items()}
    designs = {k: leaf_design(l, 4) for k, l in leaves.items()}
    frequency = np.asarray(designs['train'].mean(axis=0)).ravel()
    mean, basis, _ = tree_centered_leaf_basis(dense.coefficients, dense.leaf_offsets, frequency, 'multilabel')
    cfg = dict(budget_kinds=['joint'], budget_fractions=[.99], base_rank_fractions=[.5],
               residual_capacity_fractions=[1.], tiny_selectors=['balanced_mse'], margin_scale=.5,
               decision_grid=[0, -1, 1], decision_shrinkage=[.5, 1.])
    plan = uniform_plan(dense, dense, mean, basis, leaves['train'], y[roles['train']],
        leaves['calibration'], y[roles['calibration']], leaves['selection'], y[roles['selection']],
        Cost(1000, 4000), cfg, build_selections=False)
    rows = add_exact_anchor_rows(dict(rank=1, nnz=1), plan, dense, mean, basis, designs, y, roles, cfg)
    assert rows
    for row in rows:
        model = materialize_uniform(row, plan, dense, dense, mean, basis)
        assert row['nnz'] == len(model.values) == 1 and model.values[0] != 0
        assert row['parameters'] == model.parameter_count == (4+8)+1+16
        assert row['payload_bytes'] == model.payload_bytes == 16*2+16*3+4*(4+8)+12+8*8
        q = calibrated_quality(y[roles['selection']], model.raw_from_design(designs['selection']),
                               'multilabel', model.decision_shifts)
        assert q == row['validation']
