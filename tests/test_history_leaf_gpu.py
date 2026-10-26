import numpy as np
import pytest


@pytest.mark.parametrize('task', ['multilabel','multiclass'])
def test_history_gpu_keeps_old_coordinates_and_matches_frozen_prefix(task,tmp_path):
    pytest.importorskip('cupy')
    from experiments.a100_runner import build_model
    from experiments.history_leaf_boosting import HistoryLeafBoosting
    from experiments.online_leaf_boosting import CompactLeafBoosting
    from experiments.shared_leaf_basis import probabilities
    from experiments.block_leaf_model import BlockLeafModel
    from experiments.online_leaf_study import raw
    rng = np.random.default_rng(894)
    x = rng.normal(size=(320,6)).astype(np.float32)
    x[::19,0] = np.nan
    y = np.argmax(x[:,1:5],axis=1).astype(np.int32) if task == 'multiclass' else (x[:,1:5]>0).astype(np.float32)
    config = dict(ntrees=8,lr=.1,max_depth=3,lambda_l2=1.)
    args = build_model('full',task,4,config,709,3,1.).params
    reference = CompactLeafBoosting(**args,compact_mode='hybrid',task=task,warmup=2)
    reference.fit(x[:240],y[:240],eval_sets=[dict(X=x[240:],y=y[240:])])
    fixed = reference.export()
    for source in ('current','mean','history'):
        model = HistoryLeafBoosting(**args,basis_source=source,task=task,warmup=2,basis_period=2,observation_period=1)
        model.fit(x[:240],y[:240],eval_sets=[dict(X=x[240:],y=y[240:])])
        exported = model.export()
        assert model.warm_fingerprint == reference.warm_fingerprint
        first = exported.blocks[0]
        assert first.base.n_trees == 4
        n,m = int(fixed.base.node_offsets[4]),int(fixed.base.leaf_offsets[4])
        np.testing.assert_array_equal(first.base.nodes,fixed.base.nodes[:n])
        np.testing.assert_array_equal(first.base.coefficients,fixed.base.coefficients[:m])
        np.testing.assert_array_equal(first.base.basis,fixed.base.basis)
        for field in ('rows','columns','values'):
            np.testing.assert_array_equal(getattr(first,field),getattr(fixed,field)[fixed.rows < m])
        np.testing.assert_array_equal(exported.bias,reference.study_bias)
        assert len(exported.blocks) >= 2 and exported.n_trees == 8
        assert len(model.part_basis_ids) == 8 and len(model.basis_registry) >= 2
        for part,idx in zip(model.parts,model.part_basis_ids):
            assert part.coefficients.shape[1] == model.basis_registry[idx].shape[1]
        # Independent training-state replay catches applying the final basis to
        # all old coefficients; an all-final-basis export would fail here.
        np.testing.assert_allclose(exported.predict(x[240:]),probabilities(model.validation_final_raw[0],task),
                                   rtol=0,atol=2e-5)
        path = tmp_path/f'{task}-{source}.npz'
        exported.save(path)
        np.testing.assert_array_equal(BlockLeafModel.load(path).predict(x[240:]),exported.predict(x[240:]))
        assert all(r['basis_count'] == len(set(model.part_basis_ids[:r['step']])) for r in model.trace)


def test_history_gpu_without_refresh_equals_single_basis_and_refit_resets_state():
    pytest.importorskip('cupy')
    from experiments.a100_runner import build_model
    from experiments.online_leaf_boosting import CompactLeafBoosting
    from experiments.history_leaf_boosting import HistoryLeafBoosting
    rng = np.random.default_rng(945)
    x = rng.normal(size=(160,5)).astype(np.float32)
    y = (x[:,1:5]>0).astype(np.float32)
    config = dict(ntrees=6,lr=.1,max_depth=2,lambda_l2=1.)
    args = build_model('full','multilabel',4,config,129,2,1.).params
    fixed = CompactLeafBoosting(**args,compact_mode='hybrid',task='multilabel',warmup=2)
    fixed.fit(x,y)
    model = HistoryLeafBoosting(**args,basis_source='history',task='multilabel',warmup=2,
                               basis_period=20,observation_period=1)
    fingerprints = []
    for _ in range(2):
        model.fit(x,y)
        deployed = model.export()
        assert len(deployed.blocks) == 1 and len(model.part_basis_ids) == 6
        np.testing.assert_allclose(deployed.predict(x),fixed.predict(x),rtol=0,atol=2e-14)
        fingerprints.append(deployed.predict(x))
    np.testing.assert_array_equal(*fingerprints)
