from dataclasses import replace
import numpy as np
import pytest
from hypothesis import given,settings,strategies as st
from experiments.parallel_leaf_core import prefix,average_increments,dense_parts,recompress
from experiments.online_leaf_core import stitch
from experiments.history_basis_study import raw


@given(st.integers(0,1000),st.integers(1,2))
@settings(max_examples=30,deadline=None)
def test_prefix_and_average_count_checkpoint_once(seed,trees):
    from test_prepared_leaf_inference import model_for
    full = model_for('multilabel','full',seed)
    checkpoint = prefix(full,trees)
    x = np.array([[0,np.nan],[1,1],[np.nan,0],[.5,1]],np.float32)
    expected = np.broadcast_to(checkpoint.base.bias,(len(x),4)).astype(float).copy()
    leaves = checkpoint.base.leaf_indices(x)
    for ids in leaves:
        expected += checkpoint.base.coefficients[ids]
    np.testing.assert_allclose(raw(checkpoint,x),expected,rtol=0,atol=1e-14)
    zero = np.zeros(4,np.float32)
    increments = [replace(model_for('multilabel','full',seed+j),decision_shifts=zero) for j in (1,2)]
    increments = [replace(m,base=replace(m.base,bias=zero)) for m in increments]
    merged = average_increments(checkpoint,increments)
    np.testing.assert_allclose(merged.raw(x),expected+.5*raw(increments[0],x)+.5*raw(increments[1],x),rtol=0,atol=1e-14)
    assert merged.n_trees == trees+4
    assert np.array_equal(checkpoint.base.bias,full.base.bias)


@pytest.mark.parametrize('task',['multilabel','multiclass'])
def test_recompression_full_rank_preserves_predictions_and_mean(task):
    from test_prepared_leaf_inference import model_for
    checkpoint = prefix(model_for(task,'full'),2)
    zero = np.zeros(4,np.float32)
    increments = [replace(model_for(task,'full',j),decision_shifts=zero) for j in (2,3)]
    increments = [replace(m,base=replace(m.base,bias=zero)) for m in increments]
    merged = average_increments(checkpoint,increments)
    candidates = [average_increments(checkpoint,[a]) for a in increments]
    x = np.tile(np.array([[0,np.nan],[1,1],[np.nan,0],[.5,1]],np.float32),(4,1))
    compact,diagnostic = recompress(merged,candidates,x,4-(task=='multiclass'),0.)
    np.testing.assert_allclose(compact.predict(x),merged.predict(x),rtol=0,atol=2e-6)
    assert diagnostic['mean_preservation_max_error'] <= 2e-6
    assert compact.base.n_trees == merged.n_trees
    assert diagnostic['branch_subspace_overlap'] is not None
    hybrid,_ = recompress(merged,candidates,x,2,.05)
    np.testing.assert_allclose(raw(hybrid,x).mean(axis=0),raw(merged,x).mean(axis=0),atol=2e-6,rtol=0)
    assert len(hybrid.values) <= .05*len(hybrid.base.coefficients)*4


def test_dense_decode_sums_duplicate_sparse_entries_and_rejects_bad_increment():
    from test_prepared_leaf_inference import model_for
    model = model_for('multilabel','hybrid')
    decoded = stitch(dense_parts(model),None,model.base.bias,2,'multilabel')
    x = np.array([[0,np.nan],[1,1],[np.nan,0]],np.float32)
    np.testing.assert_allclose(raw(decoded,x),raw(model,x),rtol=0,atol=3e-7)
    with pytest.raises(ValueError,match='zero-bias'):
        average_increments(model,[model])
