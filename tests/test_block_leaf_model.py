from dataclasses import replace
import numpy as np
import pytest

from experiments.block_leaf_model import BlockLeafModel,load_prediction_model
from experiments.uniform_threshold_core import ThresholdModel
from experiments.shared_leaf_basis import probabilities
from test_prepared_leaf_inference import model_for


@pytest.mark.parametrize('task', ['multilabel','multiclass'])
def test_variable_basis_blocks_share_only_global_bias_and_roundtrip(task,tmp_path):
    sources = [model_for(task,'hybrid',7),model_for(task,'full',8),model_for(task,'hybrid',9)]
    zero = np.zeros(4,np.float32)
    blocks = tuple(replace(s,base=replace(s.base,bias=zero),decision_shifts=zero) for s in sources)
    bias,shifts = np.array([.1,.2,-.3,.4],np.float32),np.array([.2,-.1,0.,.3],np.float32)
    model = BlockLeafModel(blocks,bias,shifts)
    x = np.array([[0,np.nan],[1,1],[np.nan,0],[.5,1]],np.float32)
    # Independent oracle: each original predictor's unshifted raw logits,subtract
    # its bias then add exactly one shared global bias.
    from experiments.sparse_leaf_core import leaf_design
    expected = np.broadcast_to(bias,(len(x),4)).astype(float).copy()
    for source in sources:
        design = leaf_design(source.base.leaf_indices(x),len(source.base.coefficients))
        expected += source.raw_from_design(design)-source.base.bias
    np.testing.assert_allclose(model.raw(x),expected,atol=2e-14,rtol=0)
    np.testing.assert_allclose(model.predict(x),probabilities(expected,task),atol=2e-14,rtol=0)
    assert model.n_trees == 6 and model.resident_array_bytes == model.payload_bytes+16
    assert all(b.base.bias is model.blocks[0].base.bias and b.decision_shifts is b.base.bias for b in model.blocks)
    with pytest.raises(ValueError):
        model.blocks[0].base.bias.setflags(write=True)
    path = tmp_path/'bank.npz'
    model.save(path)
    loaded = load_prediction_model(path)
    np.testing.assert_array_equal(loaded.predict(x),model.predict(x))
    np.testing.assert_array_equal(loaded.predict_labels(x),model.predict_labels(x))
    with np.load(path,allow_pickle=False) as f:
        assert not any(k.endswith('_bias') or k.endswith('_decision_shifts') for k in f.files)
        arrays = [f[k] for k in f.files if k not in ('version','task','n_features','block_count')
                  and not k.endswith('_representation')]
        assert sum(a.nbytes for a in arrays) == model.payload_bytes
    assert loaded.parameter_count == sum(s.base.coefficients.size+s.base.basis.size+len(s.values) for s in sources)+8
    assert loaded.predict(np.empty((0,2))).shape == (0,4)
    with pytest.raises(ValueError):
        loaded.predict(np.empty((0,3)))
    for invalid in (np.ones((2,3)),np.full((2,2),np.inf)):
        with pytest.raises(ValueError):
            loaded.raw(invalid)


def test_single_block_matches_original_model_and_rejects_nonzero_local_bias(tmp_path):
    source = model_for('multilabel','hybrid')
    zero = np.zeros(4,np.float32)
    block = replace(source,base=replace(source.base,bias=zero),decision_shifts=zero)
    model = BlockLeafModel((block,),source.base.bias,source.decision_shifts)
    x = np.array([[0,np.nan],[1,1]],np.float32)
    np.testing.assert_allclose(model.predict(x),source.predict(x),atol=2e-14,rtol=0)
    with pytest.raises(ValueError):
        BlockLeafModel((source,),zero,zero)
    path = tmp_path/'single.npz'
    source.save(path)
    assert isinstance(load_prediction_model(path),ThresholdModel)


def test_block_schema_rejects_fractional_count(tmp_path):
    source = model_for('multilabel','full')
    zero = np.zeros(4,np.float32)
    block = replace(source,base=replace(source.base,bias=zero),decision_shifts=zero)
    path = tmp_path/'valid.npz'
    BlockLeafModel((block,),zero,zero).save(path)
    with np.load(path,allow_pickle=False) as f:
        arrays = {k:f[k].copy() for k in f.files}
    arrays['block_count'] = np.array(1.5)
    bad = tmp_path/'bad.npz'
    np.savez(bad,**arrays)
    with pytest.raises(ValueError):
        BlockLeafModel.load(bad)
