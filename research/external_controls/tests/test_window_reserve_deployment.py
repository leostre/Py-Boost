from dataclasses import replace
import numpy as np
from test_abl_quality_core import fixture
from abl_protected_core import as_protected,block_cost
from experiments.shared_leaf_basis import probabilities
from window_reserve_study import replace_residual,rank_variant,topology_signature


def toy():
    original,x,_,_=fixture(36,4,3);b=np.linalg.qr(np.random.default_rng(37).normal(size=(4,4)))[0].astype(np.float32)
    low=replace(original,base=replace(original.base,coefficients=original.base.coefficients[:,:2].copy(),basis=b[:,:2].copy(),representation='low_rank'))
    return as_protected(low),x,b


def test_residual_replay_is_leaf_addressed_and_exact_slot_cost_is_support_independent():
    model,x,b=toy();keys=np.array([0,5,11]);values=np.array([.2,-.1,.4],np.float32)
    actual=replace_residual(model,0,keys,values);expected=model.raw(x);leaves=model.blocks[0].base.leaf_indices(x)
    for key,value in zip(keys,values):expected[:,key%4]+=(leaves==key//4).sum(0)*float(value)
    np.testing.assert_allclose(actual.raw(x),expected,atol=1e-12)
    assert topology_signature(actual)==topology_signature(model)
    np.testing.assert_array_equal(actual.blocks[0].base.coefficients,model.blocks[0].base.coefficients)
    other=replace_residual(model,0,np.array([1,9,19]),values)
    assert block_cost(actual)['cost_bytes']==block_cost(other)['cost_bytes']
    zero=replace_residual(model,0,keys,np.zeros(3,np.float32));assert len(zero.blocks[0].values)==0


def test_full_rank_codec_drops_decoder_but_preserves_probabilities_of_augmented_function():
    model,x,b=toy();extra=np.random.default_rng(38).normal(size=(6,1)).astype(np.float32)*.1
    actual=rank_variant(model,0,b[:,:3],extra);leaves=model.blocks[0].base.leaf_indices(x)
    expected=model.raw(x)+extra[leaves].sum(0).astype(float)@b[:,2:3].T
    np.testing.assert_allclose(probabilities(actual.raw(x),'multiclass'),probabilities(expected,'multiclass'),atol=2e-7)
    assert actual.blocks[0].base.representation=='reference_class'
    assert actual.blocks[0].base.basis.size==0 and topology_signature(actual)==topology_signature(model)
