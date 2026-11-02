from dataclasses import replace
import numpy as np
import pytest
from test_abl_quality_core import fixture
from abl_protected_core import as_protected
from six_quant_core import prepare_quant,QuantSnapshot,quantize_columns
from abl_budget_revision_core import pack


@pytest.mark.parametrize('codec',['fp16','int8'])
def test_quantized_tree_function_and_roundtrip(codec,tmp_path):
    model,x,_,_=fixture(4,13,17);model=as_protected(model);s=prepare_quant(model,np.zeros(13,np.float32),codec)
    assert s.arrays['values'].dtype==np.float32 and s.arrays['nodes'].dtype==np.float32
    q,sc=quantize_columns(model.blocks[0].base.coefficients,codec)
    w=q.astype(float)*(sc if codec=='int8' else 1.)
    expected=model.bias+sum(w[ids] for ids in model.blocks[0].base.leaf_indices(x))
    np.testing.assert_allclose(s.raw(x),expected,atol=1e-12)
    p=tmp_path/'q.npz';s.save(p);again=QuantSnapshot.load(p)
    assert again.resident_bytes==s.resident_bytes
    np.testing.assert_array_equal(s.raw(x),again.raw(x))


def test_int8_error_bound_and_zero_scale():
    rng=np.random.default_rng(4);x=rng.normal(size=(40,8)).astype(np.float32);x[:,2]=0
    q,s=quantize_columns(x,'int8');assert (s>0).all()
    assert np.all(np.max(np.abs(q.astype(float)*s-x),axis=0)<=s/2+1e-6)


@pytest.mark.parametrize('codec',['fp16','int8'])
def test_low_rank_multiple_bases_and_sparse_residual(codec):
    model,x,_,_=fixture(4,13,5);rng=np.random.default_rng(19);blocks=[];expected=np.broadcast_to(model.base.bias,(len(x),13)).astype(float,copy=True)
    for rank in (3,5):
        b=np.linalg.qr(rng.normal(size=(13,rank)))[0].astype(np.float32)
        a=rng.normal(size=(10,rank)).astype(np.float32)
        base=replace(model.base,basis=b,coefficients=a,representation='low_rank',bias=np.zeros(13,np.float32))
        block=replace(model,base=base,rows=np.array([1,3],np.int32),columns=np.array([2,7],np.int32),values=np.array([.1,-.3],np.float32))
        blocks.append(block);qa,sa=quantize_columns(a,codec);qb,sb=quantize_columns(b,codec)
        values=(qa.astype(float)*(sa if codec=='int8' else 1.))@(qb.astype(float)*(sb if codec=='int8' else 1.)).T
        values[block.rows,block.columns]+=block.values
        for ids in base.leaf_indices(x):expected+=values[ids]
    snapshot=prepare_quant(pack(blocks,model.base.bias),np.zeros(13,np.float32),codec)
    np.testing.assert_allclose(snapshot.raw(x),expected,atol=2e-12)
