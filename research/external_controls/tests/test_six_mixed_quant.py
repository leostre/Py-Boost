from dataclasses import replace
import numpy as np
import pytest
from test_abl_quality_core import fixture
from abl_budget_revision_core import pack
from six_mixed_quant import prepare_mixed,MixedSnapshot


@pytest.mark.parametrize('codec',['fp16','int8'])
@pytest.mark.parametrize('scope',['a_only','b_only','warm_only','ab_only'])
def test_mixed_factors_keep_float_sections_and_roundtrip(codec,scope,tmp_path):
    m,x,_,_=fixture(4,7,3);rng=np.random.default_rng(14);b=np.linalg.qr(rng.normal(size=(7,3)))[0].astype(np.float32)
    dense=replace(m,base=replace(m.base,bias=np.zeros(7,np.float32)))
    latent=replace(dense,base=replace(dense.base,basis=b,coefficients=rng.normal(size=(6,3)).astype(np.float32),representation='low_rank'))
    model=pack([dense,latent],m.base.bias);s=prepare_mixed(model,np.zeros(7,np.float32),codec+'-'+scope)
    assert all(int(v)%4==0 for row in s.arrays['meta'] for v in row[4:6])
    qcode=1 if codec=='fp16' else 2
    expected=[[qcode if scope=='warm_only' else 0,0],[qcode if scope in ('a_only','ab_only') else 0,qcode if scope in ('b_only','ab_only') else 0]]
    np.testing.assert_array_equal(s.arrays['qformat'],expected)
    error=np.max(np.abs(s.raw(x)-model.raw(x)));assert error<(.01 if codec=='fp16' else .15)
    p=tmp_path/'s.npz';s.save(p);again=MixedSnapshot.load(p);np.testing.assert_array_equal(s.raw(x),again.raw(x));assert again.resident_bytes==s.resident_bytes
