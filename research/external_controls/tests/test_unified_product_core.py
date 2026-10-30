from dataclasses import replace
import numpy as np
import pytest
from test_abl_quality_core import fixture
from abl_protected_core import as_protected
from unified_product_extensions import weighted_pair
from abl_quality_core import fingerprint


@pytest.mark.parametrize('weight',[.25,.5,.75])
def test_weighted_model_preserves_logits_including_bias_and_sparse_terms(weight):
    a,x,_,_=fixture(9,4,5);b,_,_,_=fixture(22,4,3)
    basis=np.linalg.qr(np.random.default_rng(5).normal(size=(4,2)))[0].astype(np.float32)
    b=replace(b,base=replace(b.base,representation='low_rank',coefficients=b.base.coefficients[:,:2].copy(),basis=basis),rows=np.array([0,2],np.int32),columns=np.array([1,3],np.int32),values=np.array([.2,-.4],np.float32))
    before=(fingerprint(a),fingerprint(b));first,second=as_protected(a),as_protected(b)
    merged=weighted_pair(first,second,weight)
    np.testing.assert_allclose(merged.raw(x),(1-weight)*first.raw(x)+weight*second.raw(x),atol=2e-6,rtol=0)
    assert merged.n_trees==first.n_trees+second.n_trees and (fingerprint(a),fingerprint(b))==before
    with pytest.raises(ValueError):weighted_pair(first,second,float('nan'))
