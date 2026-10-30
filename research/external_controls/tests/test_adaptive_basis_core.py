import numpy as np
import pytest
import budget_quality_worker
from adaptive_basis_core import risk_basis,risk_weights
from experiments.shared_leaf_basis import functional_basis


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_unit_weights_match_functional_subspace_and_all_prefixes_are_orthonormal(task):
    x=np.random.default_rng(7).normal(size=(81,7));before=x.copy()
    a,_=risk_basis(x,task,np.ones(len(x)),np.ones(7));b=functional_basis(x,task)[1]
    np.testing.assert_array_equal(x,before)
    for rank in [1,3,6]:
        np.testing.assert_allclose(a[:,:rank].T@a[:,:rank],np.eye(rank),atol=2e-6)
        np.testing.assert_allclose(a[:,:rank]@a[:,:rank].T,b[:,:rank]@b[:,:rank].T,atol=2e-6)
    if task=='multiclass':np.testing.assert_allclose(a.sum(0),0,atol=2e-6)


def test_column_importance_changes_selected_direction_but_respects_original_metric():
    x=np.array([[3.,0],[-3.,0],[0,1.],[0,-1.]])
    a,_=risk_basis(x,'multilabel',np.ones(4),np.array([1.,100.]))
    assert abs(a[1,0])>.999
    with pytest.raises(ValueError):risk_basis(x,'multilabel',np.zeros(4),np.ones(2))
    rows,cols=risk_weights(np.array([[1,0],[1,0],[1,0],[0,1]]),x,'multilabel')
    assert np.isfinite(rows).all() and (rows>0).all() and cols[1]>cols[0]
