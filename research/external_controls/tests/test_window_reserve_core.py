import numpy as np
import pytest
from scipy import sparse
from window_reserve_core import quota,support,sparse_operator,objective,fit_support,fit_extra_rank


def test_fractional_quota_and_all_equal_slot_allocators():
    scores=np.random.default_rng(3).uniform(size=(64,6));offsets=np.arange(9)*8
    assert quota(48,.02)==0 and quota(384,.02)==7
    old,_=support(scores,offsets,.02,8,'floor');assert len(old)==0
    for kind in ('carry','uniform','score','random'):
        keys,blocks=support(scores,offsets,.02,8,kind,seed=9)
        assert len(keys)==7 and len(set(keys))==7
        assert all(b['allocated']<=b['global_limit'] for b in blocks)
        again,_=support(scores,offsets,.02,8,kind,seed=9);np.testing.assert_array_equal(keys,again)
    for w in (1,3,8):
        keys,_=support(scores,offsets,.05,w,'score');assert len(keys)==quota(scores.size,.05)


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_sparse_loss_gradient_and_rank_operator_against_dense_finite_difference(task):
    rng=np.random.default_rng(21);n,c=18,4
    leaves=rng.integers(0,5,(n,3));design=sparse.csr_matrix((np.ones(leaves.size),(np.repeat(np.arange(n),3),leaves.ravel())),shape=(n,5))
    keys=np.array([0,3,7,9,17]);op=sparse_operator(design,keys,c);delta=rng.normal(size=len(keys))*.02
    dense=np.zeros((n,c))
    for k,v in zip(keys,delta):dense[:,k%c]+=design[:,k//c].toarray().ravel()*v
    np.testing.assert_allclose((op@delta).reshape(n,c),dense)
    raw=rng.normal(size=(n,c));y=rng.integers(0,c,n) if task=='multiclass' else rng.integers(0,2,(n,c))
    forward=lambda w:(op@w).reshape(n,c);backward=lambda g:np.asarray(op.T@g.ravel()).ravel()
    value,grad=objective(delta,raw,y,task,forward,backward,1.)
    eps=1e-5;eye=np.eye(len(keys))*eps
    numerical=np.array([(objective(delta+e,raw,y,task,forward,backward,1.)[0]-objective(delta-e,raw,y,task,forward,backward,1.)[0])/(2*eps) for e in eye])
    np.testing.assert_allclose(grad,numerical,atol=1e-8)
    values,record=fit_support(design,keys,raw,y,task);assert record['objective_after']<record['objective_before']
    extra=np.linalg.qr(rng.normal(size=(c,2)))[0];a,r=fit_extra_rank(design,extra,raw,y,task)
    assert a.shape==(5,2) and r['objective_after']<r['objective_before']
