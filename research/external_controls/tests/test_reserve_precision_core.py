import numpy as np
import pytest
from scipy import sparse
from window_reserve_core import fit_support,sparse_operator,objective,fit_extra_rank
from reserve_precision_core import refine_support,refine_rank


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_precision_preserves_original_objective_and_support(task):
    rng=np.random.default_rng(43);n,c=40,4
    design=sparse.csr_matrix((np.ones(n*2),(np.repeat(np.arange(n),2),np.c_[rng.integers(0,4,n),rng.integers(4,8,n)].ravel())),shape=(n,8))
    keys=np.array([0,1,4,7,12,16,18,21,25,31]);raw=rng.normal(size=(n,c));y=rng.integers(0,c,n) if task=='multiclass' else rng.integers(0,2,(n,c))
    initial,_=fit_support(design,keys,raw,y,task,maxiter=2);values,record=refine_support(design,keys,initial,raw,y,task)
    op=sparse_operator(design,keys,c);forward=lambda w:(op@w).reshape(n,c);backward=lambda g:np.asarray(op.T@g.ravel()).ravel()
    before=objective(initial,raw,y,task,forward,backward,1.)[0];after=objective(values,raw,y,task,forward,backward,1.)[0]
    assert after<before and values.shape==initial.shape and record['maximum_gradient']<1e-3
    extra=np.linalg.qr(rng.normal(size=(c,2)))[0];a,_=fit_extra_rank(design,extra,raw,y,task,maxiter=2);b,r=refine_rank(design,extra,a,raw,y,task)
    forward=lambda w:np.asarray(design@w.reshape(8,2))@extra.T;backward=lambda g:np.asarray(design.T@(g@extra)).ravel()
    assert objective(b.ravel(),raw,y,task,forward,backward,1.)[0]<objective(a.ravel(),raw,y,task,forward,backward,1.)[0]
    assert r['maximum_gradient']<1e-3
