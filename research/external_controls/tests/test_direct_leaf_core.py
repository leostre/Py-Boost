import numpy as np
import pytest
from scipy.special import logsumexp
from direct_leaf_core import derivatives,leaf_statistics_reference,coupled_step,diagonal_latent_curvature


@pytest.mark.parametrize('task',['multiclass','multilabel'])
def test_projected_curvature_matches_finite_difference_and_full_solve(task):
    rng=np.random.default_rng(27);raw=rng.normal(size=(11,4));y=rng.integers(0,4,11) if task=='multiclass' else rng.integers(0,2,(11,4))
    basis=np.linalg.qr(rng.normal(size=(4,3)))[0];leaves=np.arange(11)%3
    stats=leaf_statistics_reference(raw,y,leaves,basis,task);g,h=derivatives(raw,y,task)
    a=rng.normal(size=3)*.1;leaf=1;mask=leaves==leaf;eps=1e-5
    def loss(u):
        z=raw[mask]+u@basis.T
        return float(np.sum(logsumexp(z,axis=1)-z[np.arange(len(z)),y[mask]])) if task=='multiclass' else float(np.sum(np.logaddexp(0,z)-y[mask]*z))
    numerical=np.array([(loss(np.eye(3)[j]*eps)-loss(-np.eye(3)[j]*eps))/(2*eps) for j in range(3)])
    np.testing.assert_allclose(numerical,stats['gradient'][leaf],atol=1e-8)
    from scipy.special import expit,softmax
    p=softmax(raw,axis=1) if task=='multiclass' else expit(raw)
    diag=diagonal_latent_curvature(p,basis,task)
    np.testing.assert_allclose(diag,np.einsum('cr,ncs,sr->nr',basis,h,basis),atol=1e-12)
    full=leaf_statistics_reference(raw,y,leaves,np.eye(4),task)
    expected=np.stack([np.linalg.solve(full['full_hessian'][i]+np.eye(4),-full['full_gradient'][i]) for i in range(3)])
    np.testing.assert_allclose(coupled_step(full['gradient'],full['hessian'],1.),expected,atol=1e-12)
    step=coupled_step(stats['gradient'],stats['hessian'],1.)
    residual=np.einsum('lij,lj->li',stats['hessian']+np.eye(3),step)+stats['gradient']
    np.testing.assert_allclose(residual,0,atol=1e-12)


def test_curvature_is_not_generally_diagonal_after_projection():
    raw=np.array([[2.,-1.,.5],[0.,1.,2.]])
    b=np.linalg.qr(np.array([[1.,1.],[2.,-1.],[.5,2.]]))[0]
    stats=leaf_statistics_reference(raw,np.array([0,2]),np.array([0,0]),b,'multiclass')
    assert abs(stats['hessian'][0,0,1])>1e-3
