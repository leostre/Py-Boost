import numpy as np
from quant_metric_core import projected_ridge_step

def test_nonorthogonal_basis_ridge_matches_full_decoded_objective():
    b=np.array([[2.,0.],[0.,.3],[.5,.4]]);hfull=np.diag([1.,3.,2.]);gfull=np.array([.2,-.7,1.])
    a=projected_ridge_step(b.T@gfull,b.T@hfull@b,b,2.)
    expected=np.linalg.lstsq(b.T@(hfull+2*np.eye(3))@b,-b.T@gfull,rcond=None)[0]
    np.testing.assert_allclose(a,expected,atol=1e-14)
    assert np.linalg.norm(a-np.linalg.solve(b.T@hfull@b+2*np.eye(2),-b.T@gfull))>.05

def test_orthogonal_basis_recovers_identity_regularizer():
    rng=np.random.default_rng(13);b=np.linalg.qr(rng.normal(size=(8,3)))[0];h=np.diag([2.,3.,4.]);g=np.array([1.,-2.,3.])
    np.testing.assert_allclose(projected_ridge_step(g,h,b),np.linalg.solve(h+np.eye(3),-g),atol=1e-14)
