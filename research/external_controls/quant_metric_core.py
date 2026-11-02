"""Decoded-space ridge remains B.T B after finite-precision basis rounding."""
import numpy as np

def projected_ridge_step(gradient,hessian,basis,penalty=1.):
    b=np.asarray(basis,float);g=np.asarray(gradient,float);h=np.asarray(hessian,float)
    if b.ndim!=2 or g.shape!=(b.shape[1],) or h.shape!=(b.shape[1],b.shape[1]) or penalty<=0:raise ValueError('invalid projected quadratic')
    return np.linalg.solve(h+penalty*(b.T@b),-g)
