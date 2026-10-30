"""Exact projected curvature identities and leaf Newton solve; no GPU imports."""
import numpy as np
from scipy.special import expit,softmax


def derivatives(raw,y,task):
    z=np.asarray(raw,float);y=np.asarray(y);n,c=z.shape
    if task=='multiclass':
        p=softmax(z,axis=1);g=p-np.eye(c)[y.astype(int)]
        h=np.array([np.diag(v)-np.outer(v,v) for v in p])
    elif task=='multilabel':
        p=expit(z);g=p-y;h=np.array([np.diag(v*(1-v)) for v in p])
    else:raise ValueError('unknown task')
    return g,h


def leaf_statistics_reference(raw,y,leaves,basis,task):
    """Small independent dense-Hessian oracle for tests, never large training."""
    g,h=derivatives(raw,y,task);b=np.asarray(basis,float);leaves=np.asarray(leaves,int)
    if leaves.shape!=(len(g),) or leaves.min()<0:raise ValueError('leaf domain')
    m=int(leaves.max())+1;c=g.shape[1];r=b.shape[1]
    gradient=np.zeros((m,c));hessian=np.zeros((m,c,c))
    for i,leaf in enumerate(leaves):gradient[leaf]+=g[i];hessian[leaf]+=h[i]
    cross=hessian@b;reduced=np.einsum('cr,lcs->lrs',b,cross)
    return dict(full_gradient=gradient,diagonal=np.diagonal(hessian,axis1=1,axis2=2).copy(),
        cross=cross,gradient=gradient@b,hessian=reduced,full_hessian=hessian)


def coupled_step(gradient,hessian,regularization):
    g,h=np.asarray(gradient,float),np.asarray(hessian,float)
    if regularization<=0 or h.shape!=(*g.shape,g.shape[-1]) or not np.isfinite(h).all() or not np.isfinite(g).all():raise ValueError('finite SPD regularized domain required')
    return np.linalg.solve(h+regularization*np.eye(g.shape[-1]),-g[...,None])[...,0]


def diagonal_latent_curvature(probability,basis,task):
    p,b=np.asarray(probability,float),np.asarray(basis,float)
    if task=='multiclass':return p@(b*b)-(p@b)**2
    if task=='multilabel':return (p*(1-p))@(b*b)
    raise ValueError('unknown task')
