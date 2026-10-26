"""Bounded-workspace fixed-support diagonal G/H Newton leaf solve.

This is algebraically the existing weighted projection, not a new quality
algorithm. Support and basis are supplied by the caller; neither is optimized.
"""
import numpy as np
from experiments.online_leaf_core import LeafProjection


def direct_newton(gradient, diagonal, basis, learning_rate, rows=(), columns=(), block_size=8):
    g,d,b=(np.asarray(a,dtype=float) for a in (gradient,diagonal,basis))
    original_rows,original_cols=np.asarray(rows),np.asarray(columns)
    if (g.ndim!=2 or min(g.shape)<1 or d.shape!=g.shape or b.ndim!=2
            or b.shape[0]!=g.shape[1] or not 1<=b.shape[1]<=b.shape[0]
            or not all(np.isfinite(a).all() for a in (g,d,b)) or np.any(d<=0)
            or not np.allclose(b.T@b,np.eye(b.shape[1]),atol=2e-5)
            or not np.isfinite(learning_rate) or not 0<learning_rate<=1
            or isinstance(block_size,bool) or not isinstance(block_size,(int,np.integer)) or block_size<1
            or original_rows.ndim!=1 or original_rows.shape!=original_cols.shape
            or (original_rows.size and (original_rows.dtype.kind not in 'iu' or original_cols.dtype.kind not in 'iu'))):
        raise ValueError('invalid direct Newton domain')
    rr,cc=original_rows.astype(np.int64),original_cols.astype(np.int64)
    if (np.any(rr<0) or np.any(rr>=len(g)) or np.any(cc<0) or np.any(cc>=g.shape[1])
            or len(np.unique(rr*g.shape[1]+cc))!=len(rr)):
        raise ValueError('invalid fixed sparse support')
    order=np.argsort(rr*g.shape[1]+cc,kind='stable')
    rr,cc=rr[order],cc[order]
    coefficients=np.empty((len(g),b.shape[1]),np.float32)
    for start in range(0,len(g),block_size):
        stop=min(start+block_size,len(g))
        dd,gg=d[start:stop].copy(),g[start:stop].copy()
        mask=(rr>=start)&(rr<stop)
        local=rr[mask]-start
        dd[local,cc[mask]]=0
        gg[local,cc[mask]]=0
        lhs=np.einsum('cr,lc,cs->lrs',b,dd,b,optimize=True)
        rhs=-learning_rate*gg@b
        eigenvalues,vectors=np.linalg.eigh(lhs)
        cutoff=1e-12*np.maximum(eigenvalues[:,-1:],np.finfo(float).tiny)
        inverse=np.divide(1.,eigenvalues,out=np.zeros_like(eigenvalues),where=eigenvalues>cutoff)
        coordinates=np.einsum('lri,lr->li',vectors,rhs)*inverse
        coefficients[start:stop]=np.einsum('lri,li->lr',vectors,coordinates).astype(np.float32)
    values=np.empty(len(rr),np.float32)
    error=0.
    # Decode only a block. No complete leaf-by-output target or decoded array.
    for start in range(0,len(g),block_size):
        stop=min(start+block_size,len(g))
        decoded=coefficients[start:stop].astype(float)@b.T
        mask=(rr>=start)&(rr<stop)
        lr,lc=rr[mask]-start,cc[mask]
        values[mask]=(-learning_rate*g[rr[mask],lc]/d[rr[mask],lc]-decoded[lr,lc]).astype(np.float32)
        decoded[lr,lc]+=values[mask]
        difference=-learning_rate*g[start:stop]/d[start:stop]-decoded
        error+=float(np.sum(d[start:stop]*difference**2))
    keep=values!=0
    return LeafProjection(coefficients,rr[keep].astype(np.int32),cc[keep].astype(np.int32),values[keep],error)
