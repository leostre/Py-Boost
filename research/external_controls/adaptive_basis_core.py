"""Train-only risk rotations; numerical contracts independent of datasets/I/O."""
from dataclasses import replace
import numpy as np
import budget_quality_worker
from experiments.gradient_basis_core import contrast_matrix
from experiments.sparse_leaf_core import training_weights,coordinate_scores


def risk_weights(y,raw,task):
    raw=np.asarray(raw,float);y=np.asarray(y);n,c=raw.shape
    if task=='multiclass':
        if y.shape!=(n,) or not np.isin(y,np.arange(c)).all():raise ValueError('invalid classes')
        indicator=np.eye(c)[y.astype(int)]
    elif task=='multilabel' and y.shape==raw.shape and np.isin(y,[0,1]).all():indicator=y
    else:raise ValueError('invalid task/target')
    if not np.isfinite(raw).all():raise ValueError('nonfinite outputs')
    support=indicator.sum(0)
    columns=np.clip(np.sqrt(n/np.maximum(support,1)),1.,10.)
    columns/=columns.mean()
    w=training_weights(y,raw,task,'balanced_margin',.5)
    rows=np.clip(n*w.mean(1),.1,10.);rows/=rows.mean()
    return rows,columns


def risk_basis(contribution,task,rows,columns,max_rank=None):
    """Weighted functional PCA in the valid output contrast space.

    Metric is separable positive row/output importance, not a full Hessian
    claim and not a guarantee for F1. Whitening is undone before QR.
    """
    z=np.asarray(contribution,float);rows=np.asarray(rows,float);columns=np.asarray(columns,float)
    n,c=z.shape
    if rows.shape!=(n,) or columns.shape!=(c,) or np.any(rows<=0) or np.any(columns<=0):raise ValueError('positive matching weights required')
    if not all(np.isfinite(a).all() for a in (z,rows,columns)):raise ValueError('nonfinite geometry')
    if max_rank is not None and (not isinstance(max_rank, int) or not 1<=max_rank<=c-int(task=='multiclass')):
        raise ValueError('invalid truncated rank')
    center=(rows@z)/rows.sum()
    if task=='multilabel' and max_rank is not None:
        from scipy.linalg import eigh
        x=z-center;root=np.sqrt(columns)
        gram=(x.T@(rows[:,None]*x))*root[:,None]*root[None,:]
        eigen,vectors=eigh((gram+gram.T)/2,subset_by_index=(c-max_rank,c-1),check_finite=False)
        candidates=vectors[:,::-1]/root[:,None]
        q,r=np.linalg.qr(candidates);q*=np.where(np.diag(r)>=0,1.,-1.)
        return q.astype(np.float32),dict(weighted_eigenvalues=np.maximum(eigen[::-1],0).tolist(),
            spectral_trace=float(np.trace(gram)),truncated_rank=max_rank,
            row_weight_min=float(rows.min()),row_weight_max=float(rows.max()),
            column_weight_min=float(columns.min()),column_weight_max=float(columns.max()))
    k=contrast_matrix(c,task);x=(z-center)@k
    metric=k.T@(columns[:,None]*k);root=np.linalg.cholesky(metric)
    gram=root.T@(x.T@(rows[:,None]*x))@root
    eigen,vectors=np.linalg.eigh((gram+gram.T)/2)
    candidates=k@np.linalg.solve(root.T,vectors[:,::-1])
    q,r=np.linalg.qr(candidates);q*=np.where(np.diag(r)>=0,1.,-1.)
    if max_rank is not None:q=q[:,:max_rank]
    return q.astype(np.float32),dict(weighted_eigenvalues=np.maximum(eigen[::-1],0).tolist(),
        row_weight_min=float(rows.min()),row_weight_max=float(rows.max()),
        column_weight_min=float(columns.min()),column_weight_max=float(columns.max()))


def all_support_order(group,y,teacher,selector):
    residual=group.residual;frequency=group.frequency
    error=np.asarray(group.designs['train']@residual.astype(float))-frequency@residual.astype(float)
    weights=training_weights(y,teacher,group.view.task,selector,.5)
    score=coordinate_scores(group.designs['train'],residual,error,selector,weights)
    order=np.argsort(-score.ravel(),kind='stable')
    # Include zero-score coordinates to match a prescribed nonzero quota.
    # Their potential harm remains an observed point, never silently filtered.
    return order[residual.ravel()[order]!=0]
