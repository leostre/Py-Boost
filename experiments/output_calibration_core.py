"""CPU output-calibration controls with exact persistent storage accounting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.metrics import f1_score

from experiments.shared_leaf_basis import LeafEnsemble, probabilities
from experiments.sparse_leaf_core import SparseLeafModel, contrast, leaf_design


def indicator(target,c,task):
    y=np.eye(c)[np.asarray(target,dtype=int)] if task=='multiclass' else np.asarray(target,dtype=float)
    if y.ndim!=2 or y.shape[1]!=c or not np.isin(y,[0,1]).all():
        raise ValueError('invalid calibration labels')
    return y


def calibration_weights(target,c,task,balanced):
    y=indicator(target,c,task);n=len(y)
    if task=='multiclass':
        count=y.sum(axis=0)
        w=1/np.maximum(count[np.asarray(target,dtype=int)],1) if balanced else np.ones(n)
        return w/w.sum()
    if balanced:
        count=y.sum(axis=0)
        w=y/np.maximum(count,1)+(1-y)/np.maximum(n-count,1)
    else:
        w=np.ones_like(y)
    return w/w.sum(axis=0,keepdims=True)/c


def affine_objective(theta,raw,y,weights,task,regularization):
    c=raw.shape[1];a,b=theta[:c],theta[c:]
    z=raw*a+b
    if task=='multiclass':
        norm=logsumexp(z,axis=1,keepdims=True)
        residual=(np.exp(z-norm)-y)*weights[:,None]
        objective=np.sum(weights*(norm.ravel()-np.sum(y*z,axis=1)))
    else:
        objective=np.sum(weights*(np.logaddexp(0,z)-y*z))
        residual=(expit(z)-y)*weights
    objective+=regularization/(2*c)*(np.sum((a-1)**2)+np.sum(b**2))
    gradient=np.r_[np.sum(residual*raw,axis=0)+regularization/c*(a-1),
                   np.sum(residual,axis=0)+regularization/c*b]
    return float(objective),gradient


def fit_affine(raw,target,task,balanced,regularization):
    if regularization<=0:
        raise ValueError('positive calibration regularization required')
    raw=contrast(raw,task);c=raw.shape[1]
    y=indicator(target,c,task)
    if len(y)!=len(raw) or not np.isfinite(raw).all():
        raise ValueError('invalid calibration input')
    weights=calibration_weights(target,c,task,balanced)
    initial=np.r_[np.ones(c),np.zeros(c)]
    result=minimize(affine_objective,initial,args=(raw,y,weights,task,regularization),
        jac=True,method='L-BFGS-B',bounds=[(.25,4.)]*c+[(-4.,4.)]*c,
        options={'maxiter':150,'ftol':1e-10,'gtol':1e-6})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError('nonfinite affine fit')
    return {'scale':result.x[:c].tolist(),'shift':result.x[c:].tolist(),
            'converged':bool(result.success),'iterations':int(result.nit),
            'initial_objective':affine_objective(initial,raw,y,weights,task,regularization)[0],
            'final_objective':float(result.fun)}


def fold_affine(base,scale,shift):
    """Absorb affine parameters into A/B/bias; no redundant persistent copy.

    QR restores the orthonormal decoder contract. Multiclass common shifts are
    removed both before and after the affine transform, preserving softmax.
    """
    c=len(base.bias);a=np.asarray(scale,dtype=float);b=np.asarray(shift,dtype=float)
    if a.shape!=(c,) or b.shape!=(c,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<=0):
        raise ValueError('invalid affine transform')
    decoder=base.basis.astype(float)
    bias=base.bias.astype(float)
    if base.task=='multiclass':
        decoder=decoder-decoder.mean(axis=0,keepdims=True)
        bias=bias-bias.mean()
    decoder=decoder*a[:,None];bias=bias*a+b
    if base.task=='multiclass':
        decoder=decoder-decoder.mean(axis=0,keepdims=True)
        bias=bias-bias.mean()
    q,r=np.linalg.qr(decoder,mode='reduced')
    return LeafEnsemble(base.nodes,base.node_offsets,base.leaf_offsets,
        (base.coefficients.astype(float)@r.T).astype(np.float32),q.astype(np.float32),
        bias.astype(np.float32),base.n_features,base.task,'low_rank')


def fit_decision_shifts(raw,target,task,grid):
    """Fit discrete decisions on calibration rows; do not modify probabilities.

    Multiclass per-class one-vs-rest fits are a bounded heuristic, not a global
    optimum of multiclass macro-F1. The selection fold judges the joint result.
    """
    z=contrast(raw,task);c=z.shape[1];truth=indicator(target,c,task).astype(bool)
    if task=='multiclass':
        best=z.argmax(axis=1);second=np.partition(z,-2,axis=1)[:,-2]
        competitor=np.broadcast_to(z.max(axis=1)[:,None],z.shape).copy()
        competitor[np.arange(len(z)),best]=second
        z=z-competitor
    candidates=sorted(set(float(t) for t in grid),key=lambda t:(abs(t),-t))
    if 0. not in candidates:
        raise ValueError('decision grid must include zero')
    result=np.zeros(c,dtype=np.float32)
    for j in range(c):
        if truth[:,j].sum()==0:
            continue
        scored=[]
        for threshold in candidates:
            chosen=z[:,j]>=threshold
            tp=np.sum(chosen&truth[:,j]);fp=np.sum(chosen&~truth[:,j]);fn=np.sum(~chosen&truth[:,j])
            den=2*tp+fp+fn
            scored.append(2*tp/den if den else 0.)
        result[j]=candidates[int(np.argmax(scored))]
    return result


def labels_from_raw(raw,task,thresholds):
    shifted=raw-thresholds if len(thresholds) else raw
    return shifted.argmax(axis=1) if task=='multiclass' else shifted>=0


def calibrated_quality(target,raw,task,thresholds):
    # Keep the inherited probability/loss definitions; override decisions only.
    from experiments.a100_runner import quality
    q=quality(target,probabilities(raw,task),task)
    if len(thresholds):
        labels=labels_from_raw(raw,task,thresholds)
        q['f1_macro']=float(f1_score(target,labels,average='macro',zero_division=0))
        q['f1_weighted']=float(f1_score(target,labels,average='weighted',zero_division=0))
    return q


@dataclass(frozen=True)
class DecisionLeafModel:
    leaf: SparseLeafModel
    decision_shifts: np.ndarray

    def __post_init__(self):
        a=self.decision_shifts
        if a.dtype!=np.float32 or a.ndim!=1 or len(a) not in (0,len(self.leaf.base.bias)) or not np.isfinite(a).all():
            raise ValueError('invalid decision shifts')

    @property
    def parameter_count(self):
        return self.leaf.parameter_count+len(self.decision_shifts)

    @property
    def payload_bytes(self):
        return self.leaf.payload_bytes+self.decision_shifts.nbytes

    def raw_from_design(self,design):
        return self.leaf.raw_from_design(design)

    def predict(self,features):
        return probabilities(self.raw_from_design(leaf_design(self.leaf.base.leaf_indices(features),
                                                              len(self.leaf.base.coefficients))),self.leaf.base.task)

    def predict_labels(self,features):
        raw=self.raw_from_design(leaf_design(self.leaf.base.leaf_indices(features),len(self.leaf.base.coefficients)))
        return labels_from_raw(raw,self.leaf.base.task,self.decision_shifts)

    def save(self,path):
        base=self.leaf.base
        names=('nodes','node_offsets','leaf_offsets','coefficients','basis','bias')
        np.savez(path,version=np.array(3),**{n:getattr(base,n) for n in names},
            n_features=np.array(base.n_features),task=np.array(base.task),
            rows=self.leaf.rows,columns=self.leaf.columns,values=self.leaf.values,decision_shifts=self.decision_shifts)

    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as data:
            names=('nodes','node_offsets','leaf_offsets','coefficients','basis','bias')
            expected={*names,'version','n_features','task','rows','columns','values','decision_shifts'}
            if set(data.files)!=expected or int(data['version'])!=3:
                raise ValueError('unknown calibrated artifact schema')
            base=LeafEnsemble(**{n:data[n].copy() for n in names},n_features=int(data['n_features']),
                task=str(data['task']),representation='low_rank')
            leaf=SparseLeafModel(base,data['rows'].copy(),data['columns'].copy(),data['values'].copy())
            return cls(leaf,data['decision_shifts'].copy())
