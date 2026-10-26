"""Training-only gradient output basis, with actual storage diagnostics."""
from dataclasses import dataclass
import numpy as np
from experiments.shared_leaf_basis import probabilities
from experiments.gradient_basis_core import (GradientBasisState,gradient_signal,contrast_matrix,
    basis_from_gram,projector_overlap,captured_energy)


def diagonal_derivatives(labels,scores,task):
    y,f=np.asarray(labels),np.asarray(scores,dtype=float)
    if f.ndim!=2 or min(f.shape)<2 or not np.isfinite(f).all():
        raise ValueError('invalid gradient scores')
    p=probabilities(f,task)
    if task=='multiclass':
        if y.shape!=(len(f),) or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=f.shape[1]):
            raise ValueError('invalid multiclass training target')
        g=p-np.eye(f.shape[1])[y]
        h=p*(1-p)*(f.shape[1]/(f.shape[1]-1))
    elif task=='multilabel':
        if y.shape!=f.shape or not np.isin(y,[0,1]).all():
            raise ValueError('invalid multilabel training target')
        g,h=p-y,p*(1-p)
    else: raise ValueError('unknown task')
    return g,h


@dataclass(frozen=True)
class BasisEstimate:
    basis: np.ndarray
    spectrum: np.ndarray
    diagnostics: dict


class AbsoluteHistory:
    def __init__(self,kind,outputs,task,beta=.8):
        self.state=GradientBasisState(kind,beta)
        self.previous=contrast_matrix(outputs,task).astype(np.float32)
        self.task=task

    def observe(self,labels,scores,regularization,rank):
        g,h=diagonal_derivatives(labels,scores,self.task)
        z=gradient_signal(g,h,regularization,self.task)
        prior=self.previous
        if not 1<=rank<=prior.shape[1]: raise ValueError('invalid diagnostic rank')
        previous_future_capture=captured_energy(z,prior[:,:rank]) if self.state.observations else None
        self.state.observe(z)
        gram,halves=self.state.matrices()
        b,info=basis_from_gram(gram,self.task,prior.shape[1],prior)
        first,_=basis_from_gram(halves[0],self.task,rank,prior[:,:rank])
        second,_=basis_from_gram(halves[1],self.task,rank,prior[:,:rank])
        values=np.array(info['eigenvalues'])
        algorithm=sum(a.nbytes for a in (self.state.mean,self.state.gram) if a is not None)
        diagnostic=sum(a.nbytes for a in (self.state.half_grams or []))
        result=dict(kind=self.state.kind,observations=self.state.observations,status=info['status'],
            diagnostic_rank=rank,algorithm_state_bytes=algorithm,diagnostic_state_bytes=diagnostic,
            previous_basis_bytes=prior.nbytes,training_signal_bytes=z.nbytes,
            current_capture=captured_energy(z,b[:,:rank]),future_capture_of_previous=previous_future_capture,
            temporal_overlap=projector_overlap(prior[:,:rank],b[:,:rank]),
            same_step_split_overlap=projector_overlap(first,second),
            spectrum_retained=float(values[:rank].sum()/values.sum()) if values.sum()>0 else None)
        self.previous=b.copy()
        return BasisEstimate(b.copy(),np.sqrt(np.maximum(values,0)),result)
