"""Explicit EMA first/second moments of aligned, curvature-scaled gradients."""
from dataclasses import dataclass
import numpy as np


def contrast_matrix(outputs,task):
    if outputs < 2 or task not in ('multilabel','multiclass'):
        raise ValueError('invalid gradient output space')
    return (np.linalg.qr(np.vstack((np.eye(outputs-1),-np.ones(outputs-1))))[0]
            if task == 'multiclass' else np.eye(outputs))


def gradient_signal(grad,hess,regularization,task):
    g,h = np.asarray(grad,dtype=float),np.asarray(hess,dtype=float)
    if (g.ndim != 2 or h.shape != g.shape or min(g.shape) < 2
            or not np.isfinite(g).all() or not np.isfinite(h).all()
            or np.any(h < 0) or regularization <= 0):
        raise ValueError('invalid gradient/curvature snapshot')
    z = -g/np.sqrt(h+regularization/len(g))
    if task == 'multiclass':
        z -= z.mean(axis=1,keepdims=True)
    elif task != 'multilabel':
        raise ValueError('invalid task')
    norm = np.linalg.norm(z)
    if not np.isfinite(norm) or not np.isfinite(z).all():
        raise ValueError('nonfinite scaled gradient signal')
    return z/norm if norm > 0 else np.zeros_like(z)


def projector_overlap(a,b):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('equal-rank bases required')
    return float(np.clip(np.linalg.norm(a.T@b,'fro')**2/a.shape[1],0.,1.))


def captured_energy(signal,basis):
    z,b = np.asarray(signal,dtype=float),np.asarray(basis,dtype=float)
    energy = float(np.sum(z*z))
    return float(np.clip(np.sum((z@b)**2)/energy,0.,1.)) if energy > 0 else None


def basis_from_gram(gram,task,rank,previous):
    c = len(gram)
    k = contrast_matrix(c,task)
    if (not 1 <= rank <= k.shape[1] or previous.shape != (c,rank)
            or not np.isfinite(previous).all() or not np.allclose(previous.T@previous,np.eye(rank),atol=2e-5)):
        raise ValueError('invalid gradient basis rank')
    covariance = np.asarray(gram,dtype=float)
    if covariance.shape != (c,c) or not np.isfinite(covariance).all():
        raise ValueError('invalid gradient Gram matrix')
    projected = k.T@((covariance+covariance.T)/2)@k
    eigenvalues,vectors = np.linalg.eigh(projected)
    if eigenvalues[0] < -1e-10*max(1.,abs(eigenvalues[-1])):
        raise ValueError('Gram matrix is not positive semidefinite')
    values = np.maximum(eigenvalues[::-1],0.)
    energy = float(values.sum())
    if energy <= 1e-24:
        return previous.copy(),dict(status='zero_signal_keep_previous',retained_energy=None,eigenvalues=values.tolist())
    basis = k@vectors[:,::-1][:,:rank]
    pivots = np.argmax(np.abs(basis),axis=0)
    signs = np.where(basis[pivots,np.arange(rank)] >= 0,1.,-1.)
    basis = (basis*signs).astype(np.float32)
    return basis,dict(status='updated',retained_energy=float(values[:rank].sum()/energy),
        cutoff_gap_ratio=float(values[rank-1]/max(values[rank],1e-30)) if rank < len(values) else None,
        eigenvalues=values.tolist())


@dataclass
class GradientBasisState:
    kind: str
    beta: float = .8

    def __post_init__(self):
        if self.kind not in ('current','mean','history') or not 0 <= self.beta < 1:
            raise ValueError('invalid gradient history estimator')
        self.mass,self.observations = 0.,0
        self.mean,self.gram,self.half_grams = None,None,None
        self.shape = None

    def observe(self,signal):
        z = np.asarray(signal,dtype=float)
        if z.ndim != 2 or min(z.shape) < 2 or not np.isfinite(z).all():
            raise ValueError('invalid aligned gradient signal')
        if self.shape is not None and z.shape != self.shape:
            raise ValueError('object/output axes changed across history')
        self.shape = z.shape
        if self.kind == 'current':
            self.gram = z.T@z
            self.half_grams = [a.T@a for a in (z[::2],z[1::2])]
            self.mass = 1.
        elif self.kind == 'mean':
            self.mean = (np.zeros_like(z) if self.mean is None else self.beta*self.mean)+(1-self.beta)*z
            self.mass = self.beta*self.mass+(1-self.beta)
        else:
            gram = z.T@z
            halves = [a.T@a for a in (z[::2],z[1::2])]
            self.gram = (np.zeros_like(gram) if self.gram is None else self.beta*self.gram)+(1-self.beta)*gram
            self.half_grams = ([np.zeros_like(gram),np.zeros_like(gram)] if self.half_grams is None
                               else [self.beta*a for a in self.half_grams])
            self.half_grams = [a+(1-self.beta)*b for a,b in zip(self.half_grams,halves)]
            self.mass = self.beta*self.mass+(1-self.beta)
        self.observations += 1

    def matrices(self):
        if self.observations == 0:
            raise ValueError('history has no observations')
        if self.kind == 'mean':
            mean = self.mean/self.mass
            return mean.T@mean,[a.T@a for a in (mean[::2],mean[1::2])]
        return self.gram/self.mass,[a/self.mass for a in self.half_grams]

    def estimate(self,task,rank,previous):
        gram,halves = self.matrices()
        basis,diagnostics = basis_from_gram(gram,task,rank,previous)
        first,first_info = basis_from_gram(halves[0],task,rank,previous)
        second,second_info = basis_from_gram(halves[1],task,rank,previous)
        diagnostics.update(observations=self.observations,
            temporal_overlap=projector_overlap(previous,basis) if diagnostics['status'] == 'updated' else None,
            same_step_split_overlap=projector_overlap(first,second) if
                first_info['status'] == second_info['status'] == 'updated' else None,
            split_signal_status=[first_info['status'],second_info['status']],
            random_overlap_reference=rank/(len(gram)-(task == 'multiclass')),
            algorithm_state_bytes=self.gram.nbytes if self.kind == 'current' else
                (self.mean.nbytes if self.kind == 'mean' else self.gram.nbytes),
            diagnostic_state_bytes=sum(a.nbytes for a in (self.half_grams or [])))
        return basis,diagnostics
