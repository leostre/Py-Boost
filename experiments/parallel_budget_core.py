"""Two-branch plans, zero-bias averaging and whole-function union basis."""
from dataclasses import dataclass,replace
import numpy as np
from experiments.parallel_leaf_core import dense_parts
from experiments.online_leaf_core import stitch
from experiments.shared_leaf_basis import functional_basis
from experiments.gradient_basis_core import projector_overlap
from experiments.history_basis_study import raw

VARIANTS=('par_serial','par_half','parallel2_average','parallel2_union')


@dataclass(frozen=True)
class BranchPlan:
    trees:int
    lr:float
    seed_offset:int
    sample:float


def branch_plans(method,block):
    if method not in VARIANTS or block not in range(5): raise ValueError('invalid parallel plan')
    if block==0: return (BranchPlan(40,.1,0,1.),)
    if method.startswith('parallel2_'): return (BranchPlan(20,.1,0,.8),BranchPlan(20,.1,1,.8))
    return (BranchPlan(40,.05 if method=='par_half' else .1,0,.8),)


def mean_increment(increments):
    if not increments: raise ValueError('no branch increments')
    first=increments[0].base
    parts=[]
    for model in increments:
        if (np.any(model.base.bias) or np.any(model.decision_shifts)
                or model.base.task!=first.task or model.base.n_features!=first.n_features):
            raise ValueError('invalid zero-bias branch')
        parts.extend(replace(p,coefficients=(p.coefficients/len(increments)).astype(np.float32)) for p in dense_parts(model))
    return stitch(parts,None,np.zeros_like(first.bias),first.n_features,first.task)


@dataclass(frozen=True)
class UnionBasis:
    basis:np.ndarray
    spectrum:np.ndarray
    diagnostics:dict


def union_estimate(candidates,increments,features):
    if len(candidates)!=2 or len(increments)!=2: raise ValueError('union estimate requires two branches')
    task=candidates[0].base.task
    functions=[raw(m,features) for m in candidates]
    centered=np.concatenate([f-f.mean(axis=0,keepdims=True) for f in functions])
    _,basis,spectrum=functional_basis(centered,task)
    r=max(1,basis.shape[1]//2)
    local=[functional_basis(raw(m,features),task)[1][:,:r] for m in increments]
    return UnionBasis(basis,spectrum,dict(kind='parallel_union',branches=2,overlap_rank=r,
        increment_subspace_overlap=float(projector_overlap(local[0],local[1])),
        temporary_function_stack_bytes=int(centered.nbytes),source='centered F+delta0 and F+delta1, common function paid once'))
