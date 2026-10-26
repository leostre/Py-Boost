"""Typed pure decisions for absolute COO/CSR budgets and cumulative error."""
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from experiments.closed_budget_core import Topology,compact_cost,extend_bound,integer
from experiments.sparse_leaf_core import contrast
from experiments.expanded_budget_data import validate_roles


@dataclass(frozen=True)
class ServingCost:
    parameters: int
    coo_bytes: int
    csr_bytes: int

    @property
    def maximum_bytes(self):
        return max(self.coo_bytes,self.csr_bytes)


def serving_cost(topology,outputs,rank,nnz):
    coo=compact_cost(topology,outputs,rank,nnz)
    csr=coo.payload_bytes-4*nnz+(4*(topology.leaves+1) if nnz else 0)
    return ServingCost(coo.parameters,coo.payload_bytes,csr)


@dataclass(frozen=True)
class AbsoluteAllocation:
    rank: int
    nnz: int
    cap_percent: int
    cost: ServingCost


def absolute_allocations(topology,outputs,ceiling,budget,*,fixed=False):
    budget=integer(budget,'absolute byte budget',1)
    integer(ceiling,'rank ceiling',1)
    if ceiling>outputs: raise ValueError('rank ceiling exceeds outputs')
    maximum=max((r for r in range(1,ceiling+1)
                 if serving_cost(topology,outputs,r,0).maximum_bytes<=budget),default=0)
    if maximum==0: return ()
    ranks=[max(1,ceiling//2)] if fixed else sorted({max(1,math.floor(maximum*f)) for f in (.5,.75,1.)})
    result=[]
    for rank in ranks:
        if rank>maximum: continue
        base=serving_cost(topology,outputs,rank,0).maximum_bytes
        # Any nonempty CSR must pay its leaf offset vector before entries.
        capacity=max(0,min((budget-base)//12,
                          (budget-base-4*(topology.leaves+1))//8,topology.leaves*outputs))
        seen=set()
        for percent in ((5,) if fixed else (0,1,2,5)):
            count=min(capacity,percent*(budget//4)//100)
            if count in seen: continue
            seen.add(count)
            cost=serving_cost(topology,outputs,rank,count)
            if cost.maximum_bytes>budget: raise AssertionError('incorrect absolute allocation')
            result.append(AbsoluteAllocation(rank,count,percent,cost))
    return tuple(result)


def absolute_depth(topology,remaining,outputs,rank,nnz,budget):
    for depth in (5,4,3):
        future=extend_bound(topology,remaining,depth)
        if serving_cost(future,outputs,rank,min(nnz,future.leaves*outputs)).maximum_bytes<=budget:
            return depth
    return None


@dataclass(frozen=True)
class CumulativeError:
    values: np.ndarray
    task: str
    updates: int=0

    def __post_init__(self):
        a=np.asarray(self.values,dtype=np.float64)
        if a.ndim!=2 or not np.isfinite(a).all() or self.task not in ('multilabel','multiclass'):
            raise ValueError('invalid cumulative error state')
        if self.updates<0: raise ValueError('negative observation count')
        # A caller cannot mutate a previous checkpoint through NumPy aliases.
        object.__setattr__(self,'values',np.frombuffer(a.tobytes(),dtype=np.float64).reshape(a.shape))

    def candidate_error(self,before,after):
        before,after=np.asarray(before,float),np.asarray(after,float)
        if before.shape!=self.values.shape or after.shape!=self.values.shape:
            raise ValueError('error axes changed')
        return self.values+contrast(before-after,self.task)

    def advance(self,before,after):
        return CumulativeError(self.candidate_error(before,after),self.task,self.updates+1)


class Criterion(str,Enum):
    LOCAL='local'
    CUMULATIVE='cumulative'


@dataclass(frozen=True)
class Quality:
    loss: float
    macro: float
    weighted: float

    @classmethod
    def from_row(cls,row):
        result=cls(float(row['loss']),float(row['f1_macro']),float(row['f1_weighted']))
        if not all(math.isfinite(v) and v>=0 for v in (result.loss,result.macro,result.weighted)):
            raise ValueError('invalid candidate quality')
        return result

    def within(self,reference):
        return (self.loss<=1.05*reference.loss and self.macro>=.95*reference.macro
                and self.weighted>=.95*reference.weighted)


@dataclass(frozen=True)
class CandidateValue:
    index: int
    identifier: str
    quality: Quality
    error: float
    cost: ServingCost

    def __post_init__(self):
        if self.index<0 or not math.isfinite(self.error) or self.error<0:
            raise ValueError('invalid candidate index or cumulative error')


@dataclass(frozen=True)
class Accept:
    index: int


@dataclass(frozen=True)
class Reject:
    reason: str
    candidates: int


def decide_candidate(candidates,reference,budget,criterion):
    criterion=Criterion(criterion)
    in_budget=[c for c in candidates if c.cost.maximum_bytes<=budget]
    valid=[c for c in in_budget if c.quality.within(reference)]
    if not valid:
        return Reject('no_memory_feasible' if not in_budget else 'local_quality_rejected',len(candidates))
    def key(c):
        local=(-c.quality.macro,c.quality.loss,c.cost.maximum_bytes,c.identifier)
        return ((c.error,)+local) if criterion==Criterion.CUMULATIVE else local
    return Accept(min(valid,key=key).index)


def widen_roles(training_labels,original,outputs,task,seed):
    """Only old training labels enter this function; test labels cannot enter."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    labels=np.asarray(training_labels)
    if len(labels)!=len(original['train']): raise ValueError('old training axis mismatch')
    y=np.eye(outputs)[labels.astype(int)] if task=='multiclass' else labels
    if y.shape!=(len(labels),outputs) or not np.isin(y,[0,1]).all():
        raise ValueError('invalid training labels for role widening')
    count=sum(len(v) for v in original.values())
    validate_roles(original,count)
    keep,removed=next(MultilabelStratifiedShuffleSplit(n_splits=1,test_size=.25,
                     random_state=seed+310000).split(np.zeros((len(y),1)),y))
    for_control,for_selection=next(MultilabelStratifiedShuffleSplit(n_splits=1,test_size=2/3,
                     random_state=seed+310001).split(np.zeros((len(removed),1)),y[removed]))
    old_train=np.asarray(original['train'])
    result={k:np.asarray(v).copy() for k,v in original.items()}
    result['train']=old_train[keep].copy()
    result['control']=np.r_[result['control'],old_train[removed[for_control]]]
    result['selection']=np.r_[result['selection'],old_train[removed[for_selection]]]
    validate_roles(result,count)
    return result
