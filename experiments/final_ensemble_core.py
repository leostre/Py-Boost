"""Same-role two-model convex score mixtures, never cross-fold ensembles."""
from dataclasses import dataclass,asdict
import itertools
import numpy as np
from experiments.uniform_threshold_core import threshold_variants,choice_key
from experiments.shared_leaf_basis import quality_guard


@dataclass(frozen=True)
class Component:
    method: str
    artifact: str
    sha256: str
    coo_bytes: int
    csr_bytes: int
    parameters: int

    def __post_init__(self):
        if (any(not isinstance(s,str) or not s for s in (self.method,self.artifact,self.sha256))
                or any(isinstance(v,bool) or not isinstance(v,int) or v<1
                       for v in (self.coo_bytes,self.csr_bytes,self.parameters))):
            raise ValueError('invalid ensemble component')


def feasible_pairs(components,outputs,budget):
    if (isinstance(outputs,bool) or not isinstance(outputs,int) or outputs<1
            or isinstance(budget,bool) or not isinstance(budget,int) or budget<0):
        raise ValueError('invalid ensemble budget domain')
    unique={}
    for c in components:
        prior=unique.setdefault(c.sha256,c)
        if (prior.coo_bytes,prior.csr_bytes,prior.parameters)!=(c.coo_bytes,c.csr_bytes,c.parameters):
            raise ValueError('same artifact has conflicting costs')
    extra=16+4*outputs
    return [(a,b,max(a.coo_bytes+b.coo_bytes,a.csr_bytes+b.csr_bytes)+extra)
            for a,b in itertools.combinations(unique.values(),2)
            if max(a.coo_bytes+b.coo_bytes,a.csr_bytes+b.csr_bytes)+extra<=budget]


def choose_mixture(components,calibration_raw,calibration_y,selection_raw,selection_y,task,outputs,budget,reference,config):
    rows=[]
    for left,right,cost in feasible_pairs(components,outputs,budget):
        a,b=left.sha256,right.sha256
        if max(np.max(np.abs(calibration_raw[a]-calibration_raw[b])),
               np.max(np.abs(selection_raw[a]-selection_raw[b])))<=1e-8:
            continue
        for weight in (.25,.5,.75):
            rc=weight*calibration_raw[a]+(1-weight)*calibration_raw[b]
            rs=weight*selection_raw[a]+(1-weight)*selection_raw[b]
            for variant in threshold_variants(rc,calibration_y,rs,selection_y,task,
                config['decision_grid'],config['decision_shrinkage']):
                rows.append(dict(candidate_id=f'{left.method}+{right.method}-w{weight}-t{variant["threshold_fraction"]}',
                    left=asdict(left),right=asdict(right),weight=weight,payload_bytes=cost,
                    parameters=left.parameters+right.parameters+outputs+2,**variant))
    allowed=[r for r in rows if quality_guard(r['validation'],reference)]
    selected=min(allowed or rows,key=choice_key) if rows else None
    return dict(state='accepted' if allowed else ('no_quality_feasible' if rows else 'no_distinct_memory_feasible_pair'),
                candidate=selected,candidates=rows)
