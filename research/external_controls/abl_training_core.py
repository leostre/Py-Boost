"""Pure training allocations and calibration-aware local fidelity checks."""
import math

import numpy as np

from experiments.cumulative_budget_core import AbsoluteAllocation,serving_cost
from calibration_pilot_core import fit_controls,control_for,POLICIES
from budget_quality_core import quality


def reserved_depth(topology,remaining,outputs,minimum_rank,budget):
    from experiments.closed_budget_core import extend_bound
    if remaining<1:raise ValueError('positive remaining horizon required')
    for depth in (5,3):
        future=extend_bound(topology,remaining,depth)
        if serving_cost(future,outputs,minimum_rank,0).maximum_bytes<=budget:return depth
    return None


def training_allocations(topology,outputs,ceiling,budget,parent_parameters,*,fixed=False):
    """Keep the established cost envelope; cap residual by its actual parent."""
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in
            (outputs,ceiling,budget,parent_parameters)) or ceiling>outputs:
        raise ValueError('positive integer allocation contract required')
    maximum=max((r for r in range(1,ceiling+1)
        if serving_cost(topology,outputs,r,0).maximum_bytes<=budget),default=0)
    if maximum==0:return ()
    ranks=[max(1,ceiling//2)] if fixed else sorted({max(1,math.floor(maximum*f)) for f in (.5,.75,1.)})
    result=[]
    for rank in ranks:
        if rank>maximum:continue
        base=serving_cost(topology,outputs,rank,0).maximum_bytes
        capacity=max(0,min((budget-base)//12,
            (budget-base-4*(topology.leaves+1))//8,topology.leaves*outputs))
        seen=set()
        for percent in ((5,) if fixed else (0,1,2,5)):
            count=min(capacity,percent*parent_parameters//100)
            if count in seen:continue
            seen.add(count);cost=serving_cost(topology,outputs,rank,count)
            if cost.maximum_bytes>budget:raise AssertionError('allocation exceeded its byte envelope')
            result.append(AbsoluteAllocation(rank,count,percent,cost))
    return tuple(result)


def calibrated_local_variants(cal_raw,cal_y,control_raw,control_y,task,grid,shrinkages):
    """Fit on calibration only, evaluate control; no final selection/test data.

    Calibration is an output policy; training derivatives still use unscaled
    raw AB+L scores. The persistent float32 T is priced outside this envelope.
    """
    if task=='multilabel':
        from experiments.uniform_threshold_core import threshold_variants
        return [dict(v,temperature=1.,policy='multilabel_threshold') for v in
            threshold_variants(cal_raw,cal_y,control_raw,control_y,task,grid,shrinkages)]
    if task!='multiclass':raise ValueError('unknown training task')
    ids=tuple(range(len(cal_y)));fitted=fit_controls(cal_raw,cal_y,ids,ids,grid)
    variants=[]
    for policy in POLICIES:
        control=control_for(policy,fitted)
        variants.append(dict(threshold_fraction=policy.shift_fraction,temperature=control.temperature,
            policy=policy.name,decision_shifts=list(control.shifts),
            validation=quality(control_y,control_raw,control.temperature,control.shifts)))
    return variants
