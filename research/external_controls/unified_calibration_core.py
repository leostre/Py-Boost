"""Identical calibration policies without native budget/training imports."""
from calibration_pilot_core import fit_controls,control_for,POLICIES
from budget_quality_core import quality

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
