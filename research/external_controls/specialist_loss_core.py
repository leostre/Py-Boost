"""Fixed-score loss attribution; no parameter selection or calibration."""
import math
import numpy as np
import external_quality_core
from experiments.shared_leaf_basis import probabilities
from experiments.compactness_core import normalize_multiclass_probabilities
from experiments.output_calibration_core import labels_from_raw


def object_losses(raw,target):
    raw=np.asarray(raw,dtype=np.float64)
    target=np.asarray(target)
    if (raw.ndim!=2 or len(raw)==0 or raw.shape[1]<2 or target.shape!=(len(raw),)
            or not np.issubdtype(target.dtype,np.integer) or not np.isfinite(raw).all()
            or np.any(target<0) or np.any(target>=raw.shape[1])):
        raise ValueError('invalid frozen multiclass scores or targets')
    p=normalize_multiclass_probabilities(probabilities(raw,'multiclass'))
    return -np.log(np.clip(p,1e-7,1-1e-7)[np.arange(len(target)),target])


def decompose(target,specialist,reference,specialist_shifts,reference_shifts):
    target=np.asarray(target)
    if np.shape(specialist)!=np.shape(reference): raise ValueError('score domains differ')
    a,b=object_losses(specialist,target),object_losses(reference,target)
    for shift in (specialist_shifts,reference_shifts):
        if np.asarray(shift).shape not in ((0,),(np.shape(specialist)[1],)) or not np.isfinite(shift).all():
            raise ValueError('invalid frozen decision shifts')
    ac=labels_from_raw(specialist,'multiclass',np.asarray(specialist_shifts))==target
    bc=labels_from_raw(reference,'multiclass',np.asarray(reference_shifts))==target
    delta=a-b
    groups=[]
    for name,mask in (('both_correct',ac&bc),('specialist_only_error',~ac&bc),
                      ('reference_only_error',ac&~bc),('both_wrong',~ac&~bc)):
        groups.append(dict(group=name,count=int(mask.sum()),
            specialist_loss_sum=float(a[mask].sum()),reference_loss_sum=float(b[mask].sum()),
            excess_loss_sum=float(delta[mask].sum()),
            contribution_to_mean_excess=float(delta[mask].sum()/len(target))))
    positive=np.maximum(delta,0)
    top=np.argsort(-positive,kind='stable')[:max(1,math.ceil(.1*len(target)))]
    classes=[]
    for label in range(np.shape(specialist)[1]):
        mask=target==label
        classes.append(dict(label=label,support=int(mask.sum()),
            specialist_loss_sum=float(a[mask].sum()),reference_loss_sum=float(b[mask].sum()),
            excess_loss_sum=float(delta[mask].sum())))
    return dict(objects=len(target),specialist_mean_loss=float(a.mean()),reference_mean_loss=float(b.mean()),
        mean_excess=float(delta.mean()),specialist_errors=int((~ac).sum()),reference_errors=int((~bc).sum()),
        groups=groups,classes=classes,top_tenth_count=len(top),
        top_tenth_share_of_positive_excess=float(positive[top].sum()/positive.sum()) if positive.sum() else 0.,
        positive_excess_sum=float(positive.sum()),negative_excess_sum=float(np.minimum(delta,0).sum())),a,b
