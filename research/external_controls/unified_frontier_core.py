"""Ungated frontier and shared-data calibration. No files, model fits or test API."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import f1_score
import budget_quality_worker  # pinned native_source import boundary
from unified_calibration_core import calibrated_local_variants
from calibration_pilot_core import partition


@dataclass(frozen=True)
class Fold:
    fitting: tuple[int, ...]
    validation: tuple[int, ...]


def make_folds(y, task, seed, count=3):
    y=np.asarray(y);n=len(y)
    if count<2 or n<count:raise ValueError('at least one row per fold required')
    if task=='multiclass':
        held=partition(y,[n//count+int(i<n%count) for i in range(count)],seed)
    elif task=='multilabel':
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        splitter=MultilabelStratifiedKFold(n_splits=count,shuffle=True,random_state=seed)
        held=tuple(tuple(map(int,v)) for _,v in splitter.split(np.zeros((n,1)),y))
    else:raise ValueError('unknown task')
    all_ids=set(range(n))
    folds=tuple(Fold(tuple(sorted(all_ids-set(v))),tuple(v)) for v in held)
    validate_folds(folds,n)
    return folds


def validate_folds(folds,n,complete=True):
    held=[]
    for f in folds:
        a,b=set(f.fitting),set(f.validation)
        if not a or not b or a&b or len(a)!=len(f.fitting) or len(b)!=len(f.validation):
            raise ValueError('empty, duplicate or leaking fold')
        if min(a|b)<0 or max(a|b)>=n:raise ValueError('fold outside development')
        held.extend(f.validation)
    if len(held)!=len(set(held)) or (complete and set(held)!=set(range(n))):
        raise ValueError('validation is not an exact disjoint cover')


def sufficient(y,z,task,temperature,shifts):
    z=np.asarray(z,float);y=np.asarray(y);s=np.asarray(shifts,float)
    if z.ndim!=2 or len(y)!=len(z) or s.shape!=(z.shape[1],) or temperature<=0:
        raise ValueError('mismatched score domains')
    if not np.isfinite(z).all() or not np.isfinite(s).all():raise ValueError('nonfinite outputs')
    if task=='multiclass':
        p=np.clip(softmax(z/temperature,axis=1),1e-7,1-1e-7)
        pred=(z/temperature-s).argmax(1)
        loss=-np.log(p[np.arange(len(y)),y.astype(int)])
        brier=np.sum(p*p,axis=1)-2*p[np.arange(len(y)),y.astype(int)]+1
    elif task=='multilabel':
        if temperature!=1.:raise ValueError('this matched multilabel protocol retains T=1')
        p=np.clip(expit(z),1e-7,1-1e-7);pred=z-s>=0
        loss=np.mean(-y*np.log(p)-(1-y)*np.log1p(-p),axis=1)
        brier=np.mean((p-y)**2,axis=1)
    else:raise ValueError('unknown task')
    return pred,loss,brier


def metrics(y,pred,loss,brier,task,outputs):
    args={'labels':np.arange(outputs)} if task=='multiclass' else {}
    return dict(loss=float(np.mean(loss)),brier=float(np.mean(brier)),
        f1_macro=float(f1_score(y,pred,average='macro',zero_division=0,**args)),
        f1_weighted=float(f1_score(y,pred,average='weighted',zero_division=0,**args)))


def variants(fit_z,fit_y,validation_z,validation_y,task,grid,shrinkages):
    source=calibrated_local_variants(fit_z,fit_y,validation_z,validation_y,task,grid,shrinkages)
    result={}
    for v in source:
        name=(v['policy'] if task=='multiclass' else
              ('raw' if v['threshold_fraction']==0 else f"raw_shift_{v['threshold_fraction']:g}"))
        result[name]=dict(temperature=float(v['temperature']),shifts=list(map(float,v['decision_shifts'])))
    return result


def replay_calibration(z,y,task,folds,grid,shrinkages,complete=True):
    z=np.asarray(z,float);y=np.asarray(y);validate_folds(folds,len(y),complete)
    collected={};states=[];validation=[]
    for f in folds:
        a,b=np.asarray(f.fitting),np.asarray(f.validation)
        fitted=variants(z[a],y[a],z[b],y[b],task,grid,shrinkages)
        states.append(dict(fitting=list(f.fitting),validation=list(f.validation),policies=fitted))
        validation.extend(f.validation)
        for name,policy in fitted.items():
            collected.setdefault(name,[]).append(sufficient(y[b],z[b],task,policy['temperature'],policy['shifts']))
    arrays={};scores={};ids=np.asarray(validation,np.int64)
    for name,parts in collected.items():
        pred,loss,brier=[np.concatenate([p[k] for p in parts]) for k in range(3)]
        scores[name]=metrics(y[ids],pred,loss,brier,task,z.shape[1])
        arrays[name]=(pred,loss,brier)
    return dict(scores=scores,states=states,validation=ids,arrays=arrays)


def choose_policy(scores,objective):
    if objective not in ('loss','f1_macro'):raise ValueError('unknown objective')
    def key(item):
        name,q=item
        value=(q['loss'],) if objective=='loss' else (-q['f1_macro'],q['loss'])
        return (*value,int(name!='raw'),name)
    return min(scores.items(),key=key)[0]


def quality_change(value,reference):
    if reference<=0:return None
    return float(1-value/reference)


def frontier(rows,objective='f1_macro'):
    """Exact measured frontier; no 5% quality or 75% memory filter."""
    maximize=objective!='loss'
    if objective not in ('f1_macro','f1_weighted','loss'):raise ValueError('unknown frontier')
    ordered=sorted(rows,key=lambda r:(r['cost_bytes'], -r['quality'][objective] if maximize else r['quality'][objective],r['uid']))
    keep=[];best=-np.inf if maximize else np.inf
    for row in ordered:
        value=row['quality'][objective]
        if not np.isfinite(value) or row['cost_bytes']<=0:raise ValueError('finite measured points required')
        if value>best if maximize else value<best:
            keep.append(row['uid']);best=value
    return keep
