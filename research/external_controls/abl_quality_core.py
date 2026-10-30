"""Pure rank/residual plans, fitted representations and development-only choices.

The caller selects the pinned native_source import root before importing this
module. No file, test target, GPU, model training, or global RNG is used here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import numpy as np
from scipy import sparse

from calibration_pilot_core import (POLICIES, POLICY_BY_NAME, choose_policy,
    control_for, policy_key)
from experiments.shared_leaf_basis import functional_basis, compress
from experiments.sparse_leaf_core import (contrast, training_weights,
    coordinate_scores, score_order, make_sparse_model)
from experiments.uniform_threshold_core import ThresholdModel

CASES=('helena_1204','mnist_3104','rt_iot2022_3204')
RANK_FRACTIONS=(.125,.25,.5,.75,1.)
RESIDUAL_FRACTIONS=(0.,.01,.02,.05)
MODES=('raw','temperature','shifts','temperature_shifts')
GROUPS=('native','ab','abl')


@dataclass(frozen=True)
class RepresentationPlan:
    rank: int
    residual_fraction: float
    maximum_nnz: int


def rank_levels(classes,leaves,training_rows):
    if min(classes,leaves,training_rows)<2:
        raise ValueError('at least two classes, leaves and training rows required')
    ceiling=min(classes-1,leaves,training_rows)
    return tuple(sorted({max(1,math.ceil(f*ceiling)) for f in RANK_FRACTIONS}))


def representation_plans(classes,leaves,training_rows,parent_parameters):
    if isinstance(parent_parameters,bool) or not isinstance(parent_parameters,int) or parent_parameters<1:
        raise ValueError('positive integer parameter denominator required')
    return tuple(RepresentationPlan(r,p,math.floor(p*parent_parameters))
        for r in rank_levels(classes,leaves,training_rows) for p in RESIDUAL_FRACTIONS)


def source_models(records,case_id,family,compression_only=False):
    record=next(r for r in records if r['case_id']==case_id and r['family']==family)
    if family=='tabpfn8':
        selected=[m for m in record['models'] if m['identity']=='frozen8']
    else:
        rounds=(160,200,400,800) if compression_only else (40,80,160,200,400,800)
        selected=[m for m in record['models'] if m['origin']=='new' and m['depth'] in (3,5) and m['rounds'] in rounds]
        if len({(m['depth'],m['rounds']) for m in selected})!=len(selected):
            raise ValueError('duplicate origin=new source identity')
    expected=1 if family=='tabpfn8' else (8 if compression_only else 12)
    if len(selected)!=expected:raise ValueError('incomplete fixed source grid')
    return record,tuple(sorted(selected,key=lambda m:(m['depth'],m['rounds'],m['identity'])))


def fingerprint(model):
    h=hashlib.sha256()
    for a in (*base_arrays(model.base),model.rows,model.columns,model.values,model.decision_shifts):
        h.update(str((a.shape,a.dtype.str)).encode());h.update(np.ascontiguousarray(a).tobytes())
    h.update(str((model.base.task,model.base.representation,model.base.n_features)).encode())
    return h.hexdigest()


def base_arrays(base):
    return tuple(getattr(base,n) for n in ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias'))


def array_ledger(model,prepared=None):
    """Price one deployed snapshot; also report retaining both physical copies."""
    prepared=model.prepare_inference() if prepared is None else prepared
    arrays=[(name,getattr(prepared.base,name)) for name in
        ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias')]
    arrays.extend((name,getattr(prepared,name)) for name in ('indptr','columns','values','decision_shifts'))
    rows=[dict(name=name,shape=list(a.shape),dtype=str(a.dtype),nbytes=a.nbytes) for name,a in arrays]
    deployed=sum(a['nbytes'] for a in rows)
    if deployed!=prepared.resident_bytes:raise ValueError('deployed ledger mismatch')
    # prepare() owns copies, including immutable byte-backed base arrays. Holding
    # the original COO model alongside it is NOT max(COO, CSR).
    return dict(prepared_bytes=int(deployed),coo_bytes=int(model.payload_bytes),
        cost_bytes=int(deployed+4),both_retained_bytes=int(model.payload_bytes+deployed+4),
        temperature_bytes=4,parameters=int(model.parameter_count+1),arrays=rows,
        memory_kind='unique arrays of one prepared-only serving snapshot plus one float32 T; no whole-process claim')


def design_raw(model,design):
    b=model.base
    latent=np.asarray(design@b.coefficients.astype(np.float64))
    if b.representation=='low_rank':raw=latent@b.basis.T
    elif b.representation=='reference_class':raw=np.column_stack((latent,np.zeros(len(latent))))
    else:raw=latent
    raw+=b.bias
    if len(model.values):
        residual=sparse.coo_matrix((model.values.astype(float),(model.rows,model.columns)),
            shape=(len(b.coefficients),len(b.bias))).tocsr()
        raw+=(design@residual).toarray()
    return np.asarray(raw,np.float64)


def fit_representation_family(dense,train_design,training_target):
    full=full_output_base(dense)
    if full.task!='multiclass' or len(dense.values):
        raise ValueError('dense multiclass parent required')
    contributions=np.asarray(train_design@full.coefficients.astype(float))
    teacher=contributions+full.bias
    mean,basis,singular=functional_basis(contributions,full.task)
    frequency=np.asarray(train_design.mean(axis=0)).ravel()
    weights=training_weights(training_target,teacher,dense.base.task,'balanced_mse')
    return dict(mean=mean,basis=basis,singular=singular,frequency=frequency,
        teacher=teacher,weights=weights)


def full_output_base(model):
    """Decode the saved C-1 reference coordinates; do not change serving cost.

    The zero column is temporary fitting geometry. Original artifacts and their
    reference-class array ledger remain the ordinary-model comparator.
    """
    base=model.base
    if base.representation=='dense':return base
    if base.representation!='reference_class':raise ValueError('uncompressed parent required')
    coefficients=np.column_stack((base.coefficients,np.zeros(len(base.coefficients),np.float32)))
    return replace(base,coefficients=coefficients,representation='dense')


def prepare_rank(dense,train_design,state,rank):
    full=full_output_base(dense)
    base=compress(full,state['mean'],state['basis'][:,:rank])
    pure=ThresholdModel(base,np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),dense.decision_shifts.copy())
    residual=(contrast(full.coefficients,dense.base.task)-
        contrast(base.coefficients.astype(float)@base.basis.T,dense.base.task)).astype(np.float32)
    error=contrast(state['teacher']-design_raw(pure,train_design),dense.base.task)
    scores=coordinate_scores(train_design,residual,error,'balanced_mse',state['weights'])
    return pure,residual,score_order(scores,residual)


def build_representation(pure,residual,order,state,plan):
    count=min(plan.maximum_nnz,len(order))
    if count==0:return pure
    corrected=make_sparse_model(pure.base,residual,order,count,state['frequency'])
    return ThresholdModel(corrected.base,corrected.rows,corrected.columns,corrected.values,pure.decision_shifts.copy())


def allowed_policies(mode,objective):
    if mode not in MODES or objective not in ('loss','f1_macro'):raise ValueError('invalid decision mode')
    names=[]
    for policy in POLICIES:
        if mode=='raw' and policy.name!='raw':continue
        if mode=='temperature' and policy.shift_fraction:continue
        if mode=='shifts' and policy.temperature:continue
        if objective=='loss' and policy.shift_fraction:continue
        names.append(policy.name)
    return tuple(names)


def choose_mode_policy(scores,mode,objective):
    names=allowed_policies(mode,objective)
    return min(names,key=lambda name:policy_key(name,scores[name],objective))


def membership(meta,group):
    if group=='native':return meta['family']=='pyboost'
    if group=='ab':return meta['family']=='compact' and meta['nnz']==0
    if group=='abl':return meta['family']=='compact'
    if group in ('catboost','xgboost','tabpfn8'):return meta['family']==group
    raise ValueError('unknown candidate family')


def choose_budget(evaluations,metas,group,mode,objective,budget=None):
    candidates=[]
    for row in evaluations:
        meta=metas[row['uid']]
        if not membership(meta,group):continue
        if budget is not None and (meta['cost_bytes'] is None or meta['cost_bytes']>budget):continue
        policy=choose_mode_policy(row['validation'],mode,objective)
        score=row['validation'][policy]
        size=meta['cost_bytes'] or 0
        key=(score['loss'],size,row['uid']) if objective=='loss' else (-score['f1_macro'],score['loss'],size,row['uid'])
        candidates.append((key,dict(uid=row['uid'],policy=policy,validation=score,cost_bytes=meta['cost_bytes'])))
    return min(candidates,key=lambda item:item[0])[1] if candidates else None


def fitted_control(row,policy_name):
    fit=row['refit']
    return control_for(POLICY_BY_NAME[policy_name],(fit['temperature'],np.asarray(fit['raw_shifts'],np.float32),
        np.asarray(fit['temperature_shifts'],np.float32)))
