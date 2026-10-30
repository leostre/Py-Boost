"""Bounded broad-study decisions; no file, tree-fit, GPU or test entry point."""
from dataclasses import replace
from fractions import Fraction
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from unified_frontier_core import Fold
from residual_gate_core import fit_policies, sufficient, quality, choose_policy
from abl_budget_revision_core import pack
from experiments.parallel_leaf_core import prefix


ORDINARY_COUNTS=(10,20,30,40,60,80,100,120,140,160,180,200)


def budget_reference(outputs):
    if not isinstance(outputs,(int,np.integer)) or outputs<1:raise ValueError('positive output domain required')
    return int(200*(7*16+8*4*outputs+16)+32*outputs+4096)


def byte_limit(reference,ratio):return int(Fraction(str(ratio))*reference)


def ranks_for(c):
    return (1,16,32,64,c) if c>64 else tuple(sorted({1,*[max(1,int(np.ceil(c*f))) for f in (.25,.5,.75,1.)]}))


def grouped_folds(x,seed):
    _,groups=np.unique(np.asarray(x),axis=0,return_inverse=True)
    if len(np.unique(groups))<2:raise ValueError('at least two D feature groups required')
    proposals=GroupShuffleSplit(n_splits=64,test_size=.5,random_state=seed).split(x,groups=groups)
    a,b=min(proposals,key=lambda pair:abs(len(pair[1])-.5*len(x)))
    return (Fold(tuple(map(int,a)),tuple(map(int,b))),Fold(tuple(map(int,b)),tuple(map(int,a))))


def assess(raw,y,folds,grid,shrinkages):
    raw=np.asarray(raw,float);y=np.asarray(y);arrays={};states=[]
    for fold in folds:
        a,b=np.asarray(fold.fitting),np.asarray(fold.validation)
        fitted=fit_policies(raw[a],y[a],'multilabel',grid,shrinkages)
        states.append(dict(fitting=list(fold.fitting),validation=list(fold.validation),policies=fitted))
        for name,policy in fitted.items():
            if name not in arrays:arrays[name]=(np.zeros_like(y,dtype=bool),np.empty(len(y)),np.empty(len(y)))
            for dst,src in zip(arrays[name],sufficient(y[b],raw[b],'multilabel',policy)):dst[b]=src
    scores={k:quality(y,*v,'multilabel',raw.shape[1]) for k,v in arrays.items()}
    chosen=choose_policy(scores,'multilabel')
    refitted=fit_policies(raw,y,'multilabel',grid,shrinkages)
    return dict(policy=chosen,quality=scores[chosen],scores=scores,states=states,refit_control=refitted[chosen]),arrays[chosen]


def model_prefix(model,count):
    if not 1<=count<=model.n_trees:raise ValueError('invalid prefix')
    blocks=[];remaining=count
    for block in model.blocks:
        if remaining<=0:break
        k=min(remaining,block.base.n_trees);blocks.append(prefix(block,k));remaining-=k
    return pack(blocks,model.bias)


def train_shortlist(pool,maximum=24,mandatory=()):
    """Preserve cost coverage and train-quality tradeoffs without a D argument."""
    pool=list(pool);by={p['uid']:p for p in pool};chosen={u:by[u] for u in mandatory if u in by}
    ordered=sorted(pool,key=lambda p:(p['cost']['cost_bytes'],p['train_risk_mse'],p['uid']))
    for ids in np.array_split(np.arange(len(ordered)),max(1,maximum-len(chosen))):
        if len(ids):
            p=min((ordered[i] for i in ids),key=lambda p:(p['train_risk_mse'],p['cost']['cost_bytes'],p['uid']));chosen[p['uid']]=p
    for p in sorted(pool,key=lambda p:(p['train_risk_mse'],p['cost']['cost_bytes'],p['uid'])):
        if len(chosen)>=maximum:break
        chosen[p['uid']]=p
    return list(chosen.values())[:maximum]


def qkey(m):return (-m['assessment']['quality']['f1_macro'],m['assessment']['quality']['loss'],m['cost_bytes'],m['uid'])


def select_cases(cases):
    result=[]
    for case in cases:
        models=case['models'];families=sorted({m['family'] for m in models})
        # The mixed bank is a separate 24-query catalog, never the union of two full searches.
        ordinary=[m for m in models if m['family']=='ordinary']
        ordinary=sorted(ordinary,key=lambda m:(m['cost_bytes'],m['uid']))
        ordinary12=[ordinary[i] for i in np.linspace(0,len(ordinary)-1,min(12,len(ordinary))).astype(int)]
        post=[m for m in models if m['family']=='posthoc']
        post=sorted(post,key=lambda m:(m['cost_bytes'],m['uid']))
        post12=[post[i] for i in np.linspace(0,len(post)-1,min(12,len(post))).astype(int)]
        catalogs={f:[m for m in models if m['family']==f] for f in families}
        catalogs['mixed24']=list({m['uid']:m for m in ordinary12+post12}.values())
        for family,pool in catalogs.items():
            for ratio in (.25,.5,.7,1.):
                limit=byte_limit(case['analytic_reference_bytes'],ratio)
                affordable=[m for m in pool if m['cost_bytes']<=limit]
                row=dict(case=case['case'],family=family,ratio=ratio,limit_bytes=limit,queries=len(pool))
                result.append(dict(row,uid=min(affordable,key=qkey)['uid']) if affordable else dict(row,uid=None,reason='no_affordable_model'))
    return dict(selected=result,test_evaluated=False)
