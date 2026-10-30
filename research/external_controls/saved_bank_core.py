"""Pure repeated-calibration and byte-allocation rules for saved model banks."""
import hashlib
import numpy as np
from unified_frontier_core import choose_policy
from unified_serving_core import bank_frontier

SELECTORS=('standard','mean','conservative')


def policy_summary(replays,task):
    """One versus three repeats; mean is the equal-query control for mean-SD."""
    objective='loss' if task=='multiclass' else 'f1_macro'
    names=list(replays[0]['scores']);assert all(set(r['scores'])==set(names) for r in replays)
    average={name:{k:float(np.mean([r['scores'][name][k] for r in replays])) for k in replays[0]['scores'][name]} for name in names}
    single=choose_policy(replays[0]['scores'],objective);mean=choose_policy(average,objective)
    return {s:dict(policy=single if s=='standard' else mean,
        repeats=[replays[0]['scores'][single]] if s=='standard' else [r['scores'][mean] for r in replays]) for s in SELECTORS}


def score_options(models,selector,anchor,task):
    ref=np.mean([q['loss' if task=='multiclass' else 'f1_macro'] for q in anchor['summaries'][selector]['repeats']])
    denominator=max(float(ref),1e-12);result=[]
    for m in models:
        values=np.asarray([1-q['loss']/denominator if task=='multiclass' else q['f1_macro']/denominator-1 for q in m['summaries'][selector]['repeats']])
        mean=float(values.mean());sd=float(values.std(ddof=1)) if len(values)>1 else 0.
        result.append(dict(uid=m['uid'],family=m['family'],cost_bytes=m['cost_bytes'],utility=mean-(sd if selector=='conservative' else 0.),
            utility_mean=mean,utility_sd=sd,policy=m['summaries'][selector]['policy']))
    return result


def limited(options,cap):
    if cap is None or len(options)<=cap:return options
    ordinary=sorted([o for o in options if o['family']=='ordinary'],key=lambda o:hashlib.sha256(o['uid'].encode()).hexdigest())
    compact=sorted([o for o in options if o['family']!='ordinary'],key=lambda o:hashlib.sha256(o['uid'].encode()).hexdigest())
    # Balanced deterministic catalogue; order depends on UID only, never quality.
    selected=[]
    while ordinary or compact:
        for pool in (ordinary,compact):
            if pool and len(selected)<cap:selected.append(pool.pop(0))
        if len(selected)==cap:break
    return selected


def allocate(cases,budgets,strategy,cap=None):
    if strategy not in ('ordinary_equal','ordinary','compact','mixed'):raise ValueError('unknown bank strategy')
    pools=[]
    for options in cases:
        pool=[o for o in options if strategy=='mixed' or (o['family']=='ordinary')==(strategy!='compact')]
        # Match the candidate cap after filtering each strategy's domain.
        pools.append(limited(pool,cap))
    if any(not p for p in pools):return [dict(budget=b,status='infeasible',reason='empty_strategy_pool') for b in budgets]
    if strategy!='ordinary_equal':return bank_frontier(pools,budgets)
    result=[]
    for budget in budgets:
        affordable=[[o for o in p if o['cost_bytes']<=budget//len(pools)] for p in pools]
        if any(not p for p in affordable):result.append(dict(budget=budget,status='infeasible'));continue
        selected=[min(p,key=lambda o:(-o['utility'],o['cost_bytes'],o['uid'])) for p in affordable]
        result.append(dict(budget=budget,status='feasible',cost_bytes=sum(o['cost_bytes'] for o in selected),
            utility=float(np.mean([o['utility'] for o in selected])),choices=[o['uid'] for o in selected]))
    return result
