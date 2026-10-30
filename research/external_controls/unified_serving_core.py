"""Exact block coalescing and finite model-bank allocation; no GPU import."""
from dataclasses import replace
import numpy as np
from abl_protected_core import block_cost
from experiments.block_leaf_model import BlockLeafModel


def summary(values):
    return dict(zip(('p50_ms','p95_ms','p99_ms'),map(float,np.percentile(values,[50,95,99]))))


def quality_key(model,task):
    objective='loss' if task=='multiclass' else 'f1_macro'
    q=next(e['quality'] for e in model['evaluations'] if e['objective']==objective)
    return (q['loss'],-q['f1_macro'],model['cost_bytes']) if task=='multiclass' else (-q['f1_macro'],q['loss'],model['cost_bytes'])


def compatible(a,b):
    return (a.base.representation==b.base.representation and a.base.task==b.base.task
        and a.base.n_features==b.base.n_features and a.base.coefficients.shape[1]==b.base.coefficients.shape[1]
        and np.array_equal(a.base.basis,b.base.basis))


def join_blocks(blocks):
    first=blocks[0];noff=[];loff=[];rows=[];n=l=0
    for b in blocks:
        if not compatible(first,b):raise ValueError('decoders differ')
        noff.extend(b.base.node_offsets[:-1]+n);loff.extend(b.base.leaf_offsets[:-1]+l)
        rows.append(b.rows+l);n+=len(b.base.nodes);l+=len(b.base.coefficients)
    base=replace(first.base,nodes=np.concatenate([b.base.nodes for b in blocks]),
        node_offsets=np.asarray(noff+[n],np.int64),leaf_offsets=np.asarray(loff+[l],np.int64),
        coefficients=np.concatenate([b.base.coefficients for b in blocks]))
    return replace(first,base=base,rows=np.concatenate(rows).astype(np.int32),
        columns=np.concatenate([b.columns for b in blocks]),values=np.concatenate([b.values for b in blocks]))


def coalesce(model):
    groups=[]
    for b in model.blocks:
        if groups and compatible(groups[-1][0],b):groups[-1].append(b)
        else:groups.append([b])
    return BlockLeafModel(tuple(join_blocks(g) for g in groups),model.bias.copy(),model.decision_shifts.copy())


def bank_frontier(cases,budgets):
    """Multiple-choice knapsack via exact nondominated partial sums (small banks)."""
    if not cases or any(not r for r in cases):raise ValueError('nonempty options for every task required')
    def nondominated(rows):
        best=-np.inf;result=[]
        for row in sorted(rows,key=lambda r:(r[0],-r[1],r[2])):
            if row[1]>best:result.append(row);best=row[1]
        return result
    # Exact dominance removal, including equal-byte alternatives. No binning.
    cases=[[dict(cost_bytes=c,utility=u,uid=uid[0]) for c,u,uid in nondominated(
        [(o['cost_bytes'],o['utility'],(o['uid'],)) for o in options])] for options in cases]
    independent=[min(options,key=lambda o:(-o['utility'],o['cost_bytes'],o['uid'])) for options in cases]
    min_costs=[min(o['cost_bytes'] for o in options) for options in cases];minimum=sum(min_costs)
    best_cost=sum(o['cost_bytes'] for o in independent)
    active=[b for b in budgets if minimum<=b<best_cost]
    limit=max(active,default=0);states=[(0,0.,())]
    for index,options in enumerate(cases) if active else []:
        remaining=sum(min_costs[index+1:])
        expanded=[(cost+o['cost_bytes'],value+o['utility'],ids+(o['uid'],))
            for cost,value,ids in states for o in options if cost+o['cost_bytes']+remaining<=limit]
        states=nondominated(expanded)
        if not states:break
    results=[]
    for budget in budgets:
        if budget<minimum:results.append(dict(budget=budget,status='infeasible'));continue
        if budget>=best_cost:
            r=best_cost,sum(o['utility'] for o in independent),tuple(o['uid'] for o in independent)
        else:r=max((r for r in states if r[0]<=budget),key=lambda r:(r[1],-r[0]))
        results.append(dict(budget=budget,status='feasible',cost_bytes=r[0],utility=r[1]/len(cases),choices=list(r[2])))
    return results
