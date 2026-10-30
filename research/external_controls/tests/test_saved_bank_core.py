from itertools import product
import copy
import numpy as np
import pytest
from saved_bank_core import policy_summary,score_options,limited,allocate


def q(f1,loss=.3):return dict(f1_macro=f1,f1_weighted=f1,loss=loss,brier=.1)


def test_conservative_and_mean_share_queries_but_differ_on_unstable_choice():
    def model(uid,values):
        replays=[dict(scores={'raw':q(v)}) for v in values]
        return dict(uid=uid,family='ordinary',cost_bytes=10,summaries=policy_summary(replays,'multilabel'))
    anchor=model('anchor',[.5,.5,.5]);stable=model('stable',[.6,.6,.6]);unstable=model('unstable',[.5,.7,.75])
    mean=score_options([stable,unstable],'mean',anchor,'multilabel');robust=score_options([stable,unstable],'conservative',anchor,'multilabel')
    assert max(mean,key=lambda o:o['utility'])['uid']=='unstable'
    assert max(robust,key=lambda o:o['utility'])['uid']=='stable'
    assert all(len(m['summaries']['mean']['repeats'])==len(m['summaries']['conservative']['repeats'])==3 for m in (stable,unstable))


def test_exact_mixed_and_ordinary_allocation_and_equal_quota_failure():
    cases=[[dict(uid=f'{i}-{j}',family='ordinary' if j%2==0 else 'compact',cost_bytes=3+i+j,utility=.1*j*(i+1)) for j in range(4)] for i in range(3)]
    for strategy in ('ordinary','compact','mixed'):
        for a in allocate(cases,[6,15,24],strategy):
            pools=[[o for o in p if strategy=='mixed' or (o['family']=='ordinary')==(strategy=='ordinary')] for p in cases]
            possible=[p for p in product(*pools) if sum(x['cost_bytes'] for x in p)<=a['budget']]
            if not possible:assert a['status']=='infeasible'
            else:assert a['utility']==pytest.approx(max(np.mean([o['utility'] for o in p]) for p in possible))
    assert allocate(cases,[12],'ordinary_equal')[0]['status']=='infeasible'


def test_catalogue_cap_is_quality_blind_and_does_not_mutate_inputs():
    options=[dict(uid=str(i),family='ordinary' if i%2 else 'compact',cost_bytes=i+1,utility=float(i)) for i in range(20)]
    before=copy.deepcopy(options);first=[o['uid'] for o in limited(options,8)]
    changed=[dict(o,utility=-o['utility']*77) for o in options]
    assert first==[o['uid'] for o in limited(changed,8)] and options==before
