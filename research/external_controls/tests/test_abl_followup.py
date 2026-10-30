"""Independent audit ranks and sealed serving plans on synthetic choices."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
import budget_quality_worker
from abl_quality_core import choose_budget
from audit_abl_quality import independent_choice
from abl_serving_core import serving_plans
from abl_serving_study import summarize


def test_independent_auditor_replays_budget_rankings_and_nested_policy_ties():
    rng=np.random.default_rng(47)
    policies=['raw','temperature','raw_shift_0.25','raw_shift_0.5','raw_shift_1',
        'temperature_shift_0.25','temperature_shift_0.5','temperature_shift_1']
    metas={
        'full':dict(family='pyboost',cost_bytes=100,nnz=0),
        'ab':dict(family='compact',cost_bytes=50,nnz=0),
        'hybrid':dict(family='compact',cost_bytes=75,nnz=3),
        'cat':dict(family='catboost',cost_bytes=None,nnz=0),
        'xgb':dict(family='xgboost',cost_bytes=None,nnz=0),
        'tfm':dict(family='tabpfn8',cost_bytes=None,nnz=0)}
    for repeat in range(12):
        rows=[dict(uid=uid,plan=0,validation={name:dict(loss=float(rng.random()),
            f1_macro=float(rng.random())) for name in policies}) for uid in metas]
        if repeat==0:
            for row in rows:
                for score in row['validation'].values():score.update(loss=.5,f1_macro=.5)
        for mode in ('raw','temperature','shifts','temperature_shifts'):
            for objective in ('loss','f1_macro'):
                for group in ('native','ab','abl','catboost','xgboost','tabpfn8'):
                    for budget in (None,49,50,74,75,100):
                        choice=dict(plan=0,mode=mode,objective=objective,group=group)
                        if budget is not None:choice['budget_bytes']=budget
                        assert independent_choice(rows,metas,choice)==choose_budget(rows,metas,group,mode,objective,budget)


def test_serving_uses_sealed_choices_including_missing_native_and_both_objectives():
    @dataclass
    class C:
        temperature:float
        shifts:tuple
    meta=dict(source=dict(artifact='saved.npz',sha256='abc'),rank=2,nnz=5,cost_bytes=50)
    choices=[]
    for objective in ('loss','f1_macro'):
        for ratio in (.25,.5,1,2,4):
            for group in ('native','ab','abl'):
                choices.append(dict(scheme='crossfit',repeat=0,mode='temperature_shifts',
                    objective=objective,budget_ratio=ratio,group=group,plan=0,
                    chosen=None if group=='native' else dict(uid='chosen',policy='raw')))
    case=dict(case_id='tiny',rows=[dict(uid='chosen',plan=0)],metas={'chosen':meta},
        budget_choices=choices,record={})
    result=serving_plans(case,lambda row,policy:C(1.,(0.,0.)))
    assert len(result)==6 and {r['objective'] for r in result}=={'loss','f1_macro'}
    assert {r['budget_ratio'] for r in result}=={.5,1,2}
    assert all(r['models']['native'] is None for r in result)
    assert all(r['models']['abl']['uid']=='chosen' for r in result)


def test_timing_summary_preserves_request_pairing():
    item=dict(job=dict(case='tiny',objective='loss',budget_ratio=1,
        models={'native':dict(uid='f'),'abl':dict(uid='h')}),samples=[])
    for block in range(3):
        for request in range(32):
            for role,scale in (('native',1),('abl',.7)):
                item['samples'].append(dict(role=role,batch=32,mode='device',
                    block=block,request=request,milliseconds=(request+1)*scale))
    result=summarize([item],[])
    assert abs(result['paired_ratios'][0]['paired_median_compact_over_native']-.7)<1e-12
