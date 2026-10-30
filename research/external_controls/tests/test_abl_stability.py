from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
import budget_quality_worker
from abl_quality_core import choose_budget
from abl_stability_core import NumericalTolerance,select_stable


def test_exact_stable_boundary_reproduces_previous_selector():
    rng=np.random.default_rng(12)
    names=['raw','temperature','raw_shift_0.25','raw_shift_0.5','raw_shift_1',
        'temperature_shift_0.25','temperature_shift_0.5','temperature_shift_1']
    metas={'n':dict(family='pyboost',cost_bytes=100,nnz=0),
        'a':dict(family='compact',cost_bytes=70,nnz=0),'h':dict(family='compact',cost_bytes=80,nnz=2)}
    for _ in range(20):
        rows=[dict(uid=u,validation={n:dict(loss=float(rng.random()),f1_macro=float(rng.random())) for n in names}) for u in metas]
        for mode in ('raw','temperature','shifts','temperature_shifts'):
            for objective in ('loss','f1_macro'):
                for group in ('native','ab','abl'):
                    for budget in (69,70,80,100):
                        assert select_stable(rows,metas,group,mode,objective,budget,NumericalTolerance())==choose_budget(rows,metas,group,mode,objective,budget)


def test_numerical_band_prefers_cheaper_equivalent_but_keeps_real_macro_gain():
    metas={'n':dict(family='pyboost',cost_bytes=70,nnz=0),
        'h':dict(family='compact',cost_bytes=90,nnz=5)}
    rows=[dict(uid='n',validation={'raw':dict(loss=.3,f1_macro=.8)}),
        dict(uid='h',validation={'raw':dict(loss=.3-1e-10,f1_macro=.8)})]
    assert select_stable(rows,metas,'deployed','raw','loss',100,NumericalTolerance())['uid']=='h'
    assert select_stable(rows,metas,'deployed','raw','loss',100,NumericalTolerance(1e-6,1e-12))['uid']=='n'
    rows[1]['validation']['raw']['f1_macro']=.8001
    assert select_stable(rows,metas,'deployed','raw','f1_macro',100,NumericalTolerance(1e-6,1e-12))['uid']=='h'
