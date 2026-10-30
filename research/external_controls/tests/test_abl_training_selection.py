from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
import budget_quality_worker
from abl_training_study import training_choice
from abl_training_development_audit import independent_training_choice


def test_training_families_use_same_bands_but_keep_unbounded_reference_distinct():
    names=['raw','temperature','raw_shift_0.25','raw_shift_0.5','raw_shift_1',
        'temperature_shift_0.25','temperature_shift_0.5','temperature_shift_1']
    metas={'full/small':dict(family='full',cost_bytes=70,nnz=0),
        'full/large':dict(family='full',cost_bytes=140,nnz=0),
        'fixed/x':dict(family='fixed',cost_bytes=95,nnz=3),
        'joint/y':dict(family='joint',cost_bytes=85,nnz=0),
        'posthoc/z':dict(family='posthoc',cost_bytes=90,nnz=2)}
    rng=np.random.default_rng(170)
    for _ in range(10):
        rows=[dict(uid=uid,validation={name:dict(loss=.3+int(rng.integers(30))*1e-7,
            f1_macro=.8+float(rng.choice([0,1e-14,1e-5]))) for name in names}) for uid in metas]
        for mode in ('raw','temperature','shifts','temperature_shifts'):
            for objective in ('loss','f1_macro'):
                for group in ('full_unbounded','native_budget','posthoc','fixed','joint','deployed'):
                    for budget in (69,70,85,95,140):
                        choice=dict(mode=mode,objective=objective,group=group)
                        assert training_choice(rows,metas,group,mode,objective,budget)==independent_training_choice(rows,metas,choice,budget)
    rows=[dict(uid=u,validation={'raw':dict(loss=loss,f1_macro=.8)})
        for u,loss in [('full/small',.3),('full/large',.2),('fixed/x',.15)]]
    assert training_choice(rows,metas,'full_unbounded','raw','loss',100)['uid']=='full/large'
    assert training_choice(rows,metas,'native_budget','raw','loss',100)['uid']=='full/small'
    assert training_choice(rows,metas,'deployed','raw','loss',100)['uid']=='fixed/x'
