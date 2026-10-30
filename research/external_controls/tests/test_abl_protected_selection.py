from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from abl_protected_study import primary_choice
from audit_abl_protected import choose


def test_protected_families_preserve_independent_ranking_and_full_reference():
    rng=np.random.default_rng(71)
    names=['raw','temperature','raw_shift_0.25','raw_shift_0.5','raw_shift_1',
        'temperature_shift_0.25','temperature_shift_0.5','temperature_shift_1']
    metas={f'{family}/{i}':dict(family=family,cost_bytes=(i+1)*40,nnz=i)
        for family in ('full','functional','current','mean','history') for i in range(2)}
    for _ in range(4):
        rows=[dict(uid=uid,validation={name:dict(loss=.3+int(rng.integers(30))*1e-7,
            f1_macro=.7+float(rng.choice([0,1e-14,1e-5]))) for name in names}) for uid in metas]
        for mode in ('raw','shifts','temperature_shifts'):
            for objective in ('loss','f1_macro'):
                for group in ('full_unbounded','native_budget','functional','current','mean','history','deployed'):
                    for budget in (30,40,80):
                        c=dict(mode=mode,objective=objective,group=group)
                        assert primary_choice(rows,metas,c,budget)==choose(rows,metas,c,budget)
