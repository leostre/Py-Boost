"""Independent stable rank implementation at numerical and byte boundaries."""
import numpy as np

import budget_quality_worker
from audit_abl_width import width_choice
from abl_stability_core import NumericalTolerance,select_stable


def test_independent_width_rank_matches_fixed_band_and_preserves_real_f1_gains():
    rng=np.random.default_rng(190)
    names=['raw','temperature','raw_shift_0.25','raw_shift_0.5','raw_shift_1',
        'temperature_shift_0.25','temperature_shift_0.5','temperature_shift_1']
    metas={'n0':dict(family='pyboost',cost_bytes=50,nnz=0),'n1':dict(family='pyboost',cost_bytes=100,nnz=0),
        'a0':dict(family='compact',cost_bytes=49,nnz=0),'a1':dict(family='compact',cost_bytes=70,nnz=0),
        'h0':dict(family='compact',cost_bytes=69,nnz=1),'h1':dict(family='compact',cost_bytes=101,nnz=5)}
    for repeat in range(12):
        rows=[dict(uid=uid,plan=0,validation={name:dict(
            loss=.3+int(rng.integers(0,30))*1e-7,
            f1_macro=.8+float(rng.choice([0,1e-14,1e-5]))) for name in names}) for uid in metas]
        for mode in ('raw','temperature','shifts','temperature_shifts'):
            for objective in ('loss','f1_macro'):
                for group in ('native','ab','abl','deployed'):
                    for budget in (48,49,50,69,70,100,101):
                        choice=dict(plan=0,mode=mode,objective=objective,group=group,budget_bytes=budget)
                        expected=select_stable(rows,metas,group,mode,objective,budget,NumericalTolerance(1e-6,1e-12))
                        assert width_choice(rows,metas,choice)==expected
