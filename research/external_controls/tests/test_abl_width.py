from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
import budget_quality_worker
from budget_quality_worker import digest
from calibration_pilot_core import make_plans,Scheme
from abl_width_study import single_crossfit


def test_width_comparison_reuses_exact_crossfit_zero_instead_of_first_pooled_plan(tmp_path):
    y=np.arange(75)%5
    plans=[p for repeat in range(10) for p in make_plans(y,[20,20,35],repeat,(1.,),20261023)
        if p.scheme in (Scheme.POOLED,Scheme.CROSSFIT)]
    payload=dict(plans=[dict(asdict(p),scheme=p.scheme.value) for p in plans],
        train_ids=list(range(100,110)),dev_ids=list(range(75)),test_ids=list(range(110,120)),
        target=y.tolist(),sizes=[20,20,35])
    payload=json.loads(json.dumps(payload))
    path=tmp_path/'plans.json';path.write_text(json.dumps(payload),encoding='utf-8')
    before=path.read_bytes()
    index,plan,export=single_crossfit(dict(plans_path=str(path),plans_sha256=digest(path)))
    assert index==1 and plan==plans[1]
    assert len(export['plans'])==1 and export['plans'][0]==payload['plans'][1]
    assert {k:v for k,v in export.items() if k!='plans'}=={k:v for k,v in payload.items() if k!='plans'}
    assert sorted(i for fold in plan.folds for i in fold.validation)==list(range(75))
    assert path.read_bytes()==before
