from pathlib import Path
import json
import numpy as np
from test_abl_quality_core import fixture
from budget_quality_worker import write,digest
from abl_protected_core import as_protected


def test_final_seals_before_first_test_label_read_and_never_reselects(tmp_path,monkeypatch):
    import unified_final_evaluation as worker
    x=np.linspace(-1,1,40,dtype=np.float32)[:,None];y=np.arange(40)%4
    dataset=tmp_path/'data.npz';splits=tmp_path/'splits.npz';np.savez(dataset,features=x,target=y)
    np.savez(splits,train=np.arange(24),calibration=np.arange(24,28),control=np.arange(28,32),selection=np.arange(32,36),test=np.arange(36,40))
    record=dict(case=dict(task='multiclass',outputs=4,split_file=str(splits),split_sha256=digest(splits)),dataset=dict(path=str(dataset),sha256=digest(dataset)))
    models=[]
    for block in range(4):
        model,_,_,_=fixture(17+block,4,block+1);path=tmp_path/f'model{block}.npz';as_protected(model).save(path)
        q=dict(loss=1/(block+1),f1_macro=.5,brier=.2,f1_weighted=.5)
        evaluations=[dict(objective=o,quality=q,refit_control=dict(temperature=1.,shifts=[0.]*4)) for o in ('loss','f1_macro')]
        models.append(dict(uid=f'ordinary/{block}',family='ordinary',block=block,artifact=str(path),sha256=digest(path),evaluations=evaluations,
            cost_bytes=40*(block+1),parameters=10,trees=block+1,blocks=1,nnz=0))
    case=tmp_path/'case.json';write(case,dict(case='toy_7',task='multiclass',record=record,models=models,reference_bytes=160))
    report=tmp_path/'training.json';write(report,dict(status='unified_training_complete',cases=[dict(case='toy_7',path=str(case),sha256=digest(case),audit=dict(status='passed'))]))
    external=tmp_path/'external.json';write(external,dict(status='unified_external_serving_complete',test_predictions=[]))
    original=worker.data_test;reads=[]
    def checked(record):
        assert (tmp_path/'result/pretest_seal.json').exists();reads.append(1);return original(record)
    monkeypatch.setattr(worker,'data_test',checked)
    cfg=dict(unified_final=dict(native_sources={'fixture':dict(path=str(report),sha256=digest(report))},serving_sources={},
        external=dict(path=str(external),sha256=digest(external)),maximum_models=20))
    worker.run(cfg,tmp_path/'result');result=json.loads((tmp_path/'result/report.json').read_text())
    assert len(reads)==1 and result['audits_passed'] and result['new_tree_fits']==0
    seal=json.loads((tmp_path/'result/pretest_seal.json').read_text())
    assert [r['uid'] for r in result['rows']]==[r['uid'] for r in seal['choices']]
