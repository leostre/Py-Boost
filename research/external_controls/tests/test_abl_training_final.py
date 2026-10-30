"""Final-test access must fail closed on incomplete development."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from budget_quality_worker import digest,write
from abl_training_final import run
from abl_training_final import evaluate_case


def test_final_refuses_pilot_before_reading_any_dataset(tmp_path):
    p=tmp_path/'report.json';write(p,dict(status='pilot_development_complete',test_accessed=False))
    cfg=dict(abl_training_development_completed=dict(path=str(p),sha256=digest(p)))
    with pytest.raises(RuntimeError,match='sealed development required'):run(cfg,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_final_refuses_incomplete_case_seal_before_reading_test(tmp_path):
    audit=tmp_path/'development_audit.json';write(audit,dict(status='passed'))
    seal=tmp_path/'pretest_choices.json';write(seal,dict(status='abl_training_choices_sealed_before_test',cases=[],
        development_audit_sha256=digest(audit)))
    p=tmp_path/'report.json';write(p,dict(status='training_development_complete',test_accessed=False,
        pretest_seal_sha256=digest(seal)))
    cfg=dict(abl_training_development_completed=dict(path=str(p),sha256=digest(p)))
    with pytest.raises(RuntimeError,match='nine-case pretest audit required'):run(cfg,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_evaluation_reads_only_selected_models_and_does_not_claim_unperformed_compression(tmp_path):
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External/tests'))
    from test_abl_quality_core import fixture
    from abl_quality_core import array_ledger
    model,x,_,y=fixture(21,4,3);artifact=tmp_path/'model.npz';model.save(artifact)
    source=tmp_path/'data.npz';np.savez(source,features=x,target=y)
    split=tmp_path/'roles.npz';np.savez(split,test=np.arange(17,37))
    trajectory=tmp_path/'trajectory.json';write(trajectory,dict(cycles=[dict(block=0,action='ordinary',
        initial_from_representation=None,projection_seconds=0.,fit=None,error_state_bytes=0,cpu_process_peak_rss_bytes=0)],elapsed_seconds=0.))
    uid='full/b00';ledger=array_ledger(model)
    meta=dict(uid=uid,family='full',artifact=str(artifact),sha256=digest(artifact),ledger=ledger,
        cost_bytes=ledger['cost_bytes'],block=0,rank=None,nnz=0,rounds=3,leaves=len(model.base.coefficients),internal_nodes=len(model.base.nodes))
    row=dict(uid=uid,refit=dict(temperature=1.,raw_shifts=[0.]*4,temperature_shifts=[0.]*4))
    choices=[]
    for mode in ('raw','temperature','shifts','temperature_shifts'):
        for objective in ('loss','f1_macro'):
            for group in ('full_unbounded','native_budget','posthoc','fixed','joint','deployed'):
                chosen=dict(uid=uid,policy='raw',validation=dict(loss=1.,f1_macro=.2)) if group in ('full_unbounded','native_budget','deployed') else None
                choices.append(dict(mode=mode,objective=objective,group=group,budget_bytes=None if group=='full_unbounded' else 100000,chosen=chosen))
    case=dict(case_id='synthetic',record=dict(case=dict(split_file=str(split),split_sha256=digest(split),outputs=4),
        dataset=dict(path=str(source),sha256=digest(source))),budget_bytes=100000,rows=[row],choices=choices,
        metas={uid:meta,'unselected/missing':dict(artifact='must-not-be-read')},
        trajectories=[dict(mode='full',path=str(trajectory),sha256=digest(trajectory))])
    p=tmp_path/'case.json';write(p,case)
    result=evaluate_case({},dict(path=str(p),sha256=digest(p)),'synthetic-seal',tmp_path/'out')
    evaluated=json.loads(Path(result['path']).read_text())
    assert set(evaluated['predictions'])=={uid}
    assert len(evaluated['points'])==48
    assert all(not q['actual_compact_gradient_training'] for q in evaluated['points'] if q['status']=='evaluated')
