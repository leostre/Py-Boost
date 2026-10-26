import json
import pytest
from test_cumulative_budget_projection import run_synthetic_shell
from experiments import cumulative_budget_audit as audit


def synthetic_source(monkeypatch,tmp_path):
    study,cfg,report,output,calls=run_synthetic_shell(monkeypatch,tmp_path)
    study.cumulative_training(cfg,report,output)
    path=output/'cumulative_development_manifest.json'
    cfg=dict(cfg,cumulative_development_manifest=str(path),cumulative_development_manifest_sha256=study.sha256(path))
    return study,cfg,path


def test_independent_saved_feedback_ledger_and_frozen_test_replay(monkeypatch,tmp_path):
    study,cfg,path=synthetic_source(monkeypatch,tmp_path)
    result={}
    out=tmp_path/'audit'
    out.mkdir()
    audit.cumulative_final_test(cfg,result,out)
    assert result['audit']['trees']==1200
    assert result['audit']['checkpoints']==15
    assert result['audit']['cumulative_error_max_abs_error']==0
    assert result['audit']['accumulated_fp32_materialization_drift_max_abs']==0
    assert len(result['results'])==5 and all(r['joint_success'] for r in result['results'])
    assert result['new_fits']==0 and (out/'pretest_audit.json').exists()


def test_unaccepted_frozen_decision_cannot_reach_test_scoring(monkeypatch,tmp_path):
    study,cfg,path=synthetic_source(monkeypatch,tmp_path)
    source=json.loads(path.read_text(encoding='utf-8'))
    source['records'][0]['decisions'][0]['state']='no_quality_feasible'
    path.write_text(json.dumps(source),encoding='utf-8')
    cfg['cumulative_development_manifest_sha256']=study.sha256(path)
    test_reads=[]
    original=audit.calibrated_quality
    def spy(y,*args,**kwargs):
        if len(y)==32: test_reads.append(True)
        return original(y,*args,**kwargs)
    monkeypatch.setattr(audit,'calibrated_quality',spy)
    out=tmp_path/'audit'
    out.mkdir()
    with pytest.raises(RuntimeError,match='final frozen selection changed'):
        audit.cumulative_final_test(cfg,{},out)
    assert not test_reads and not (out/'pretest_audit.json').exists()
