import numpy as np
from test_cumulative_budget_projection import run_synthetic_shell
from experiments import absolute_history_study as history
from experiments import cumulative_budget_audit as audit


def test_history_shell_reuses_controls_seals_and_independently_audits(monkeypatch,tmp_path):
    study,cfg,parent_report,parent_output,calls=run_synthetic_shell(monkeypatch,tmp_path)
    study.cumulative_training(cfg,parent_report,parent_output)
    manifest=parent_output/'cumulative_development_manifest.json'
    calls.clear()
    cfg=dict(cfg,cumulative_development_manifest=str(manifest),cumulative_development_manifest_sha256=study.sha256(manifest),
        cumulative_methods=['gh_current','gh_mean','gh_history'],cumulative_pilot_tree_cap=600,history_beta=.8)
    out=tmp_path/'history'
    out.mkdir()
    report={}
    history.absolute_history_training(cfg,report,out)
    assert calls==[('increment',40)]*15 and report['tree_fits']==600
    assert len(report['records'][0]['decisions'])==8 and not report['test_evaluated']
    for trajectory in report['records'][0]['trajectories']:
        assert trajectory['accepted_checkpoints']==5
        assert [c['projection']['basis_source']['observations'] for c in trajectory['cycles']]==list(range(1,6))
    path=out/'cumulative_development_manifest.json'
    cfg.update(cumulative_development_manifest=str(path),cumulative_development_manifest_sha256=study.sha256(path))
    testout=tmp_path/'history_audit'
    testout.mkdir()
    results={}
    audit.cumulative_final_test(cfg,results,testout)
    assert results['audit']['trees']==600 and results['audit']['checkpoints']==15
    assert len(results['results'])==8 and all(r['joint_success'] for r in results['results'])
