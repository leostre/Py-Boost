import json
from pathlib import Path
import numpy as np
import pytest
from test_cumulative_budget_projection import run_synthetic_shell
from experiments import absolute_validation_study as validation


def synthetic_parent(monkeypatch,tmp_path):
    study,cfg,parent_report,parent_output,calls=run_synthetic_shell(monkeypatch,tmp_path)
    study.cumulative_training(cfg,parent_report,parent_output)
    calls.clear()
    parent=parent_output/'cumulative_development_manifest.json'
    roles=json.loads(Path(cfg['cumulative_roles_manifest']).read_text(encoding='utf-8'))
    old=roles['cases'][0]
    roles['cases'].append(dict(old,seed=2))
    expanded=tmp_path/'expanded.json'
    study.write_report(expanded,roles)
    cfg.update(absolute_roles_manifest=str(expanded),absolute_roles_manifest_sha256=study.sha256(expanded),
        cumulative_development_manifest=str(parent),cumulative_development_manifest_sha256=study.sha256(parent),
        expected_reused_tree_fits=1200)
    return study,cfg,calls,expanded


def test_reuse_exact_roles_and_account_only_new_tree_fits(monkeypatch,tmp_path):
    study,cfg,calls,expanded=synthetic_parent(monkeypatch,tmp_path)
    out=tmp_path/'validation'
    out.mkdir()
    report={}
    validation.absolute_validation_training(cfg,report,out)
    assert report['cases']==2 and report['new_tree_fits']==1200 and report['reused_tree_fits']==1200
    assert report['total_tree_fits_including_reused']==2400 and sum(n for _,n in calls)==1200
    manifest=json.loads((out/'cumulative_development_manifest.json').read_text(encoding='utf-8'))
    assert manifest['records'][0]['reused'] and not manifest['records'][1].get('reused')
    assert not manifest['test_evaluated']


def test_changed_calibration_ids_abort_before_any_new_fit(monkeypatch,tmp_path):
    study,cfg,calls,expanded=synthetic_parent(monkeypatch,tmp_path)
    source=json.loads(expanded.read_text(encoding='utf-8'))
    case=source['cases'][0]
    with np.load(case['split_file']) as f: roles={k:f[k].copy() for k in f.files}
    roles['train'][0],roles['calibration'][0]=roles['calibration'][0],roles['train'][0]
    changed=tmp_path/'changed.npz'
    np.savez(changed,**roles)
    source['cases'][0]=dict(case,split_file=str(changed),split_sha256=study.sha256(changed))
    study.write_report(expanded,source)
    cfg['absolute_roles_manifest_sha256']=study.sha256(expanded)
    out=tmp_path/'validation'
    out.mkdir()
    with pytest.raises(RuntimeError,match='changed development roles'):
        validation.absolute_validation_training(cfg,{},out)
    assert not calls
