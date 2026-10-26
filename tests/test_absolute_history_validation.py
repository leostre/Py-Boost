import copy
import pytest
import json
import numpy as np
from experiments.absolute_history_validation import case_cost,match_reused_control


def test_reuse_matches_immutable_controls_and_real_fit_prices():
    control=dict(full=dict(sha256='fixed'),reference_validation=dict(f1_macro=.5),decisions=[dict(method='periodic')])
    record=dict(control,decisions=[*control['decisions'],dict(method='gh_mean')],
        trajectories=[dict(cycles=[dict(fit=dict(trees=40)),dict(state='rejected',fit=dict(trees=40))])],full_fits=[])
    match_reused_control(record,control)
    assert case_cost(record)==80
    for key in ('full','reference_validation','decisions'):
        changed=copy.deepcopy(control)
        changed[key]=[] if key=='decisions' else {}
        with pytest.raises(RuntimeError,match='changed full/baseline'):
            match_reused_control(record,changed)


def test_wrapper_trains_only_fresh_roles_and_merges_original_order(tmp_path,monkeypatch):
    import experiments.absolute_history_validation as shell
    roles=dict(cases=[dict(dataset='a',seed=1),dict(dataset='b',seed=2)])
    def record(name,seed,trees):
        return dict(dataset=name,seed=seed,full=dict(sha256=name),reference_validation=dict(loss=.2),
                    decisions=[dict(method='periodic')],full_fits=[],
                    trajectories=[dict(cycles=[dict(fit=dict(trees=trees))])])
    controls=dict(records=[record('a',1,200),record('b',2,200)])
    pilot=dict(records=[dict(record('a',1,40),decisions=[dict(method='periodic'),dict(method='gh_mean')])])
    config=dict(expected_reused_history_trees=40)
    for key,value in (('absolute_roles_manifest',roles),('cumulative_development_manifest',controls),
                      ('history_pilot_manifest',pilot)):
        path=tmp_path/(key+'.json')
        shell.write_report(path,value)
        config[key]=str(path)
        config[key+'_sha256']=shell.sha256(path)
    monkeypatch.setattr(shell,'load_expanded',lambda *args:(np.zeros((3,2)),np.zeros(3)))
    monkeypatch.setattr(shell,'load_roles',lambda *args:dict(train=np.arange(3)))
    calls=[]
    def train(cfg,report,out):
        selected=json.loads(shell.Path(cfg['cumulative_roles_manifest']).read_text(encoding='utf-8'))
        assert selected['cases']==[roles['cases'][1]]
        assert shell.sha256(cfg['cumulative_roles_manifest'])==cfg['cumulative_roles_manifest_sha256']
        calls.append(out)
        report.update(records=[record('b',2,80)],tree_fits=80,gpu_observations=[])
    monkeypatch.setattr(shell,'absolute_history_training',train)
    out=tmp_path/'out'
    out.mkdir()
    report={}
    shell.expanded_history_training(config,report,out)
    merged=json.loads((out/'cumulative_development_manifest.json').read_text(encoding='utf-8'))
    assert len(calls)==1 and merged['tree_fits']==80 and merged['reused_tree_fits']==40
    assert merged['baseline_training_cost']==400 and not merged['test_evaluated']
    assert [r['dataset'] for r in merged['records']]==['a','b']
    assert merged['records'][0]['reused'] is True
    assert report['new_tree_fits']==80 and report['reused_tree_fits']==40
    controls['records'].append(controls['records'][0])
    shell.write_report(shell.Path(config['cumulative_development_manifest']),controls)
    config['cumulative_development_manifest_sha256']=shell.sha256(config['cumulative_development_manifest'])
    with pytest.raises(RuntimeError,match='duplicate'):
        shell.expanded_history_training(config,{},out)
    assert len(calls)==1
