import copy
import numpy as np
import pytest
from experiments.absolute_reserve_validation import keyed,baseline_equal,tree_cost


def test_cohort_baselines_and_actual_pilot_price():
    old=dict(dataset='a',seed=1,full=dict(sha256='same'),reference_validation=dict(loss=.3),
             decisions=[dict(method='periodic')],trajectories=[],full_fits=[])
    pilot=dict(old,decisions=[*old['decisions'],dict(method='local_reserve'),dict(method='cumulative_reserve')],
               trajectories=[dict(cycles=[dict(fit=dict(trees=40)),dict(fit=dict(trees=40))])])
    baseline_equal(pilot,old)
    assert tree_cost(pilot)==80
    with pytest.raises(RuntimeError,match='duplicate'): keyed([old,old])
    for key in ('full','reference_validation','decisions'):
        changed=copy.deepcopy(old)
        changed[key]=[] if key=='decisions' else {}
        with pytest.raises(RuntimeError,match='baselines'): baseline_equal(pilot,changed)


def test_whole_wrapper_reuses_secondary_pilot_cost_and_trains_only_new(tmp_path,monkeypatch):
    import experiments.absolute_reserve_validation as shell
    monkeypatch.setattr(shell.subprocess,'run',lambda *a,**k:None)
    role=dict(cases=[dict(dataset='a',seed=1),dict(dataset='b',seed=2)])
    def row(name,seed,trees):
        return dict(dataset=name,seed=seed,full=dict(sha256=name),reference_validation=dict(loss=.2),
                    decisions=[dict(method='periodic')],full_fits=[],
                    trajectories=[dict(cycles=[dict(fit=dict(trees=trees))])])
    controls=dict(records=[row('a',1,200),row('b',2,200)])
    pilot=dict(records=[dict(row('a',1,80),decisions=[dict(method='periodic'),dict(method='cumulative_reserve')])])
    cfg=dict(reserve_expected_reused_trees=80)
    for key,value in (('reserve_roles_manifest',role),('cumulative_development_manifest',controls),('reserve_pilot_manifest',pilot)):
        path=tmp_path/(key+'.json')
        shell.write_report(path,value)
        cfg[key]=str(path)
        cfg[key+'_sha256']=shell.sha256(path)
    monkeypatch.setattr(shell,'load_expanded',lambda *a:(np.zeros((3,1)),np.zeros(3)))
    monkeypatch.setattr(shell,'load_roles',lambda *a:dict(train=np.arange(3)))
    calls=[]
    def train(config,report,path):
        import json
        assert json.loads(shell.Path(config['cumulative_roles_manifest']).read_text())['cases']==[role['cases'][1]]
        calls.append(path)
        report.update(records=[row('b',2,160)],tree_fits=160,gpu_observations=[])
    monkeypatch.setattr(shell,'absolute_history_training',train)
    out=tmp_path/'out'
    out.mkdir()
    result={}
    shell.reserve_validation_training(cfg,result,out)
    assert len(calls)==1 and result['new_tree_fits']==160 and result['reused_tree_fits']==80
