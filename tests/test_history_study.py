import json
import subprocess
import sys
from dataclasses import replace
import numpy as np
import pytest
from experiments.block_leaf_model import BlockLeafModel,load_prediction_model
from experiments.history_basis_study import seal_model,evaluate_case,verify_prefix,load_parent
from experiments.online_leaf_study import raw
from experiments.a100_runner import sha256,write_report


def test_sealing_mixed_formats_reload_and_quality_guard(tmp_path):
    from test_prepared_leaf_inference import model_for
    full = model_for('multilabel','full')
    zero = np.zeros_like(full.base.bias)
    block = replace(full,base=replace(full.base,bias=zero),decision_shifts=zero)
    bank = BlockLeafModel((block,),full.base.bias.copy(),zero)
    x = np.tile(np.array([[0,np.nan],[1,1],[np.nan,0],[.5,1]],np.float32),(3,1))
    y = full.predict_labels(x)
    directory = tmp_path/'toy_1'
    directory.mkdir()
    split = dict(calibration=np.arange(4),selection=np.arange(4,8),validation=np.arange(8),test=np.arange(8,12))
    np.savez(directory/'splits.npz',**split)
    config = dict(decision_grid=[0,-.5,.5],decision_shrinkage=[.25,.5,1.])
    replay = dict(features=x[split['validation']])
    choices = [seal_model(name,model,directory,tmp_path,x,y,split['calibration'],split['selection'],
                         split['validation'],config,replay) for name,model in [('online_full',full),('history',bank)]]
    np.savez(directory/'replay.npz',**replay)
    case = dict(dataset='toy',seed=1,task='multilabel',choices=choices)
    write_report(directory/'selection.json',case)
    result = subprocess.run([sys.executable,'-m','experiments.history_basis_verify',str(directory)],
                            capture_output=True,text=True,check=True,timeout=60)
    assert 'history_cpu_replay_passed' in result.stdout
    rows = evaluate_case(case,tmp_path,x,y)
    assert all(r['quality_pass'] and r['validation_quality_pass'] for r in rows)
    for choice in choices:
        model = load_prediction_model(tmp_path/choice['artifact'])
        assert choice['parameters'] == model.parameter_count and choice['payload_bytes'] == model.payload_bytes
    altered = {**case,'choices':[choices[0],{**choices[1],
        'validation':{**choices[1]['validation'],'loss':2*choices[0]['validation']['loss']}}]}
    assert not evaluate_case(altered,tmp_path,x,y)[1]['validation_quality_pass']


def test_prefix_guard_rejects_reinterpreted_old_coordinates():
    from test_prepared_leaf_inference import model_for
    fixed = model_for('multilabel','rank')
    zero = np.zeros_like(fixed.base.bias)
    first = replace(fixed,base=replace(fixed.base,bias=zero),decision_shifts=zero)
    bank = BlockLeafModel((first,),fixed.base.bias.copy(),zero)
    verify_prefix(bank,fixed,2)
    rotated = replace(first,base=replace(first.base,basis=-first.base.basis))
    with pytest.raises(AssertionError):
        verify_prefix(BlockLeafModel((rotated,),bank.bias,zero),fixed,2)


def test_frozen_parent_loader_checks_models_inputs_and_config(tmp_path):
    from test_prepared_leaf_inference import model_for
    directory = tmp_path/'toy_1'
    directory.mkdir()
    model = model_for('multilabel','full')
    choices = []
    for method in ('online_full','online_hybrid','online_more_rank'):
        path = directory/f'{method}.npz'
        model.save(path)
        choices.append(dict(method=method,artifact=f'toy_1/{method}.npz',sha256=sha256(path)))
    np.savez(directory/'splits.npz',train=np.array([0]),validation=np.array([1]),test=np.array([2]))
    parameters = dict(ntrees=200,lr=.1,max_depth=5,lambda_l2=1.,online_warmup=20,online_rank_fraction=.5,
        online_residual_fraction=.05,decision_grid=[0],decision_shrinkage=[1.])
    case = dict(dataset='toy',seed=1,task='multilabel',choices=choices,split_sha256=sha256(directory/'splits.npz'))
    report = dict(status='done',scientific_fits=20,environment_sha256='env',config=parameters,file_locks={},cases=[case])
    write_report(tmp_path/'report.json',report)
    config = dict(**parameters,online_sources=dict(toy=str(tmp_path)),
        online_report_hashes=dict(toy=sha256(tmp_path/'report.json')),expected_environment_sha256='env')
    locks = {}
    _,_,controls,dense = load_parent('toy',1,config,locks)
    assert set(controls) == {'online_full','fixed_hybrid','fixed_more_rank'}
    assert len(locks) == 5 and dense.base.representation == 'dense'
    with pytest.raises(AssertionError):
        load_parent('toy',1,{**config,'lr':.2},{})
    with (directory/'online_hybrid.npz').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(RuntimeError,match='model changed'):
        load_parent('toy',1,config,{})
