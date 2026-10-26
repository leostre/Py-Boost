"""Real pure projections and synthetic effect-shell replay; never train on GPU."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from experiments.shared_leaf_basis import LeafEnsemble
from experiments.uniform_threshold_core import ThresholdModel
from experiments.cumulative_budget_core import CumulativeError
from experiments.cumulative_budget_projection import cumulative_proposal


def dense(trees,c=4):
    return LeafEnsemble(np.tile(np.array([[1,0,-1,-2]],np.float32),(trees,1)),
        np.arange(trees+1,dtype=np.int64),2*np.arange(trees+1,dtype=np.int64),
        np.zeros((2*trees,c),np.float32),np.empty((0,0),np.float32),np.zeros(c,np.float32),1,'multilabel')


def config():
    return dict(margin_scale=.5,decision_grid=[-1,0,1],decision_shrinkage=[.5,1],lr=.1,
                own_pool_limit_bytes=1024**3,cumulative_pilot_tree_cap=1200,
                cumulative_methods=['periodic','local','cumulative'],bounded_tree_execution=True,
                max_depth=5,lambda_l2=1,ntrees=200)


def test_same_candidate_list_different_ranking_and_no_extra_role_accepted():
    model=dense(3)
    x=np.arange(40,dtype=np.float32).reshape(-1,1)%2
    y=np.tile((np.arange(40)%2)[:,None],(1,4)).astype(np.float32)
    development={k:(x.copy(),y.copy()) for k in ('train','calibration','control')}
    state=CumulativeError(np.zeros((len(x),4)),'multilabel')
    one,info1,e1=cumulative_proposal(model,development,100000,state,config(),criterion='local')
    two,info2,e2=cumulative_proposal(model,development,100000,state,config(),criterion='cumulative')
    assert one is not None and two is not None
    assert info1['candidates']==info2['candidates']
    assert max(one.payload_bytes,one.prepare_inference().resident_bytes)<=100000
    np.testing.assert_array_equal(state.values,np.zeros((len(x),4)))
    assert e1.updates==e2.updates==1
    with pytest.raises(ValueError,match='only train'):
        cumulative_proposal(model,dict(development,test=(x,y)),100000,state,config())


def run_synthetic_shell(monkeypatch,tmp_path,cap=1200):
    from experiments import cumulative_budget_study as study
    from experiments import parallel_leaf_worker,leaf_budget_runtime,inference_gpu_guard
    x=np.arange(160,dtype=np.float32).reshape(-1,1)%2
    y=np.tile((np.arange(160)%2)[:,None],(1,4)).astype(np.float32)
    roles=dict(train=np.arange(72),calibration=np.arange(72,88),control=np.arange(88,104),
               selection=np.arange(104,128),test=np.arange(128,160))
    split=tmp_path/'splits.npz'
    np.savez(split,**roles)
    data=tmp_path/'data.npz'
    np.savez(data,features=x,target=y)
    case=dict(dataset='toy',seed=1,task='multilabel',outputs=4,absolute_budget_bytes=1000000,
              split_file=str(split),split_sha256=study.sha256(split),
              source_split_file=str(split),source_split_sha256=study.sha256(split))
    source=tmp_path/'source.json'
    source.write_text(json.dumps(dict(cases=[case],datasets={'toy':dict(path=str(data),sha256=study.sha256(data))})))
    cfg=dict(config(),cumulative_roles_manifest=str(source),cumulative_roles_manifest_sha256=study.sha256(source),
             cumulative_pilot_tree_cap=cap)
    calls=[]
    def increment(request):
        with np.load(request['data_path']) as f:
            assert set(f.files)=={'train_x','train_y','validation_x','validation_y'}
            np.testing.assert_array_equal(f['train_y'],y[roles['train']])
            np.testing.assert_array_equal(f['validation_y'],y[roles['calibration']])
        model=ThresholdModel(dense(request['trees']),np.empty(0,np.int32),np.empty(0,np.int32),
                             np.empty(0,np.float32),np.zeros(4,np.float32))
        model.save(request['output'])
        calls.append(('increment',request['trees']))
        return dict(trees=request['trees'],seconds=0.,artifact=request['output'],sha256=study.sha256(request['output']),
                    data_sha256=study.sha256(request['data_path']),initial_sha256=study.sha256(request['initial_path']))
    def full(case,depth,cfg,xx,yy,rr):
        assert all(kind=='increment' for kind,_ in calls[:15])
        np.testing.assert_array_equal(yy[roles['test']],np.zeros_like(y[roles['test']]))
        np.testing.assert_array_equal(yy[roles['selection']],np.zeros_like(y[roles['selection']]))
        calls.append(('full',200))
        return dense(200),dict(depth=depth,trees=200,seconds=0.)
    class Observer:
        samples=[]
        def __enter__(self): return self
        def __exit__(self,*args): return False
    monkeypatch.setattr(study,'load_expanded',lambda *args:(x.copy(),y.copy()))
    monkeypatch.setattr(parallel_leaf_worker,'fit_increment',increment)
    monkeypatch.setattr(leaf_budget_runtime,'fit_dense',full)
    monkeypatch.setattr(inference_gpu_guard,'SharedGpuObserver',Observer)
    monkeypatch.setattr(study.subprocess,'run',lambda *args,**kwargs:None)
    report={}
    output=tmp_path/'out'
    output.mkdir()
    return study,cfg,report,output,calls


def test_shell_fit_order_tree_counts_final_seal_and_test_mask(monkeypatch,tmp_path):
    study,cfg,report,output,calls=run_synthetic_shell(monkeypatch,tmp_path)
    study.cumulative_training(cfg,report,output)
    assert calls==[('increment',40)]*15+[('full',200)]*3
    assert report['tree_fits']==1200 and not report['test_evaluated']
    record=report['records'][0]
    assert len(record['decisions'])==5
    assert all(d['state']=='accepted' for d in record['decisions'])
    assert all(d['coo_bytes']<=1000000 and d['csr_bytes']<=1000000 for d in record['decisions'])
    assert (output/'cumulative_development_manifest.json').exists()


def test_tree_cap_stops_before_forbidden_fit_effect(monkeypatch,tmp_path):
    study,cfg,report,output,calls=run_synthetic_shell(monkeypatch,tmp_path,cap=1199)
    with pytest.raises(RuntimeError,match='tree cap exceeded before full control'):
        study.cumulative_training(cfg,report,output)
    assert sum(n for _,n in calls)==1000
    assert calls.count(('full',200))==2
