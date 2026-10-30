"""Toy integration contract: no learner/GPU, real transforms/export/selection."""
import json
from pathlib import Path
import numpy as np
import broad_training as worker
from budget_quality_worker import digest,write
from abl_budget_revision_core import blank
from experiments.shared_leaf_basis import LeafEnsemble


def test_complete_broad_case_with_fake_fit_and_real_catalogue(tmp_path,monkeypatch):
    rng=np.random.default_rng(9);x=rng.normal(size=(65,3)).astype(np.float32);y=rng.integers(0,2,(65,6)).astype(np.float32)
    data=tmp_path/'data.npz';np.savez(data,features=x,target=y)
    split=tmp_path/'split.npz';np.savez(split,train=np.arange(40),calibration=np.arange(40,45),control=np.arange(45,50),selection=np.arange(50,55),test=np.arange(55,65))
    cc=dict(dataset='toy',seed=1,task='multilabel',outputs=6,split_file=str(split),split_sha256=digest(split))
    manifest=tmp_path/'manifest.json';write(manifest,dict(cases=[cc],datasets={'toy':dict(path=str(data),sha256=digest(data))}))
    preflight=tmp_path/'preflight.json';write(preflight,dict(status='broad_preflight_complete'))
    def fake_fit(request):
        trees=request['trees'];c=6;basis=np.asarray(request.get('basis',[]),np.float32)
        r=basis.shape[1] if basis.size else c
        nodes=np.tile(np.array([[1.,0.,-1.,-2.]],np.float32),(trees,1))
        a=np.tile(np.linspace(-.03,.04,2*r,dtype=np.float32).reshape(2,r),(trees,1))
        base=LeafEnsemble(nodes,np.arange(trees+1,dtype=np.int64),2*np.arange(trees+1,dtype=np.int64),a,
             basis if basis.size else np.empty((0,0),np.float32),np.zeros(c,np.float32),3,'multilabel','low_rank' if basis.size else 'dense')
        model=blank(base);p=Path(request['output']);model.save(p)
        return dict(artifact=str(p),sha256=digest(p),trees=trees,seconds=0.)
    monkeypatch.setattr(worker,'check_native_environment',lambda c:{'test_fixture':True})
    monkeypatch.setattr(worker,'grouped_gate',lambda:[])
    monkeypatch.setattr(worker,'fit_training_increment',fake_fit)
    monkeypatch.setattr(worker,'fit_direct',fake_fit)
    cfg=dict(lr=.1,decision_grid=[0,-.5,.5],decision_shrinkage=[.5,1],broad=dict(
        manifest=dict(path=str(manifest),sha256=digest(manifest)),preflight=dict(path=str(preflight),sha256=digest(preflight)),
        maximum_new_trees=42000,prior_paid_new_trees=24,protocol={'toy':True}))
    out=tmp_path/'run';worker.run(cfg,out)
    report=json.loads((out/'report.json').read_text());assert report['paid_new_trees']==1104
    assert report['test_evaluated'] is False
    case=json.loads(Path(report['cases'][0]['path']).read_text())
    assert all(q<=24 for q in case['queries'].values())
    assert len(case['models'])>100
    assert all(m['sha256']==digest(m['artifact']) for m in case['models'])
    assert any(m['family']=='direct_ABL' and m['trees']==200 for m in case['models'])
