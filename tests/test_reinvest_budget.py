import numpy as np
import pytest
from test_cumulative_budget_projection import dense,config
from experiments.uniform_threshold_core import ThresholdModel


def test_400_path_prices_every_fit_and_keeps_actual_feedback(tmp_path,monkeypatch):
    from experiments.cumulative_budget_study import train_cumulative_path
    from experiments import parallel_leaf_worker
    requests=[]
    def fit(req):
        model=ThresholdModel(dense(req['trees']),np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),np.zeros(4,np.float32))
        model.save(req['output'])
        requests.append(req)
        return dict(trees=req['trees'])
    monkeypatch.setattr(parallel_leaf_worker,'fit_increment',fit)
    x=(np.arange(40,dtype=np.float32)%2).reshape(-1,1)
    y=np.tile((np.arange(40)%2)[:,None],(1,4)).astype(np.float32)
    development={k:(x.copy(),y.copy()) for k in ('train','calibration','control')}
    case=dict(dataset='toy',seed=1,task='multilabel',outputs=4,absolute_budget_bytes=1000000)
    cfg=dict(config(),serving_tree_horizon=400,reserve_future_topology=True,cumulative_pilot_tree_cap=400)
    ledger=dict(trees=0)
    snapshots,trajectory=train_cumulative_path('cumulative_reserve',case,development,cfg,tmp_path/'path',ledger)
    assert len(requests)==len(snapshots)==10 and ledger['trees']==400
    assert snapshots[-1][1].base.n_trees==trajectory['trees_built']==400
    assert all(c['projection']['reserved_horizon']==400 for c in trajectory['cycles'])
    for bad in (0,399,True):
        with pytest.raises(ValueError,match='horizon'):
            train_cumulative_path('cumulative_reserve',case,development,dict(cfg,serving_tree_horizon=bad),tmp_path/str(bad),dict(trees=0))
