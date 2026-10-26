import json
from dataclasses import asdict
import numpy as np
import pytest
from experiments.closed_budget_core import Topology,extend_bound
from experiments.cumulative_budget_core import absolute_allocations,serving_cost,CumulativeError
from experiments.cumulative_budget_projection import cumulative_proposal
from test_cumulative_budget_projection import dense,config


@pytest.mark.parametrize('seed',range(30))
def test_future_feasible_allocations_always_cover_actual_arrays(seed):
    rng=np.random.default_rng(seed)
    t=int(rng.integers(1,201))
    c=int(rng.integers(2,161))
    topology=Topology(t,7*t,8*t)
    future=extend_bound(topology,200-t,3)
    budget=int(rng.integers(100000,2000000))
    for a in absolute_allocations(future,c,c-1,budget):
        # The projection clips support to existing nonzero coordinates.
        assert serving_cost(topology,c,a.rank,min(a.nnz,topology.leaves*c)).maximum_bytes<=a.cost.maximum_bytes<=budget


def test_projection_records_actual_and_reserved_cost_separately():
    model=dense(3)
    x=(np.arange(40,dtype=np.float32)%2).reshape(-1,1)
    y=np.tile((np.arange(40)%2)[:,None],(1,4)).astype(np.float32)
    dev={k:(x.copy(),y.copy()) for k in ('train','calibration','control')}
    cfg=dict(config(),reserve_future_topology=True,serving_tree_horizon=200)
    accepted,info,_=cumulative_proposal(model,dev,100000,CumulativeError(np.zeros_like(y),'multilabel'),cfg)
    future=extend_bound(Topology(3,3,6),197,3)
    assert info['allocation_topology']==asdict(future) and accepted is not None
    for row in info['candidates']:
        assert row['max_serving_bytes']<row['reserved_max_serving_bytes']<=100000
    with pytest.raises(ValueError,match='horizon'):
        cumulative_proposal(model,dev,100000,CumulativeError(np.zeros_like(y),'multilabel'),dict(cfg,serving_tree_horizon=2))


def test_subset_wrapper_passes_only_registered_cases_and_seals_roles(tmp_path,monkeypatch):
    import experiments.absolute_variant_study as shell
    source=tmp_path/'source.json'
    shell.write_report(source,dict(cases=[dict(dataset='a',seed=1),dict(dataset='b',seed=2)]))
    cfg=dict(variant_source_roles_manifest=str(source),variant_source_roles_manifest_sha256=shell.sha256(source),
             variant_cases=['b_2'])
    calls=[]
    def train(config,report,output):
        path=shell.Path(config['cumulative_roles_manifest'])
        assert shell.sha256(path)==config['cumulative_roles_manifest_sha256']
        assert json.loads(path.read_text())['cases']==[dict(dataset='b',seed=2)]
        calls.append(config)
    monkeypatch.setattr(shell,'absolute_history_training',train)
    monkeypatch.setattr(shell.subprocess,'run',lambda *a,**k:None)
    shell.absolute_variant_training(cfg,{},tmp_path)
    assert len(calls)==1
    with pytest.raises(RuntimeError,match='incomplete'):
        shell.absolute_variant_training(dict(cfg,variant_cases=['c_3']),{},tmp_path)
