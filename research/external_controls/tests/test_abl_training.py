from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External/tests'))
import budget_quality_worker
from abl_training_core import training_allocations,reserved_depth
from abl_training_projection import training_proposal
from abl_quality_core import fit_representation_family,prepare_rank,build_representation,RepresentationPlan
from experiments.closed_budget_core import Topology
from experiments.cumulative_budget_core import CumulativeError,serving_cost
from test_abl_quality_core import fixture


def test_parent_relative_residual_and_real_serialized_array_envelope():
    model,x,design,y=fixture(41,5,8)
    state=fit_representation_family(model,design,y)
    topology=Topology(model.base.n_trees,len(model.base.nodes),len(model.base.coefficients))
    parent=len(model.base.coefficients)*4+10
    for budget in (300,600,1200,4000):
        for allocation in training_allocations(topology,5,4,budget,parent):
            assert allocation.nnz<=allocation.cap_percent*parent//100
            pure,residual,order=prepare_rank(model,design,state,allocation.rank)
            fitted=build_representation(pure,residual,order,state,
                RepresentationPlan(allocation.rank,allocation.cap_percent/100,allocation.nnz))
            actual=max(fitted.payload_bytes,fitted.prepare_inference().resident_bytes)
            assert actual<=allocation.cost.maximum_bytes<=budget
    fixed=training_allocations(topology,5,4,4000,parent,fixed=True)
    assert all(a.rank==2 and a.cap_percent==5 for a in fixed)
    with pytest.raises(ValueError):training_allocations(topology,5,4,4000,True)


def test_projection_is_calibration_aware_conserves_input_and_rejects_selection_role():
    model,x,_,y=fixture(13,4,5)
    development={'train':(x[:21],y[:21]),'calibration':(x[21:29],y[21:29]),'control':(x[29:],y[29:])}
    error=CumulativeError(np.zeros((21,4)),'multiclass')
    cfg=dict(margin_scale=.5,decision_grid=[0,.25,-.25,.5,-.5,1,-1,2,-2],decision_shrinkage=[.25,.5,1])
    before=model.base.coefficients.copy()
    result,info,new_error=training_proposal(model.base,development,4000,error,cfg,criterion='cumulative')
    assert result is not None and 'temperature' in info['chosen']
    assert max(result.payload_bytes,result.prepare_inference().resident_bytes)<=4000
    np.testing.assert_array_equal(model.base.coefficients,before)
    assert new_error.updates==1 and error.updates==0
    assert not np.shares_memory(new_error.values,error.values)
    with pytest.raises(ValueError,match='only train/calibration/control'):
        training_proposal(model.base,dict(development,selection=(x,y)),4000,error,cfg)


def test_future_reservation_is_conservative_and_depth_monotone():
    topology=Topology(20,100,120)
    previous=0
    for budget in (1000,10000,100000,1000000,10000000):
        depth=reserved_depth(topology,80,12,3,budget)
        assert (depth or 0)>=previous;previous=depth or 0
        if depth is not None:
            from experiments.closed_budget_core import extend_bound
            assert serving_cost(extend_bound(topology,80,depth),12,3,0).maximum_bytes<=budget
