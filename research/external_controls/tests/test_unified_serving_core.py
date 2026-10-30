from dataclasses import replace
from itertools import product
import numpy as np
import pytest
from test_abl_quality_core import fixture
from abl_protected_core import as_protected,append_protected
from unified_serving_core import coalesce,bank_frontier


@pytest.mark.parametrize('classes',[4,159])
def test_full_coalescing_preserves_function_and_charges_shared_arrays_once(classes):
    a,x,_,_=fixture(4,classes,5);b,_,_,_=fixture(17,classes,3)
    model=append_protected(as_protected(a),b);joined=coalesce(model)
    assert len(model.blocks)==2 and len(joined.blocks)==1
    np.testing.assert_allclose(joined.raw(x),model.raw(x),rtol=0,atol=1e-12)
    assert joined.n_trees==model.n_trees
    assert joined.resident_array_bytes<model.resident_array_bytes


def test_bank_allocation_matches_bruteforce_including_infeasible_budget():
    cases=[[dict(uid=f'{i}-{j}',cost_bytes=7+j*5,utility=(i+1)*j*.1) for j in range(3)] for i in range(4)]
    budgets=[20,28,39,58,68]
    rows=bank_frontier(cases,budgets)
    for row,budget in zip(rows,budgets):
        possible=[p for p in product(*cases) if sum(r['cost_bytes'] for r in p)<=budget]
        if not possible:assert row['status']=='infeasible';continue
        best=max(sum(r['utility'] for r in p)/4 for p in possible)
        assert row['utility']==pytest.approx(best)


def test_sparse_lowrank_coalescing_shifts_residual_rows_and_keeps_distinct_decoders():
    from experiments.block_leaf_model import BlockLeafModel
    from experiments.uniform_threshold_core import ThresholdModel
    a,x,_,_=fixture(4,4,3);b,_,_,_=fixture(17,4,2)
    basis=np.linalg.qr(np.random.default_rng(5).normal(size=(4,2)))[0].astype(np.float32)
    def block(model,basis):
        base=replace(model.base,coefficients=model.base.coefficients[:,:2].copy(),basis=basis,representation='low_rank',bias=np.zeros(4,np.float32))
        return ThresholdModel(base,np.array([0,1],np.int32),np.array([2,3],np.int32),np.array([.4,-.2],np.float32),np.zeros(4,np.float32))
    first=block(a,basis);second=block(b,basis);third=block(a,-basis)
    model=BlockLeafModel((first,second,third),a.base.bias.copy(),np.zeros(4,np.float32))
    joined=coalesce(model);assert len(joined.blocks)==2
    x=np.r_[x,[[np.nan],[.25]]].astype(np.float32)
    np.testing.assert_allclose(model.raw(x),joined.raw(x),atol=1e-12,rtol=0)
    assert list(joined.blocks[0].rows)==[0,1,len(first.base.coefficients),len(first.base.coefficients)+1]
