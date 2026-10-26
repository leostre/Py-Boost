"""Independent CSR oracle and generated allocation-plan laws; no fitting."""
from dataclasses import replace
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.shared_leaf_basis import LeafEnsemble, reference_class_model
from experiments.uniform_threshold_core import ThresholdModel
from experiments.traversal_leaf_gpu import traversal_plan


@given(st.integers(1,128),st.integers(0,20000),st.integers(1,1000),st.booleans())
@settings(max_examples=100,deadline=None)
def test_bounded_workspace_and_shared_memory(outputs,rows,trees,residual):
    for variant in ('warp','groups','tiled'):
        p=traversal_plan(variant,outputs,outputs,residual)
        assert p.threads%32==0 and p.shared_bytes>=outputs*8
        assert p.shared_bytes<=4*(2*128)*8
        expected=4*min(rows,256)*trees if variant=='tiled' else 0
        assert p.index_workspace_bytes(rows,trees)==expected
        if rows>=256:
            assert p.index_workspace_bytes(rows+1000000,trees)==expected


@pytest.mark.parametrize('args',[('bad',1,2,False),('warp',2,1,False),
    ('groups',1,129,True),('tiled',0,1,False)])
def test_reject_invalid_study_plans(args):
    with pytest.raises(ValueError): traversal_plan(*args)


def toy_model(task,role,c,r=1):
    rng=np.random.default_rng(72+c+r)
    nodes,values,noff,loff=[],[],[0],[0]
    for t in range(7):
        block=[[(-1 if t%2 else 1),.5,-1,1],[-2,0,-2,-3]] if t%3==0 else [[2,0,-1,-2]]
        m=3 if len(block)==2 else 2
        nodes.extend(block); values.extend(rng.normal(size=(m,c)))
        noff.append(len(nodes)); loff.append(len(values))
    dense=LeafEnsemble(np.array(nodes,np.float32),np.array(noff,np.int64),
        np.array(loff,np.int64),np.array(values,np.float32),np.empty((0,0),np.float32),
        rng.normal(size=c).astype(np.float32),2,task)
    base=dense
    if role=='reference': base=reference_class_model(dense)
    elif role in ('rank','hybrid'):
        basis=np.linalg.qr(rng.normal(size=(c,r)))[0].astype(np.float32)
        base=replace(dense,coefficients=rng.normal(size=(len(values),r)).astype(np.float32),
                     basis=basis,representation='low_rank')
    keys=rng.choice(len(values)*c,min(27,len(values)*c),replace=False) if role=='hybrid' else np.array([],int)
    rows,cols=np.unravel_index(keys,(len(values),c))
    return ThresholdModel(base,rows.astype(np.int32),cols.astype(np.int32),
        rng.normal(size=len(keys)).astype(np.float32),rng.normal(size=c).astype(np.float32)*.1)


GPU_CASES=[('multilabel','full',c,1) for c in (1,31,32,33,101,128)]+[
    ('multilabel','rank',33,1),('multilabel','rank',101,50),
    ('multilabel','hybrid',14,7),('multilabel','hybrid',101,50),
    ('multiclass','reference',100,1),('multiclass','full',101,1),
    ('multiclass','hybrid',100,49),('multiclass','rank',33,33)]


@pytest.mark.parametrize('task,role,c,r',GPU_CASES)
def test_gpu_traversals_match_independent_oracle(task,role,c,r):
    cp=pytest.importorskip('cupy')
    from experiments.traversal_leaf_gpu import TraversalLeafGpu
    from experiments.prepared_inference_worker import calibrated_decisions
    from experiments.sparse_leaf_core import leaf_design
    model=toy_model(task,role,c,r)
    x=np.resize(np.array([[0,np.nan],[.5,0],[.50001,1],[np.nan,-1],[1,0]],np.float32),(259,2))
    expected=model.predict(x)
    raw=model.raw_from_design(leaf_design(model.base.leaf_indices(x),len(model.base.coefficients)))
    for variant in ('warp','groups','tiled'):
        runtime=TraversalLeafGpu(model,variant)
        actual=runtime.predict(x)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-12)
        np.testing.assert_array_equal(calibrated_decisions(actual,model),calibrated_decisions(expected,model))
        np.testing.assert_allclose(runtime.raw_device(cp.asarray(x)).get(),raw,rtol=0,atol=2e-10)
        np.testing.assert_allclose(runtime.predict(x[::-1].copy()),expected[::-1],rtol=0,atol=2e-12)
        partition=np.concatenate([runtime.predict(x[:1]),runtime.predict(x[1:32]),runtime.predict(x[32:])])
        np.testing.assert_allclose(partition,actual,rtol=0,atol=2e-12)
        assert runtime.predict(np.empty((0,2),np.float32)).shape==(0,c)
        assert runtime._basis.dtype==cp.float32
        assert actual.dtype==np.float64


def test_gpu_input_contract_snapshot_and_request_buffers(monkeypatch):
    cp=pytest.importorskip('cupy')
    from experiments.traversal_leaf_gpu import TraversalLeafGpu
    for variant in ('warp','groups','tiled'):
        model=toy_model('multilabel','hybrid',33,7)
        x=np.zeros((3,2),np.float32)
        expected=model.predict(x)
        runtime=TraversalLeafGpu(model,variant)
        for value in (model.base.nodes,model.base.coefficients,model.base.basis,model.values): value.fill(0)
        np.testing.assert_allclose(runtime.predict(x),expected,rtol=0,atol=2e-12)
        for bad in (cp.zeros((3,2),cp.float64),cp.zeros((3,3),cp.float32),
                    cp.asarray([[np.inf,0]],dtype=cp.float32),cp.zeros((3,4),cp.float32)[:,::2]):
            with pytest.raises(ValueError): runtime.predict_device(bad)
        allocations=[]
        original=cp.empty
        def tracked(shape,*args,**kwargs):
            allocations.append(tuple(shape))
            return original(shape,*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(cp,'empty',tracked)
            output=runtime.predict_device(cp.asarray(x))
            output.get()
        assert allocations==([(3,33),(3,7)] if variant=='tiled' else [(3,33)])


def test_gpu_preallocated_stream_path_matches_regular_request():
    cp=pytest.importorskip('cupy')
    from experiments.traversal_leaf_gpu import TraversalLeafGpu,baseline_launch_into
    from experiments.unified_leaf_gpu import UnifiedLeafGpu
    model=toy_model('multiclass','hybrid',100,49)
    x=cp.asarray(np.zeros((259,2),np.float32))
    for variant in ('unified','warp','groups','tiled'):
        runtime=UnifiedLeafGpu(model) if variant=='unified' else TraversalLeafGpu(model,variant)
        expected=runtime.predict_device(x).get()
        out=cp.empty((259,100),cp.float64)
        if variant=='unified': baseline_launch_into(runtime,x,out)
        else: runtime.launch_into(x,out,workspace=runtime.workspace(len(x)))
        np.testing.assert_array_equal(out.get(),expected)
