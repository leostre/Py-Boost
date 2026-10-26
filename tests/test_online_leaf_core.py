from dataclasses import replace
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.online_leaf_core import weighted_projection, compact_topology, project_part, stitch


@st.composite
def projection_cases(draw):
    c = draw(st.integers(2,10))
    r = draw(st.integers(1,c))
    m = draw(st.integers(1,6))
    flat = draw(st.lists(st.floats(-3,3,allow_nan=False,allow_infinity=False),min_size=m*c,max_size=m*c))
    weights = draw(st.lists(st.floats(.1,4,allow_nan=False,allow_infinity=False),min_size=m*c,max_size=m*c))
    entries = draw(st.lists(st.floats(-.5,.5,allow_nan=False,allow_infinity=False),min_size=c*c,max_size=c*c))
    matrix = np.eye(c)+np.tril(np.array(entries).reshape(c,c),-1)
    b = np.linalg.qr(matrix)[0][:,:r].astype(np.float32)
    fraction = draw(st.sampled_from([0.,.05,.2,.5,1.]))
    return np.array(flat).reshape(m,c),np.array(weights).reshape(m,c),b,fraction


@given(projection_cases())
@settings(max_examples=45,deadline=None)
def test_weighted_hybrid_cannot_worsen_rank_surrogate_and_preserves_inputs(case):
    target,weights,basis,fraction = case
    frozen = [a.copy() for a in (target,weights,basis)]
    rank = weighted_projection(target,weights,basis)
    hybrid = weighted_projection(target,weights,basis,fraction)
    # FP32 stored factors are judged, not ideal unrounded factors.
    assert hybrid.weighted_error <= rank.weighted_error+1e-8*(1+np.sum(weights*target**2))
    assert len(hybrid.values) <= int(np.floor(fraction*target.size))
    keys = hybrid.rows.astype(int)*target.shape[1]+hybrid.columns
    assert len(np.unique(keys)) == len(keys) and np.isfinite(hybrid.decoded(basis)).all()
    for actual,expected in zip((target,weights,basis),frozen):
        np.testing.assert_array_equal(actual,expected)
    assert np.isclose(hybrid.weighted_error,np.sum(weights*(target-hybrid.decoded(basis))**2))


def test_identity_rank_recovers_newton_target_and_all_residual_recovers_arbitrary_targets():
    target = np.array([[.3,-2.,1.],[.8,2.,-.1]],np.float32)
    weights = np.array([[1.,4.,2.],[2.,.1,8.]])
    identity = weighted_projection(target,weights,np.eye(3,dtype=np.float32))
    np.testing.assert_array_equal(identity.decoded(np.eye(3)),target.astype(float))
    basis = np.eye(3,dtype=np.float32)[:,:1]
    hybrid = weighted_projection(target,weights,basis,1.)
    np.testing.assert_allclose(hybrid.decoded(basis),target,rtol=0,atol=1e-7)


def test_weighted_solution_is_independent_of_factor_coordinate_rotation():
    b = np.linalg.qr(np.array([[1.,2.],[3.,-1.],[2.,1.]]))[0].astype(np.float32)
    rotation = np.array([[0.,-1.],[1.,0.]],np.float32)
    target = np.array([[.3,-2.,1.],[.8,2.,-.1]])
    weights = np.array([[1.,4.,2.],[2.,.1,8.]])
    a = weighted_projection(target,weights,b)
    rotated = weighted_projection(target,weights,b@rotation)
    np.testing.assert_allclose(a.decoded(b),rotated.decoded(b@rotation),atol=1e-6,rtol=0)


@pytest.mark.parametrize('bad', ['weights','basis','fraction','nonfinite'])
def test_weighted_projection_rejects_invalid_domain(bad):
    v,d,b,f = np.ones((2,3)),np.ones((2,3)),np.eye(3)[:,:2],.05
    if bad == 'weights':
        d[0,0] = 0
    elif bad == 'basis':
        b[0,0] = 3
    elif bad == 'fraction':
        f = -1
    else:
        v[0,0] = np.nan
    with pytest.raises(ValueError):
        weighted_projection(v,d,b,f)


def test_compact_remapping_stitch_and_standalone_roundtrip(tmp_path):
    from experiments.uniform_threshold_core import ThresholdModel
    nodes = np.array([[1,.5,-2,1],[-2,0,-4,-5]],np.float32)
    values = np.arange(18,dtype=np.float32).reshape(6,3)/10
    part,lookup,ids = compact_topology(nodes,values)
    assert ids.tolist() == [1,3,4] and lookup.tolist() == [-1,0,-1,1,2,-1]
    basis = np.linalg.qr(np.array([[1.,2.],[3.,-1.],[2.,1.]]))[0].astype(np.float32)
    compact = project_part(part,np.ones_like(part.coefficients),basis,.4)
    model = stitch([compact,compact],basis,np.array([.2,.3,-.1],np.float32),2,'multilabel')
    x = np.array([[0,np.nan],[1,1],[np.nan,-1]],np.float32)
    decoded = compact.decoded(basis)
    # Explicit routing of both small trees, independent of production traversal.
    raw = 2*decoded[[0,2,1]]+model.base.bias
    from experiments.shared_leaf_basis import probabilities
    np.testing.assert_allclose(model.predict(x),probabilities(raw,'multilabel'),rtol=0,atol=1e-14)
    path = tmp_path/'compact.npz'
    model.save(path)
    loaded = ThresholdModel.load(path)
    np.testing.assert_array_equal(model.predict(x),loaded.predict(x))
    arrays = (loaded.base.nodes,loaded.base.node_offsets,loaded.base.leaf_offsets,
        loaded.base.coefficients,loaded.base.basis,loaded.base.bias,
        loaded.rows,loaded.columns,loaded.values,loaded.decision_shifts)
    assert loaded.payload_bytes == sum(a.nbytes for a in arrays)


@pytest.mark.parametrize('task', ['multiclass','multilabel'])
def test_online_gpu_neutral_identity_matches_native_and_compact_replay(task,tmp_path):
    cp = pytest.importorskip('cupy')
    from experiments.a100_runner import build_model
    from experiments.online_leaf_boosting import CompactLeafBoosting
    from experiments.uniform_threshold_core import ThresholdModel
    rng = np.random.default_rng(417)
    x = rng.normal(size=(320,5)).astype(np.float32)
    x[::17,0] = np.nan
    y = np.argmax(x[:,1:5],axis=1).astype(np.int32) if task == 'multiclass' else (x[:,1:5]>0).astype(np.float32)
    config = dict(ntrees=6,lr=.1,max_depth=3,lambda_l2=1.)
    native = build_model('full',task,4,config,318,3,1.)
    native.fit(x[:240],y[:240],eval_sets=[dict(X=x[240:],y=y[240:])])
    reference = native.predict(x[240:])
    for mode,identity in [('full',False),('rank',True),('hybrid',False)]:
        model = CompactLeafBoosting(**build_model('full',task,4,config,318,3,1.).params,
            compact_mode=mode,task=task,warmup=2,force_identity=identity)
        model.fit(x[:240],y[:240],eval_sets=[dict(X=x[240:],y=y[240:])])
        exported = model.export()
        prediction = exported.predict(x[240:])
        from experiments.shared_leaf_basis import probabilities
        np.testing.assert_allclose(prediction,probabilities(model.validation_final_raw[0],task),atol=2e-5,rtol=0)
        if mode != 'hybrid':
            np.testing.assert_allclose(prediction,reference,atol=2e-5,rtol=0)
        path = tmp_path/f'{task}-{mode}.npz'
        exported.save(path)
        np.testing.assert_array_equal(ThresholdModel.load(path).predict(x[240:]),prediction)
        if mode == 'hybrid':
            assert model.basis is not None and model.full_weights == []
            assert model.warm_memberships == [] and model.warm_basis_full is None
            assert all(p.coefficients.shape[1] == model.basis.shape[1] for p in model.parts)
        assert model.training_pool_peak > 0 and len(model.parts) == 6


def test_sealed_evaluation_and_summary_keep_validation_guard(tmp_path):
    from experiments.online_leaf_study import evaluate_case,summarize
    from experiments.a100_runner import sha256
    from experiments.output_calibration_core import calibrated_quality
    from test_prepared_leaf_inference import model_for
    from experiments.online_leaf_study import raw
    model = model_for('multilabel','full')
    x = np.array([[0,np.nan],[1,1],[np.nan,0],[.5,1]],np.float32)
    y = model.predict_labels(x)
    directory = tmp_path/'toy_1'
    directory.mkdir()
    np.savez(directory/'splits.npz',test=np.array([0,1,2,3]))
    path = directory/'full.npz'
    model.save(path)
    q = calibrated_quality(y,raw(model,x),'multilabel',model.decision_shifts)
    choices = [dict(method=method,artifact='toy_1/full.npz',sha256=sha256(path),
        parameters=model.parameter_count,payload_bytes=model.payload_bytes,
        validation=q if method == 'online_full' else {**q,'loss':2*q['loss']})
        for method in ('online_full','posthoc_rank')]
    case = dict(dataset='toy',seed=1,task='multilabel',choices=choices)
    rows = evaluate_case(case,tmp_path,x,y)
    assert len(rows) == 2 and rows[1]['quality_pass'] and not rows[1]['validation_quality_pass']
    summary = next(s for s in summarize(rows) if s['method'] == 'posthoc_rank')
    assert summary['joint_75_successes'] == 0 and summary['quality_passes'] == 1


def test_case_pipeline_with_fake_trainer_seals_and_reloads_all_models(tmp_path,monkeypatch):
    """Exercise orchestration/serialization before spending real-data GPU fits.

    The fake trainer has no optimization semantics; independent numerical/GPU
    tests above cover those. This test kills late plumbing/key/schema errors.
    """
    import sys
    from types import SimpleNamespace
    from test_prepared_leaf_inference import model_for
    from experiments.shared_leaf_basis import functional_basis
    import experiments.online_leaf_study as study
    source_model = model_for('multilabel','full')
    x = np.tile(np.array([[0,np.nan],[1,1],[np.nan,0],[.5,1]],np.float32),(3,1))
    y = source_model.predict_labels(x)
    source_indices = np.arange(len(x))
    source_dir,parent_dir,out = tmp_path/'source',tmp_path/'parent',tmp_path/'result'
    for root in (source_dir,parent_dir):
        (root/'toy_1').mkdir(parents=True)
    train,validation,test = np.arange(8),np.arange(8,11),np.array([11])
    np.savez(source_dir/'toy_1/splits.npz',train=train,validation=validation,test=test,source_indices=source_indices)
    np.savez(parent_dir/'toy_1/development_splits.npz',train=train,calibration=np.array([8,9]),selection=np.array([10]))
    config = dict(source_dir=str(source_dir),inference_parent_dir=str(parent_dir),ntrees=2,
        online_warmup=1,online_rank_fraction=.5,online_residual_fraction=.05,max_depth=3,lambda_l2=1.,
        decision_grid=[0,-.5,.5],decision_shrinkage=[.25,.5,1.])
    parts = []
    b0 = source_model.base
    for i in range(2):
        part,_,_ = compact_topology(b0.nodes[b0.node_offsets[i]:b0.node_offsets[i+1]],
                                    b0.coefficients[b0.leaf_offsets[i]:b0.leaf_offsets[i+1]])
        parts.append(part)
    prefix = stitch(parts[:1],None,b0.bias,2,'multilabel')
    contribution = study.raw(prefix,x[train])-b0.bias
    basis = functional_basis(contribution,'multilabel')[1].astype(np.float32)
    snapshot = dict(parts=parts,weights=[np.ones_like(p.coefficients) for p in parts],
                    bias=b0.bias.copy(),basis=basis,warm_mean=contribution.mean(axis=0),task='multilabel')
    class FakeTrainer:
        def __init__(self,compact_mode,**kwargs):
            self.mode = compact_mode
            self.warm_fingerprint = 'same-prefix'
            self.training_pool_peak = 0
            self.stage_ms,self.trace,self.history = {},[],[]
        def fit(self,features,target,eval_sets):
            if self.mode == 'full':
                self.exported = stitch(parts,None,b0.bias,2,'multilabel')
                self.parts,self.full_weights,self.warm_basis_full = parts,snapshot['weights'],basis
            else:
                self.exported = study.posthoc(snapshot,features,self.mode,config)
                self.parts = parts  # only length is observed in this isolated I/O test
                self.full_weights,self.warm_basis_full = [],None
            self.basis = None if self.mode == 'full' else self.exported.base.basis
            self.study_bias,self.warm_train_mean = self.exported.base.bias,snapshot['warm_mean']
            self.warm_memberships = []
            self.validation_final_raw = [study.raw(self.exported,e['X']) for e in eval_sets]
        def export(self):
            return self.exported
    monkeypatch.setitem(sys.modules,'experiments.online_leaf_boosting',SimpleNamespace(CompactLeafBoosting=FakeTrainer))
    fake_cp = SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None)),
        get_default_memory_pool=lambda:SimpleNamespace(free_all_blocks=lambda:None))
    monkeypatch.setitem(sys.modules,'cupy',fake_cp)
    monkeypatch.setattr(study,'build_model',lambda *a,**kw:SimpleNamespace(params={}))
    result = study.prepare_case('toy',1,config,out,x,y,source_indices,dict(task='multilabel',sha256='synthetic'))
    assert len(result['fits']) == 4 and len(result['choices']) == 7
    assert {r['method'] for r in result['choices']} == {'online_full','online_rank','online_hybrid',
        'online_more_rank','posthoc_rank','posthoc_hybrid','posthoc_more_rank'}
    rows = study.evaluate_case(result,out,x,y)
    assert len(rows) == 7 and all(np.isfinite(r['loss_ratio']) for r in rows)
