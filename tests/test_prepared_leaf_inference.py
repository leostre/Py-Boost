from dataclasses import replace
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.prepared_leaf_inference import PreparedLeafInference
from experiments.shared_leaf_basis import LeafEnsemble, reference_class_model
from experiments.uniform_threshold_core import ThresholdModel


def model_for(task, role, seed=0):
    rng = np.random.default_rng(seed)
    # Different depths, NaN-left/NaN-right, duplicate destination leaf and offsets.
    dense = LeafEnsemble(np.array([[1, .5, -1, 1], [-2, 0, -2, -3],
                                  [-1, 0, -1, -1]], np.float32),
        np.array([0, 2, 3], np.int64), np.array([0, 3, 4], np.int64),
        rng.normal(size=(4, 4)).astype(np.float32), np.empty((0, 0), np.float32),
        rng.normal(size=4).astype(np.float32), 2, task)
    base = dense
    if role == 'reference':
        base = reference_class_model(dense)
    elif role in ('rank', 'hybrid'):
        basis = np.linalg.qr(rng.normal(size=(4, 2)))[0].astype(np.float32)
        base = replace(dense, coefficients=rng.normal(size=(4, 2)).astype(np.float32),
                       basis=basis, representation='low_rank')
    rows = np.array([3, 0, 2, 0], np.int32) if role == 'hybrid' else np.array([], np.int32)
    cols = np.array([1, 3, 0, 1], np.int32) if role == 'hybrid' else np.array([], np.int32)
    values = np.array([.3, -.4, .25, -.8], np.float32) if role == 'hybrid' else np.array([], np.float32)
    return ThresholdModel(base, rows, cols, values, np.array([.2, -.3, .1, 0], np.float32))


def prepare_cpu(model, implementation):
    if implementation == 'prepared':
        return model.prepare_inference()
    if implementation == 'compiled':
        return model.prepare_compiled_inference()
    from experiments.accumulated_leaf_inference import AccumulatedLeafInference
    return AccumulatedLeafInference.prepare(model)


@pytest.mark.parametrize('task,role', [(t, r) for t in ('multiclass', 'multilabel')
    for r in ('full', 'rank', 'hybrid')] + [('multiclass', 'reference')])
@pytest.mark.parametrize('implementation', ['prepared', 'compiled', 'accumulated'])
@given(st.lists(st.tuples(st.sampled_from([-.1, 0., .5, .500001, 1., float('nan')]),
                         st.sampled_from([-1., 0., 1., float('nan')])), min_size=1, max_size=40),
       st.integers(0, 1000))
@settings(max_examples=25, deadline=None)
def test_prepared_matches_independent_csr_and_batch_partition(task, role, implementation, rows, seed):
    model = model_for(task, role, seed)
    prepared = prepare_cpu(model, implementation)
    x = np.array(rows, np.float32)
    np.testing.assert_allclose(prepared.predict(x), model.predict(x), rtol=0, atol=2e-14)
    np.testing.assert_array_equal(prepared.predict_labels(x), model.predict_labels(x))
    # Object order/repetition must be preserved, including repeated residual destinations.
    np.testing.assert_allclose(prepared.predict(x[::-1]), prepared.predict(x)[::-1], rtol=0, atol=2e-14)
    np.testing.assert_allclose(np.concatenate([prepared.predict(a[None]) for a in x]),
                               prepared.predict(x), rtol=0, atol=2e-14)


@pytest.mark.parametrize('implementation', ['prepared', 'compiled', 'accumulated'])
def test_prepared_owns_immutable_snapshot_and_no_sparse_request(monkeypatch, implementation):
    from scipy import sparse
    import experiments.uniform_threshold_core as reference
    model = model_for('multilabel', 'hybrid')
    runtime = prepare_cpu(model, implementation)
    prepared = runtime if implementation == 'prepared' else runtime.snapshot
    x = np.array([[0, np.nan], [1, 1]], np.float32)
    expected = model.predict(x)
    for a in (model.base.nodes, model.base.coefficients, model.base.basis, model.base.bias,
              model.rows, model.columns, model.values, model.decision_shifts):
        a.fill(0)
    def forbidden(*args, **kwargs):
        raise AssertionError('request rebuilt a visit matrix, CSR, or full leaf history')
    monkeypatch.setattr(reference, 'leaf_design', forbidden)
    monkeypatch.setattr(LeafEnsemble, 'leaf_indices', forbidden)
    monkeypatch.setattr(sparse, 'coo_matrix', forbidden)
    monkeypatch.setattr(sparse, 'csr_matrix', forbidden)
    if implementation in ('compiled', 'accumulated'):
        monkeypatch.setattr(np, 'repeat', forbidden)
    np.testing.assert_allclose(runtime.predict(x), expected, rtol=0, atol=1e-14)
    arrays = (prepared.base.nodes, prepared.base.coefficients, prepared.base.basis,
              prepared.base.bias, prepared.indptr, prepared.columns, prepared.values,
              prepared.decision_shifts, prepared.base.node_offsets, prepared.base.leaf_offsets)
    if implementation == 'accumulated':
        arrays += (runtime.decoder,)
    assert runtime.resident_bytes == sum(a.nbytes for a in arrays)
    for a in arrays:
        with pytest.raises(ValueError):
            a.setflags(write=True)


@pytest.mark.parametrize('bad', [np.array([[np.inf, 0]], np.float32), np.zeros((2, 3)), np.zeros(2)])
@pytest.mark.parametrize('implementation', ['prepared', 'compiled', 'accumulated'])
def test_prepared_rejects_invalid_features(bad, implementation):
    model = model_for('multilabel', 'hybrid')
    prepared = prepare_cpu(model, implementation)
    with pytest.raises(ValueError):
        prepared.predict(bad)


@pytest.mark.parametrize('task', ['multiclass', 'multilabel'])
def test_compiled_extreme_values_raw_and_probability_contract(task):
    from experiments.compiled_leaf_inference import gather_into, finish_into
    model = model_for(task, 'hybrid')
    model.base.coefficients[...] *= np.float32(1000)
    runtime = model.prepare_compiled_inference()
    x = np.array([[0, np.nan], [1, 1], [np.nan, 0]], np.float32)
    from experiments.sparse_leaf_core import leaf_design
    expected = model.raw_from_design(leaf_design(model.base.leaf_indices(x), len(model.base.coefficients)))
    np.testing.assert_allclose(runtime.raw(x), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(runtime.predict(x), model.predict(x), rtol=0, atol=2e-14)
    assert gather_into.nopython_signatures and finish_into.nopython_signatures
    assert gather_into.targetoptions['fastmath'] is False
    assert runtime.predict(np.empty((0, 2), np.float32)).shape == (0, 4)


def test_idle_gate_rejects_idle_but_occupied_and_non_a100():
    from experiments.inference_gpu_guard import GpuSample, idle_reason
    clean = GpuSample(0, 'A100-PCIE-40GB', 10, 0, ())
    assert idle_reason(clean) is None
    assert idle_reason(replace(clean, pids=(123,), memory_mib=1800)) is not None
    assert idle_reason(replace(clean, utilization=30)) is not None
    assert idle_reason(replace(clean, memory_mib=300)) is not None
    assert idle_reason(replace(clean, name='RTX 4090')) is not None


def test_shared_capacity_policy_accepts_foreign_context_not_low_memory():
    from experiments.inference_gpu_guard import GpuSample, capacity_reason
    occupied = GpuSample(0, 'A100-PCIE-40GB', 1877, 95, (123,), total_mib=40960)
    assert capacity_reason(occupied, 2048) is None
    assert capacity_reason(replace(occupied, memory_mib=40000), 2048) is not None
    assert capacity_reason(replace(occupied, name='RTX 4090'), 2048) is not None


def test_compiled_rejects_source_indices_mutated_before_preparation():
    model = model_for('multilabel', 'hybrid')
    model.columns[0] = 999
    with pytest.raises(ValueError):
        model.prepare_compiled_inference()


def test_compiled_summaries_keep_new_path_and_fair_full_control():
    from experiments.prepared_inference_study import summarize
    rows = [dict(dataset='yeast', backend='cpu', batch_scope='all', mode='host',
                 seed=s, role=r, path=p, batch=478+s,
                 samples_ms=[1, 2, 3] if p == 'compiled' else [2, 4, 6],
                 median_ms=2 if p == 'compiled' else 4)
            for s in range(5) for r in ('full', 'rank', 'hybrid')
            for p in ('reference', 'prepared', 'compiled')]
    result = summarize(rows)
    assert len(result) == 18
    assert all(r['pairs'] == 5 for r in result)
    for r in result:
        if r['comparison'] in ('compiled_vs_reference', 'compiled_vs_prepared'):
            assert r['median_time_ratio'] == .5
        if r['comparison'] == 'compiled_vs_compiled_full':
            assert r['median_time_ratio'] == 1


def test_summary_pairs_all_scope_even_for_different_test_sizes():
    from experiments.prepared_inference_study import summarize
    rows = [dict(dataset='yeast', backend='cpu', batch_scope='all', mode='host',
                 seed=s, role=r, path=p, batch=478+s, samples_ms=[1, 2, 3], median_ms=2)
            for s in range(5) for r in ('full', 'rank', 'hybrid') for p in ('reference', 'prepared')]
    result = summarize(rows)
    assert len(result) == 9
    assert all(r['pairs'] == 5 and r['median_time_ratio'] == 1 for r in result)


def test_host_wrapper_converts_to_device_before_contiguity(monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments.prepared_leaf_gpu import PreparedLeafGpu
    host, device = np.ones((2, 2), np.float64), object()
    expected = np.array([[.25, .75], [.5, .5]])
    calls = []
    def asarray(value, dtype):
        assert value is host and dtype == np.float32
        calls.append('transfer')
        return device
    def contiguous(value):
        assert value is device
        calls.append('contiguous')
        return device
    def predict_device(value):
        assert value is device
        return SimpleNamespace(get=lambda: expected)
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(float32=np.float32,
        asarray=asarray, ascontiguousarray=contiguous))
    runtime = object.__new__(PreparedLeafGpu)
    runtime.predict_device = predict_device
    np.testing.assert_array_equal(runtime.predict(host), expected)
    assert calls == ['transfer', 'contiguous']


def test_fused_source_is_header_free():
    from experiments.fused_leaf_gpu import SOURCE
    assert '#include' not in SOURCE
    assert '__longlong_as_double(0x7ff0000000000000LL)' in SOURCE


@pytest.mark.parametrize('task', ['multiclass', 'multilabel'])
def test_fused_gpu_extreme_values_and_padded_lanes(task):
    cp = pytest.importorskip('cupy')
    from experiments.fused_leaf_gpu import FusedLeafGpu
    from experiments.sparse_leaf_core import leaf_design
    model = model_for(task, 'hybrid')
    model.base.coefficients[...] *= np.float32(1000)
    runtime = FusedLeafGpu(model)
    x = np.array([[0, np.nan], [1, 1], [np.nan, 0]], np.float32)
    expected_raw = model.raw_from_design(leaf_design(model.base.leaf_indices(x),
                                                    len(model.base.coefficients)))
    assert runtime.outputs < runtime.threads
    np.testing.assert_allclose(runtime.raw_device(cp.asarray(x)).get(), expected_raw,
                               rtol=0, atol=1e-12)
    actual = runtime.predict(x)
    assert actual.dtype == np.float64 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, model.predict(x), rtol=0, atol=2e-14)


@pytest.mark.parametrize('task,role', [(t, r) for t in ('multiclass', 'multilabel')
    for r in ('full', 'rank', 'hybrid')] + [('multiclass', 'reference')])
@pytest.mark.parametrize('implementation', ['prepared', 'fused', 'accumulated', 'unified'])
def test_prepared_gpu_matches_csr_without_design(task, role, implementation, monkeypatch):
    cp = pytest.importorskip('cupy')
    from experiments.prepared_leaf_gpu import PreparedLeafGpu
    from experiments.fused_leaf_gpu import FusedLeafGpu
    from experiments.accumulated_leaf_gpu import AccumulatedLeafGpu
    from experiments.unified_leaf_gpu import UnifiedLeafGpu
    from cupyx.scipy import sparse
    model = model_for(task, role)
    x = np.array([[0, np.nan], [1, 1], [np.nan, 0], [.5, -.1]], np.float32)
    expected = model.predict(x)
    gpu = {'prepared': PreparedLeafGpu, 'fused': FusedLeafGpu, 'accumulated': AccumulatedLeafGpu,
           'unified': UnifiedLeafGpu}[implementation](model)
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU request built sparse data')
    monkeypatch.setattr(sparse, 'coo_matrix', forbidden)
    monkeypatch.setattr(sparse, 'csr_matrix', forbidden)
    for features in (x, np.asfortranarray(x), cp.asarray(x)):
        np.testing.assert_allclose(gpu.predict(features), expected, rtol=0, atol=2e-14)


@pytest.mark.parametrize('role', ['full', 'rank', 'hybrid', 'reference'])
def test_accumulated_cpu_owns_single_output_and_inplace_blas(monkeypatch, role):
    import experiments.accumulated_leaf_inference as target
    model = model_for('multiclass', role)
    runtime = target.AccumulatedLeafInference.prepare(model)
    x = np.tile(np.array([[0, np.nan], [1, 1]], np.float32), (9, 1))
    expected = model.predict(x)
    zeros, calls = [], []
    original_zeros, original_gemm = np.zeros, target.dgemm
    def tracked_zeros(shape, *args, **kwargs):
        result = original_zeros(shape, *args, **kwargs)
        zeros.append(result)
        return result
    def tracked_gemm(alpha, a, b, **kwargs):
        assert a.flags.f_contiguous and b.flags.f_contiguous
        assert kwargs['c'].flags.f_contiguous and kwargs['overwrite_c'] == 1
        assert kwargs['beta'] == (1. if role == 'hybrid' else 0.)
        result = original_gemm(alpha, a, b, **kwargs)
        assert np.shares_memory(result, kwargs['c'])
        calls.append(kwargs['c'])
        return result
    monkeypatch.setattr(np, 'zeros', tracked_zeros)
    monkeypatch.setattr(target, 'dgemm', tracked_gemm)
    actual = runtime.predict(x)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    assert actual is zeros[0]
    assert len(zeros) == (2 if role in ('rank', 'hybrid') else 1)
    assert len(calls) == (1 if role in ('rank', 'hybrid') else 0)
    if calls:
        assert np.shares_memory(actual, calls[0])
    assert runtime.predict(np.empty((0, 2), np.float32)).shape == (0, 4)


@pytest.mark.parametrize('task', ['multiclass', 'multilabel'])
def test_accumulated_cpu_extreme_values_and_invalid_source(task):
    from experiments.accumulated_leaf_inference import AccumulatedLeafInference
    model = model_for(task, 'hybrid')
    model.base.coefficients[...] *= np.float32(1000)
    runtime = AccumulatedLeafInference.prepare(model)
    x = np.array([[0, np.nan], [1, 1], [np.nan, 0]], np.float32)
    from experiments.sparse_leaf_core import leaf_design
    expected = model.raw_from_design(leaf_design(model.base.leaf_indices(x), len(model.base.coefficients)))
    np.testing.assert_allclose(runtime.raw(x), expected, rtol=0, atol=2e-12)
    np.testing.assert_allclose(runtime.predict(x), model.predict(x), rtol=0, atol=2e-14)
    model.columns[0] = 999
    with pytest.raises(ValueError):
        AccumulatedLeafInference.prepare(model)


@pytest.mark.parametrize('backend,previous', [('cpu', 'compiled'), ('gpu', 'fused')])
def test_accumulated_summaries_compare_old_and_equally_optimized_full(backend, previous):
    from experiments.prepared_inference_study import summarize
    rows = [dict(dataset='yeast', backend=backend, batch_scope='all', mode='host',
                 seed=s, role=r, path=p, batch=478+s,
                 samples_ms=[1, 2, 3] if p == 'accumulated' else [2, 4, 6],
                 median_ms=2 if p == 'accumulated' else 4)
            for s in range(5) for r in ('full', 'rank', 'hybrid') for p in (previous, 'accumulated')]
    result = summarize(rows)
    assert len(result) == 9
    for row in result:
        assert row['pairs'] == 5
        assert row['median_time_ratio'] == (.5 if row['comparison'] == f'accumulated_vs_{previous}' else 1.)


@given(st.integers(2, 12), st.integers(1, 12), st.integers(0, 1000))
@settings(max_examples=30, deadline=None)
def test_accumulated_cpu_varied_rank_and_output_width(outputs, proposed_rank, seed):
    from experiments.accumulated_leaf_inference import AccumulatedLeafInference
    rng = np.random.default_rng(seed)
    rank = min(proposed_rank, outputs)
    source = model_for('multilabel', 'full', seed)
    base = replace(source.base, coefficients=rng.normal(size=(4, rank)).astype(np.float32),
        basis=np.linalg.qr(rng.normal(size=(outputs, rank)))[0].astype(np.float32),
        bias=rng.normal(size=outputs).astype(np.float32), representation='low_rank')
    model = ThresholdModel(base, np.array([0, 3], np.int32), np.array([outputs-1, 0], np.int32),
        np.array([.3, -.7], np.float32), np.zeros(outputs, np.float32))
    runtime = AccumulatedLeafInference.prepare(model)
    x = np.array([[0, np.nan], [1, 1], [.5, 0]], np.float32)
    np.testing.assert_allclose(runtime.predict(x), model.predict(x), rtol=0, atol=2e-14)
    np.testing.assert_allclose(np.concatenate([runtime.predict(a[None]) for a in x]),
                               runtime.predict(x), rtol=0, atol=2e-14)


@pytest.mark.parametrize('task', ['multiclass', 'multilabel'])
@pytest.mark.parametrize('role', ['full', 'rank', 'hybrid', 'reference'])
def test_accumulated_gpu_single_buffer_empty_extreme_and_ownership(task, role, monkeypatch):
    if task == 'multilabel' and role == 'reference':
        pytest.skip('reference-class representation is multiclass only')
    cp = pytest.importorskip('cupy')
    import cupy.cublas as blas
    from experiments.accumulated_leaf_gpu import AccumulatedLeafGpu
    model = model_for(task, role)
    model.base.coefficients[...] *= np.float32(1000)
    runtime = AccumulatedLeafGpu(model)
    x = np.tile(np.array([[0, np.nan], [1, 1]], np.float32), (9, 1))
    expected = model.predict(x)
    # Prepare before observing request allocations; no full decoded product or
    # separate probability array may be allocated by the runtime/GEMM wrapper.
    device = cp.asarray(x)
    runtime.predict_device(device)
    allocations, gemm_calls = [], []
    original_empty, original_gemm = cp.empty, blas.gemm
    def tracked_empty(shape, *args, **kwargs):
        result = original_empty(shape, *args, **kwargs)
        allocations.append(result)
        return result
    def tracked_gemm(*args, **kwargs):
        result = original_gemm(*args, **kwargs)
        assert result.data.ptr == kwargs['out'].data.ptr
        assert kwargs['beta'] == (1. if role == 'hybrid' else 0.)
        gemm_calls.append(result)
        return result
    monkeypatch.setattr(cp, 'empty', tracked_empty)
    monkeypatch.setattr(blas, 'gemm', tracked_gemm)
    actual = runtime.predict_device(device)
    np.testing.assert_allclose(actual.get(), expected, rtol=0, atol=2e-14)
    assert actual is allocations[0]
    assert len(allocations) == (2 if role in ('rank', 'hybrid') else 1)
    assert len(gemm_calls) == (1 if role in ('rank', 'hybrid') else 0)
    frozen = actual.get()
    runtime.predict_device(device[::-1].copy())
    np.testing.assert_array_equal(actual.get(), frozen)
    empty = runtime.predict_device(cp.asarray(np.empty((0, 2), np.float32)))
    assert empty.shape == (0, 4)
    for bad in (cp.ones((2, 3), cp.float32), cp.full((2, 2), cp.inf, cp.float32)):
        with pytest.raises(ValueError):
            runtime.predict_device(bad)


@pytest.mark.parametrize('task,role', [(t, r) for t in ('multiclass', 'multilabel')
    for r in ('full', 'rank', 'hybrid')] + [('multiclass', 'reference')])
def test_unified_gpu_single_global_output_raw_extremes_and_owned_snapshot(task, role, monkeypatch):
    cp = pytest.importorskip('cupy')
    from experiments.unified_leaf_gpu import UnifiedLeafGpu
    from experiments.sparse_leaf_core import leaf_design
    import cupy.cublas as blas
    model = model_for(task, role)
    model.base.coefficients[...] *= np.float32(1000)
    runtime = UnifiedLeafGpu(model)
    x = np.array([[0, np.nan], [1, 1], [np.nan, 0]], np.float32)
    expected = model.predict(x)
    expected_raw = model.raw_from_design(leaf_design(model.base.leaf_indices(x), len(model.base.coefficients)))
    device = cp.asarray(x)
    runtime.predict_device(device)
    allocations = []
    original_empty = cp.empty
    def tracked_empty(shape, *args, **kwargs):
        out = original_empty(shape, *args, **kwargs)
        allocations.append(out)
        return out
    def forbidden(*args, **kwargs):
        raise AssertionError('unified request called GEMM')
    monkeypatch.setattr(cp, 'empty', tracked_empty)
    monkeypatch.setattr(blas, 'gemm', forbidden)
    # Source mutation after preparation must not affect owned device snapshot.
    model.base.coefficients.fill(0)
    model.base.basis.fill(0)
    model.values.fill(0)
    out = runtime.predict_device(device)
    np.testing.assert_allclose(out.get(), expected, rtol=0, atol=2e-14)
    assert len(allocations) == 1 and out is allocations[0]
    frozen = out.get()
    np.testing.assert_allclose(runtime.raw_device(device).get(), expected_raw, rtol=0, atol=2e-12)
    runtime.predict_device(device[::-1].copy())
    np.testing.assert_array_equal(out.get(), frozen)
    assert runtime.predict_device(cp.empty((0, 2), cp.float32)).shape == (0, 4)
    for bad in (cp.ones((2, 3), cp.float32), cp.full((2, 2), cp.inf, cp.float32),
                cp.ones((3, 2), cp.float64), device[::-1]):
        with pytest.raises(ValueError):
            runtime.predict_device(bad)
    names = ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias',
             'indptr', 'columns', 'values', 'decision_shifts')
    assert runtime.resident_bytes == sum(getattr(runtime, '_'+s).nbytes for s in names)
    assert 'num_regs' in runtime.kernel_resources and 'local_size_bytes' in runtime.kernel_resources


@pytest.mark.parametrize('task', ['multilabel', 'multiclass'])
@given(outputs=st.sampled_from([2, 14, 31, 32, 33, 100, 101, 129]),
       proposed_rank=st.integers(1, 129), seed=st.integers(0, 1000))
@settings(max_examples=12, deadline=None)
def test_unified_gpu_varied_rank_padding_and_batch_partition(task, outputs, proposed_rank, seed):
    cp = pytest.importorskip('cupy')
    from experiments.unified_leaf_gpu import UnifiedLeafGpu
    rng = np.random.default_rng(seed)
    rank = min(outputs, proposed_rank)
    source = model_for(task, 'full', seed)
    base = replace(source.base, coefficients=rng.normal(size=(4, rank)).astype(np.float32),
        basis=np.linalg.qr(rng.normal(size=(outputs, rank)))[0].astype(np.float32),
        bias=rng.normal(size=outputs).astype(np.float32), representation='low_rank')
    model = ThresholdModel(base, np.array([0, 3], np.int32), np.array([outputs-1, 0], np.int32),
        np.array([.3, -.7], np.float32), np.zeros(outputs, np.float32))
    runtime = UnifiedLeafGpu(model)
    x = np.array([[0, np.nan], [1, 1], [.5, 0]], np.float32)
    expected = model.predict(x)
    np.testing.assert_allclose(runtime.predict(x), expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(runtime.predict(x[::-1]), expected[::-1], rtol=0, atol=2e-14)
    np.testing.assert_allclose(np.concatenate([runtime.predict(a[None]) for a in x]), expected,
                               rtol=0, atol=2e-14)


def test_unified_summary_uses_accumulated_and_equally_unified_full():
    from experiments.prepared_inference_study import summarize
    rows = [dict(dataset='yeast', backend='gpu', batch_scope='all', mode='device',
                 seed=s, role=r, path=p, samples_ms=[1, 2, 3] if p == 'unified' else [2, 4, 6],
                 median_ms=2 if p == 'unified' else 4)
            for s in range(5) for r in ('full', 'rank', 'hybrid') for p in ('accumulated', 'unified')]
    result = summarize(rows)
    comparisons = {r['comparison'] for r in result}
    assert comparisons == {'accumulated_vs_accumulated_full', 'unified_vs_accumulated', 'unified_vs_unified_full'}
    for r in result:
        assert r['median_time_ratio'] == (.5 if r['comparison'] == 'unified_vs_accumulated' else 1.)
