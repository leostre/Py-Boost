"""Independent numeric oracles for routing, PCA, gauge, storage and selection."""
from dataclasses import replace
import sys
from types import ModuleType
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.shared_leaf_basis import (
    LeafEnsemble, compress, functional_basis, probabilities, reference_class_model,
    select_rank,
)
from experiments.shared_leaf_gpu import device_features


def host_features(layout):
    x = np.array([[1., np.nan], [-3., .25], [8., -7.]], dtype=np.float64)
    if layout == 'F':
        return np.asfortranarray(x)
    if layout == 'slice':
        return np.repeat(x, 2, axis=1)[:, ::2]
    return x


@pytest.mark.parametrize('layout', ['C', 'F', 'slice'])
def test_host_transfer_precedes_cupy_only_contiguity(monkeypatch, layout):
    # A deliberately strict boundary double, not a replacement for GPU parity.
    # The previous implementation fails here because it passes a NumPy array
    # directly to a function that only accepts device arrays.
    class DeviceArray(np.ndarray):
        pass

    fake = ModuleType('cupy')
    calls = []
    def asarray(value, dtype):
        calls.append('transfer')
        return np.asarray(value, dtype=dtype).view(DeviceArray)
    def ascontiguousarray(value, dtype=None):
        if not isinstance(value, DeviceArray):
            raise TypeError('device array required')
        calls.append('layout')
        return np.ascontiguousarray(value, dtype=dtype).view(DeviceArray)
    fake.asarray, fake.ascontiguousarray, fake.float32 = asarray, ascontiguousarray, np.float32
    monkeypatch.setitem(sys.modules, 'cupy', fake)
    source = host_features(layout)
    original = source.copy()
    result = device_features(source, 2)
    assert calls == ['transfer', 'layout']
    assert isinstance(result, DeviceArray)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    np.testing.assert_array_equal(result, original.astype(np.float32))
    np.testing.assert_array_equal(source, original)
    with pytest.raises(ValueError, match='feature shape'):
        device_features(source, 3)


@pytest.mark.parametrize('layout', ['C', 'F', 'slice'])
def test_actual_cupy_host_and_device_inputs(layout):
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CUDA device unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    source = host_features(layout)
    result = device_features(source, 2)
    assert isinstance(result, cp.ndarray)
    assert result.dtype == cp.float32 and result.flags.c_contiguous
    np.testing.assert_array_equal(result.get(), source.astype(np.float32))
    # Already contiguous float32 device input must not be copied again.
    repeated = device_features(result, 2)
    assert repeated.data.ptr == result.data.ptr


def example(task='multilabel', c=4):
    return LeafEnsemble(
        np.array([[1, .5, -1, -2], [-2, 0, -1, -1]], dtype=np.float32),
        np.array([0, 1, 2], dtype=np.int64), np.array([0, 2, 3], dtype=np.int64),
        np.arange(3 * c, dtype=np.float32).reshape(3, c) / 20,
        np.empty((0, 0), dtype=np.float32), np.arange(c, dtype=np.float32) / 10,
        2, task)


@settings(max_examples=45, deadline=None)
@given(values=st.lists(st.floats(-5, 5, allow_nan=False, allow_infinity=False),
                       min_size=1, max_size=40), nan_left=st.booleans())
def test_routing_matches_scalar_oracle(values, nan_left):
    model = example()
    nodes = model.nodes.copy()
    nodes[0, 0] = -1 if nan_left else 1
    model = replace(model, nodes=nodes)
    x = np.column_stack(([*values, np.nan], np.zeros(len(values) + 1))).astype(np.float32)
    expected = []
    for value in x[:, 0]:
        right = not nan_left if np.isnan(value) else value > .5
        expected.append(model.coefficients[int(right)] + model.coefficients[2] + model.bias)
    np.testing.assert_allclose(model.predict(x), probabilities(expected, model.task), atol=2e-8)
    assert np.array_equal(model.leaf_indices(np.array([[.5, 0]], dtype=np.float32))[:, 0], [0, 2])


@settings(max_examples=30, deadline=None)
@given(c=st.integers(2, 12), seed=st.integers(0, 10000))
def test_full_orthogonal_basis_and_rotation_preserve_function(c, seed):
    dense = example(c=c)
    rng = np.random.default_rng(seed)
    b, _ = np.linalg.qr(rng.normal(size=(c, c)))
    mean = rng.normal(size=c)
    compressed = compress(dense, mean, b)
    x = np.array([[-1, 0], [1, np.nan]], dtype=np.float32)
    np.testing.assert_allclose(compressed.predict(x), dense.predict(x), atol=2e-7)


def test_affine_mean_and_discarded_singular_energy():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(60, 4)) + [10, 20, 30, 40]
    mean, b, singular = functional_basis(y, 'multilabel')
    for rank in (1, 2, 3, 4):
        reconstructed = mean + (y - mean) @ b[:, :rank] @ b[:, :rank].T
        assert np.sum((reconstructed - y) ** 2) == pytest.approx(np.sum(singular[rank:] ** 2), abs=1e-10)
    # A low-energy direction can carry a rare decision: it must not be silently
    # interpreted as unimportant for labels just because PCA drops it.
    rare = np.zeros((100, 2)); rare[:, 0] = np.linspace(-100, 100, 100); rare[0, 1] = .2
    m, b, _ = functional_basis(rare, 'multilabel')
    assert np.linalg.norm((rare - m) @ b[:, :1] @ b[:, :1].T + m - rare) > .1


def test_isotropic_spectrum_has_no_preferred_low_rank_direction():
    y = np.vstack((np.eye(4), -np.eye(4)))
    _, _, singular = functional_basis(y, 'multilabel')
    np.testing.assert_allclose(singular ** 2, 2)
    assert np.sum(singular[1:] ** 2) / np.sum(singular ** 2) == pytest.approx(.75)


def test_softmax_gauge_and_reference_class_are_lossless():
    dense = example(task='multiclass')
    x = np.array([[-1, 0], [1, 0]], dtype=np.float32)
    leaves = dense.leaf_indices(x)
    y = dense.contributions(leaves)
    mean, b, _ = functional_basis(np.tile(y, (3, 1)), dense.task)
    assert b.shape == (4, 3)
    np.testing.assert_allclose(b.T @ np.ones(4), 0, atol=1e-12)
    np.testing.assert_allclose(compress(dense, mean, b).predict(x), dense.predict(x), atol=1e-7)
    ref = reference_class_model(dense)
    np.testing.assert_allclose(ref.predict(x), dense.predict(x), atol=1e-7)
    assert ref.parameter_count == (len(dense.coefficients) + 1) * 3
    shifted = replace(dense, coefficients=dense.coefficients + np.array([[10], [-8], [3]], np.float32),
                      bias=dense.bias + 5)
    np.testing.assert_allclose(shifted.predict(x), dense.predict(x), atol=5e-7)


def test_roundtrip_has_no_dense_teacher_and_counts_all_arrays(tmp_path):
    dense = example(c=8)
    compressed = compress(dense, np.arange(8), np.eye(8)[:, :2])
    path = tmp_path / 'standalone.npz'
    compressed.save(path)
    loaded = LeafEnsemble.load(path)
    assert loaded.coefficients.shape == (3, 2)
    assert loaded.parameter_count == 3 * 2 + 8 * 2 + 8
    assert loaded.payload_bytes == sum(getattr(loaded, n).nbytes for n in
        ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias'))
    assert path.stat().st_size > loaded.payload_bytes
    x = np.array([[0, 0], [1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(loaded.predict(x), compressed.predict(x))


def test_tree_and_leaf_permutations_preserve_function():
    dense = example()
    nodes = dense.nodes[::-1].copy()
    permuted = replace(dense, nodes=nodes, leaf_offsets=np.array([0, 1, 3], dtype=np.int64),
                       coefficients=dense.coefficients[[2, 0, 1]])
    x = np.array([[0, 0], [1, 0]], np.float32)
    np.testing.assert_allclose(permuted.predict(x), dense.predict(x))
    nodes = dense.nodes.copy(); nodes[0, 2:] = [-2, -1]
    leaf_swapped = replace(dense, nodes=nodes, coefficients=dense.coefficients[[1, 0, 2]])
    np.testing.assert_allclose(leaf_swapped.predict(x), dense.predict(x))


@pytest.mark.parametrize('bad', [[1, .5, 0, -1], [1, .5, -1, -9], [1, .5, .5, -1]])
def test_corrupted_routing_rejected(bad):
    nodes = example().nodes.copy(); nodes[0] = bad
    with pytest.raises(ValueError):
        replace(example(), nodes=nodes)


def test_selection_obeys_three_guards_and_uses_predeclared_fallback():
    ref = {'loss': 1., 'f1_macro': .8, 'f1_weighted': .9}
    rows = [dict(rank=1, parameters=20, quality={**ref, 'f1_macro': .1}),
            dict(rank=2, parameters=50, quality=ref),
            dict(rank=3, parameters=90, quality=ref)]
    assert select_rank(rows, ref, 100) == (2, True)
    assert select_rank(rows, ref, 60) == (2, False)
    assert select_rank(rows[:1], ref, 100) == (1, False)
