from dataclasses import replace
import copy
import numpy as np
import pytest
from hypothesis import given, strategies as st

from experiments.inference_benchmark_core import frozen_choices, batch_sizes, timing_summary, paired_ratios
from experiments.shared_leaf_basis import LeafEnsemble, compress, functional_basis, reference_class_model
from experiments.uniform_threshold_core import ThresholdModel


def test_frozen_choices_ignores_unrelated_test_quality_and_preserves_selection():
    record = {'full': {'artifact': 'full'}, 'selections': [
        {'family': family, 'budget': {'kind': 'joint', 'fraction': fraction}, 'artifact': family+str(fraction)}
        for family in ('rank_only', 'sparse_05') for fraction in (.5, .75)]}
    before = copy.deepcopy(record)
    result = frozen_choices(record)
    assert {k: v['artifact'] for k, v in result.items()} == {
        'full': 'full', 'rank': 'rank_only0.75', 'hybrid': 'sparse_050.75'}
    assert record == before
    record['selections'].append(copy.deepcopy(record['selections'][-1]))
    with pytest.raises(ValueError):
        frozen_choices(record)


@given(st.integers(1, 10000))
def test_batch_sizes_never_repeat_or_invent_objects(n):
    sizes = batch_sizes(n)
    assert sizes == sorted(set(sizes))
    assert sizes[0] == 1 and sizes[-1] == n
    assert all(1 <= b <= n for b in sizes)
    assert all(min(b, n) in sizes for b in (32, 256))


def test_paired_summary_is_not_ratio_of_separate_medians():
    assert paired_ratios([1, 100, 101], [2, 3, 202]) == 2
    summary = timing_summary([3, 1, 2], 20)
    assert summary['median_ms'] == 2 and summary['rows_per_second'] == 10000
    with pytest.raises(ValueError):
        timing_summary([0], 1)
    with pytest.raises(ValueError):
        paired_ratios([1], [np.nan])


def example(task):
    return LeafEnsemble(np.array([[1, .5, -1, -2], [-2, 0, -1, -1]], np.float32),
        np.array([0, 1, 2], np.int64), np.array([0, 2, 3], np.int64),
        np.arange(12, dtype=np.float32).reshape(3, 4)/20,
        np.empty((0, 0), np.float32), np.arange(4, dtype=np.float32)/10, 2, task)


@pytest.mark.parametrize('task', ['multilabel', 'multiclass'])
@pytest.mark.parametrize('role', ['full', 'rank', 'hybrid'])
def test_gpu_adapter_matches_current_cpu_including_nan_routes(task, role):
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    from experiments.inference_benchmark_gpu import GpuThresholdReference
    x = np.array([[0., np.nan], [1., 0.], [np.nan, 2.]], np.float32)
    dense = example(task)
    if role == 'full':
        base = reference_class_model(dense) if task == 'multiclass' else dense
    else:
        mean, basis, _ = functional_basis(dense.contributions(dense.leaf_indices(x)), task)
        base = compress(dense, mean, basis[:, :2])
    rows, columns, values = np.array([], np.int32), np.array([], np.int32), np.array([], np.float32)
    if role == 'hybrid':
        rows, columns, values = np.array([0, 2], np.int32), np.array([1, 3], np.int32), np.array([-.2, .3], np.float32)
    model = ThresholdModel(base, rows, columns, values, np.zeros(4, np.float32))
    gpu = GpuThresholdReference(model)
    np.testing.assert_allclose(gpu.predict(x), model.predict(x), rtol=0, atol=2e-5)
    np.testing.assert_array_equal(gpu.route(cp.asarray(x)).get(), model.base.leaf_indices(x))
