from dataclasses import replace
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.model_bank_core import (MemoryItem, owned_bytes, capacity_prefix,
    executor_name, cohort_choices, cpu_items, latency_summary,task_order,traffic_schedule)
from experiments.shared_leaf_basis import LeafEnsemble
from experiments.uniform_threshold_core import ThresholdModel


@given(st.lists(st.integers(0, 10000), max_size=40), st.integers(0, 10000))
@settings(max_examples=100, deadline=None)
def test_unique_bytes_match_independent_buffer_lengths(sizes, gap):
    pointer, items = 1, []
    for i, size in enumerate(sizes):
        items.append(MemoryItem(str(i), pointer, size))
        pointer += size+gap+1
    assert owned_bytes(list(reversed(items))) == sum(sizes)
    if len([i for i in items if i.size]) >= 1:
        nonempty = next(i for i in items if i.size)
        with pytest.raises(ValueError, match='overlapping'):
            owned_bytes([*items, replace(nonempty, name='alias')])


def test_overlapping_slices_not_only_identical_pointer():
    with pytest.raises(ValueError, match='overlapping'):
        owned_bytes([MemoryItem('whole', 100, 16), MemoryItem('slice', 108, 8)])


@given(st.lists(st.integers(1, 1000), min_size=1, max_size=30),
       st.integers(0, 30000), st.integers(0, 30000))
def test_fixed_order_capacity_matches_linear_reference_and_monotonicity(sizes, b1, b2):
    rows = [dict(resident_bytes=n, development_pass=True) for n in sizes]
    def oracle(b):
        for i in range(len(sizes)+1):
            if sum(sizes[:i]) > b:
                return i-1
        return len(sizes)
    assert capacity_prefix(rows, b1)['count'] == oracle(b1)
    assert capacity_prefix(rows, min(b1,b2))['count'] <= capacity_prefix(rows, max(b1,b2))['count']


def test_failed_quality_is_not_capacity_even_if_free():
    rows = [dict(resident_bytes=1, development_pass=False)]
    assert capacity_prefix(rows, 100000)['count'] == 0


@pytest.mark.parametrize('n,expected', [(1,'groups'), (128,'groups'), (129,'unified_wide'),
                                     (159,'unified_wide'), (1024,'unified_wide')])
def test_explicit_executor_boundary(n, expected):
    assert executor_name(n) == expected


@pytest.mark.parametrize('n', [0, 1025])
def test_unsupported_width_rejected(n):
    with pytest.raises(ValueError): executor_name(n)


def test_frozen_cohort_does_not_need_test_metrics_and_retains_refusals():
    choice = dict(candidate=dict(candidate_id='x'), artifact='x.npz', state='no_quality_feasible')
    record = dict(full=dict(candidate=dict(candidate_id='full'), artifact='full.npz'),
        decisions=[dict(choice, fraction=.75, family=f, cap_percent=5)
                   for f in ('rank_residual','joint_static')])
    prefix = dict(choices=[dict(choice, fraction=.75, selection_scope='all_depths')])
    result = cohort_choices(record, prefix)
    assert len(result) == 4
    assert result['rank_residual']['state'] == 'no_quality_feasible'
    record['decisions'].append(record['decisions'][0].copy())
    with pytest.raises(ValueError, match='ambiguous'): cohort_choices(record, prefix)


def tiny_model(c=4, r=2):
    rng = np.random.default_rng(23)
    basis = np.linalg.qr(rng.normal(size=(c,r)))[0].astype(np.float32)
    base = LeafEnsemble(np.array([[1,0,-1,-2]], np.float32), np.array([0,1], np.int64),
        np.array([0,2], np.int64), rng.normal(size=(2,r)).astype(np.float32), basis,
        np.zeros(c, np.float32), 1, 'multilabel', 'low_rank')
    return ThresholdModel(base, np.array([0,1], np.int32), np.array([0,c-1], np.int32),
                          np.array([.3,-.2], np.float32), np.zeros(c, np.float32))


def test_snapshot_does_not_retain_original_model_and_byte_accounting_is_exact():
    source = tiny_model()
    prepared = source.prepare_inference()
    expected = prepared.predict(np.array([[-1],[1]], np.float32))
    assert owned_bytes(cpu_items(prepared)) == prepared.resident_bytes
    source.base.coefficients[:] = 999
    source.values[:] = 999
    np.testing.assert_array_equal(prepared.predict(np.array([[-1],[1]],np.float32)), expected)


def test_latency_summary_counts_real_requests():
    result = latency_summary([1,2,3,4,5])
    assert result['requests'] == 5 and result['p50_ms'] == 3
    assert result['qps'] == pytest.approx(1000/3)
    with pytest.raises(ValueError): latency_summary([])


@given(st.integers(1,24),st.integers(1,10),st.sampled_from(['uniform','skew']))
def test_traffic_conserves_count_covers_all_models_and_is_reproducible(tasks,multiple,kind):
    count=tasks*multiple
    one=traffic_schedule(tasks,count,kind,17)
    two=traffic_schedule(tasks,count,kind,17)
    assert len(one)==count and one==two
    assert set(one)==set(range(tasks))
    if kind=='uniform':
        assert max(one.count(i) for i in range(tasks))-min(one.count(i) for i in range(tasks))<=1


def test_round_robin_order_does_not_choose_good_test_tasks():
    records=[dict(dataset=d,seed=s) for d in ('a','b','c') for s in (1,2,3)]
    ordered=task_order(records)
    assert [r['dataset'] for r in ordered[:3]]==['a','b','c']
    assert [r['seed'] for r in ordered]==[1,1,1,2,2,2,3,3,3]
    assert records[1]['dataset']=='a'  # No mutation of frozen source order.
