"""Direct sklearn resampling oracle, exact 450 boundary, schedule coverage laws."""
from dataclasses import asdict
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.output_calibration_core import calibrated_quality
from experiments.shared_leaf_basis import quality_guard
from experiments.sparse_leaf_core import Budget
from experiments.stable_budget_core import (paired_indices, bootstrap_design, bootstrap_quality,
    paired_pass_mask, QualitySeries, stable_minimum, stability_requests, StableMinimum, NoStableMinimum)


@settings(max_examples=35, deadline=None)
@given(n=st.integers(2, 16), c=st.integers(2, 6), seed=st.integers(0, 100000),
       task=st.sampled_from(['multilabel', 'multiclass']))
def test_weighted_bootstrap_exactly_matches_explicit_object_resampling(n, c, seed, task):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, c))*4
    y = rng.integers(0, c, n) if task == 'multiclass' else rng.integers(0, 2, (n, c))
    shifts = rng.normal(size=c)
    indices = paired_indices(n, 7, seed+1)
    design = bootstrap_design(y, task, c, indices)
    result = bootstrap_quality(design, raw, shifts)
    oracle = [calibrated_quality(y[i], raw[i], task, shifts) for i in indices]
    for field in ('loss', 'f1_macro', 'f1_weighted'):
        np.testing.assert_allclose(getattr(result, field), [q[field] for q in oracle], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(design.weights.sum(axis=0), np.full(7, n))
    np.testing.assert_array_equal(indices, paired_indices(n, 7, seed+1))
    np.testing.assert_array_equal(y, design.target)
    mask = paired_pass_mask(result, result)
    assert mask.all()


def test_zero_support_and_missing_multiclass_labels_follow_original_definitions():
    for task, y in [('multilabel', np.zeros((4, 3), int)), ('multiclass', np.array([0, 0, 2, 2]))]:
        raw = np.array([[20, -20, -20]]*4, float)
        indices = np.array([[0]*4, [2]*4], np.int32)
        series = bootstrap_quality(bootstrap_design(y, task, 3, indices), raw, np.zeros(3))
        for k, idx in enumerate(indices):
            expected = calibrated_quality(y[idx], raw[idx], task, np.zeros(3))
            assert series.f1_macro[k] == pytest.approx(expected['f1_macro'])
            assert series.f1_weighted[k] == pytest.approx(expected['f1_weighted'])


REFERENCE = {'loss': 1., 'f1_macro': .5, 'f1_weighted': .6}


def row(cid, cost, macro=.5, kind='rank_only', nnz=0):
    return {'candidate_id': cid, 'parameters': cost//4, 'payload_bytes': cost, 'kind': kind, 'nnz': nnz,
            'validation': {'loss': 1., 'f1_macro': macro, 'f1_weighted': .6}}


def test_exact_450_boundary_no_invalid_fallback_and_no_unscored_gap():
    rows = [row('cheap', 200), row('stable', 300), row('too_bad', 100, macro=.4)]
    budget = Budget('joint', .75, 1000, 1000)
    q = stable_minimum(rows, REFERENCE, 1000, 'rank_only', budget, {'cheap': 449, 'stable': 450})
    assert isinstance(q, StableMinimum) and q.candidate['candidate_id'] == 'stable'
    assert q.checked_count == 2
    assert isinstance(stable_minimum(rows, REFERENCE, 1000, 'rank_only', budget,
                                    {'cheap': 449, 'stable': 449}), NoStableMinimum)
    with pytest.raises(ValueError, match='unscored cheaper'):
        stable_minimum(rows, REFERENCE, 1000, 'rank_only', budget, {'stable': 500})


@settings(max_examples=50, deadline=None)
@given(specs=st.lists(st.tuples(st.integers(100, 1000), st.integers(0, 500),
                              st.integers(40, 70)), min_size=0, max_size=20))
def test_selection_and_stream_schedule_match_full_independent_enumeration(specs):
    rows = [row(str(i), cost, f/100, 'sparse' if i % 2 else 'rank_only', 15 if i % 2 else 0)
            for i, (cost, count, f) in enumerate(specs)]
    scores = {str(i): count for i, (_, count, _) in enumerate(specs)}
    scopes = [{'budget': asdict(Budget('joint', .75, 200, cap)), 'family': family}
              for cap in (400, 800) for family in ('rank_only', 'sparse_02', 'sparse_05')]
    pending, streamed = set(range(len(scopes))), {}
    for r, membership in stability_requests(rows, REFERENCE, 1000, scopes):
        relevant = pending.intersection(membership)
        if not relevant:
            continue
        streamed[r['candidate_id']] = scores[r['candidate_id']]
        if scores[r['candidate_id']] >= 450:
            pending.difference_update(relevant)
    for scope in scopes:
        actual = stable_minimum(rows, REFERENCE, 1000, scope['family'], Budget(**scope['budget']), streamed)
        eligible = [r for r in rows if r['payload_bytes'] <= scope['budget']['payload_bytes'] and r['parameters'] <= 200
                    and r['validation']['f1_macro'] >= .475 and (r['kind'] == 'rank_only' or scope['family'] != 'rank_only')
                    and scores[r['candidate_id']] >= 450]
        if eligible:
            expected = min(eligible, key=lambda r: (r['payload_bytes'], r['parameters'],
                -r['validation']['f1_macro'], r['validation']['loss'], r['candidate_id']))
            assert actual.candidate == expected
            assert actual == stable_minimum(list(reversed(rows)), REFERENCE, 1000, scope['family'], Budget(**scope['budget']), scores)
        else:
            assert isinstance(actual, NoStableMinimum)


def test_joint_guard_is_not_three_separate_success_rates():
    ref = QualitySeries(np.ones(500), np.ones(500), np.ones(500))
    loss, macro, weighted = np.ones(500), np.ones(500), np.ones(500)
    loss[:40] = 1.06
    macro[40:80] = .94
    weighted[80:120] = .94
    mask = paired_pass_mask(QualitySeries(loss, macro, weighted), ref)
    assert mask.sum() == 380  # each individual check passes 460, but joint only 380


def test_pairing_cannot_be_replaced_by_separate_draws_and_thresholds_stay_fixed():
    y = np.array([[1], [0], [1], [0]])
    raw = np.array([[.2], [-.2], [-.2], [.2]])
    indices = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], np.int32)
    design = bootstrap_design(y, 'multilabel', 1, indices)
    q = bootstrap_quality(design, raw, np.zeros(1))
    np.testing.assert_array_equal(q.f1_macro, [1, 0])
    assert paired_pass_mask(q, q).all()
    changed = bootstrap_quality(design, raw, np.ones(1))
    assert not np.array_equal(q.f1_macro, changed.f1_macro)
    assert paired_pass_mask(q, QualitySeries(q.loss[::-1], q.f1_macro[::-1], q.f1_weighted[::-1])).sum() < 2
