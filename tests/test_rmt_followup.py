"""Independent conservation, causality and objective-contract checks."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.rmt_followup_core import (aggregate_duplicate_sums, duplicate_split_control,
    empirical_checks, gains_from_sums, partition_residual, quota_actions, rare_split_control, split_sums)


@st.composite
def quotas(draw):
    n = draw(st.integers(1, 25))
    probabilities = np.array(draw(st.lists(st.integers(0, 100), min_size=3 * n, max_size=3 * n))).reshape(n, 3) / 100
    prior = np.array(draw(st.lists(st.integers(0, 100), min_size=3, max_size=3))) / 100
    count = draw(st.integers(0, n))
    cut = draw(st.integers(0, n))
    return probabilities, prior, count, cut


@given(case=quotas(), low=st.sampled_from([25, 50, 75]))
@settings(max_examples=40, deadline=None)
def test_exact_quota_conservation_and_causal_prefix(case, low):
    scores, prior, count, cut = case
    decision = quota_actions(scores, prior, low, 101, count)
    assert sum(decision.budgets == low) == count
    assert sum(decision.budgets == 101) == len(scores) - count
    # Mutate only future predictions; keep the ex-ante planned quota identical.
    changed = scores.copy()
    changed[cut:] = 1 - changed[cut:]
    later = quota_actions(changed, prior, low, 101, count)
    np.testing.assert_array_equal(decision.budgets[:cut], later.budgets[:cut])
    np.testing.assert_array_equal(decision.forced[:cut], later.forced[:cut])


def test_known_sequential_opportunity_cost_not_global_sort():
    scores = np.array([[.5, .1, .5], [.5, .9, .5], [.5, .2, .5], [.5, .9, .5]])
    actual = quota_actions(scores, [.5, .5, .5], 50, 100, 2)
    assert actual.budgets.tolist() == [50, 100, 50, 100]
    assert actual.forced.tolist() == [False, False, False, True]
    assert quota_actions(np.ones((3, 3)) * .5, [.5] * 3, 50, 100, 1).budgets.tolist() == [100, 100, 50]


@pytest.mark.parametrize('scores,prior,count', [([[np.nan, 0, 0]], [0, 0, 0], 0),
    ([[2, 0, 0]], [0, 0, 0], 0), ([[0, 0, 0]], [0, 0, 0], 2),
    ([[0, 0, 0]], [0, 0, 0], True), ([[0, 0]], [0, 0, 0], 0)])
def test_quota_boundary_rejects_invalid_inputs(scores, prior, count):
    with pytest.raises(ValueError):
        quota_actions(scores, prior, 50, 100, count)


def test_finite_permutation_resolution_ties_and_64_test_holm():
    assert empirical_checks(np.zeros(64), np.zeros((100, 64)))['maxT'] == []
    strong = empirical_checks(np.ones(64) * 10, np.ones((100, 64)))
    assert strong['rankwise'] == list(range(64))
    assert strong['holm'] == []  # 1/101 exceeds .05/64, even for huge effects.
    assert strong['maxT'] == list(range(64))
    short = empirical_checks(np.ones(3) * 10, np.ones((10, 3)))
    assert short['maxT'] == []  # no p can be <=.05 at B=10
    assert short['p_rankwise'] == [1 / 11] * 3


@given(order=st.permutations([0, 1, 2, 3]))
@settings(max_examples=20, deadline=None)
def test_maxT_coordinate_reordering_preserves_selected_labels(order):
    observed = np.array([10, .2, 5, .3])
    nulls = np.stack([np.linspace(0, 1, 100)] * 4, axis=1)
    before = empirical_checks(observed, nulls)
    after = empirical_checks(observed[list(order)], nulls[:, list(order)])
    assert set(before['maxT']) == {order[i] for i in after['maxT']}
    assert len(before['maxT']) >= 1


@given(multiplicity=st.integers(2, 12), l2=st.integers(1, 8),
       gradients=st.lists(st.integers(-8, 8), min_size=8, max_size=8),
       hessians=st.lists(st.integers(1, 8), min_size=8, max_size=8))
@settings(max_examples=40, deadline=None)
def test_exact_duplicate_objective_and_leaf_conservation(multiplicity, l2, gradients, hessians):
    g, h = np.array(gradients, float), np.array(hessians, float)
    mask = np.arange(8) < 3
    G, H = np.tile(g[:, None], (1, multiplicity)), np.tile(h[:, None], (1, multiplicity))
    sums = split_sums(G, H, [mask])
    merged, penalty = aggregate_duplicate_sums(sums, [list(range(multiplicity))], l2, True)
    # Direct scalar Newton formula, not another call to the system under test.
    gl, gr, hl, hr = g[mask].sum(), g[~mask].sum(), h[mask].sum(), h[~mask].sum()
    reference = multiplicity * .5 * (gl ** 2 / (hl + l2) + gr ** 2 / (hr + l2) - (gl + gr) ** 2 / (hl + hr + l2))
    np.testing.assert_allclose(gains_from_sums(merged, penalty)[0, 0], reference, atol=1e-10)
    np.testing.assert_allclose(-merged['left_g'][0, 0] / (merged['left_h'][0, 0] + penalty[0]), -gl / (hl + l2), atol=1e-10)


def test_noise_cell_residual_has_no_candidate_split_contribution():
    masks = (np.arange(40) < 8, np.arange(40) % 2 == 0)
    value = partition_residual(np.arange(40 * 3).reshape(40, 3), masks)
    for mask in masks:
        np.testing.assert_allclose(value[mask].sum(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(value[~mask].sum(axis=0), 0, atol=1e-10)


def test_control_generators_stipulate_task_utility_not_only_energy():
    p = {'seed': 19, 'functional_rows': 256, 'functional_outputs': 64, 'functional_epochs': 2}
    G, H, masks = rare_split_control(p)
    profiles = gains_from_sums(split_sums(G[-1], H[-1], masks), 1)
    assert profiles.sum(axis=1).argmax() == 0
    assert np.argsort(-(G[-1].astype(float) ** 2).sum(axis=0))[:8].tolist().count(0) == 0
    G, H, masks, groups = duplicate_split_control(p)
    sums = split_sums(G, H, masks)
    full = gains_from_sums(sums, 1).sum(axis=1)
    weighted, penalties = aggregate_duplicate_sums(sums, groups, 1, True)
    unweighted, unweighted_penalties = aggregate_duplicate_sums(sums, groups, 1, False)
    np.testing.assert_allclose(gains_from_sums(weighted, penalties).sum(axis=1), full)
    assert full.argmax() == 1
    assert gains_from_sums(unweighted, unweighted_penalties).sum(axis=1).argmax() == 0
