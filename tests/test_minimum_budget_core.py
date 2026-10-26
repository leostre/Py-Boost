"""Independent exhaustive oracles for minimum cost, Pareto and missing choices."""
import inspect
from hypothesis import given, settings, strategies as st

from experiments.minimum_budget_core import (choose_minimum, quality_frontier,
    QualifiedMinimum, NoQualifiedMinimum)
from experiments.sparse_leaf_core import Budget

REFERENCE = {'loss': 1., 'f1_macro': .5, 'f1_weighted': .6}


def row(cid, memory, parameters=100, macro=.5, loss=1., weighted=.6, kind='rank_only', nnz=0):
    return {'candidate_id': str(cid), 'payload_bytes': memory, 'parameters': parameters,
            'kind': kind, 'nnz': nnz,
            'validation': {'loss': loss, 'f1_macro': macro, 'f1_weighted': weighted}}


def qualified(r):
    q = r['validation']
    return q['loss'] <= 1.05 and q['f1_macro'] >= .475 and q['f1_weighted'] >= .57


def test_minimum_bytes_wins_over_better_f1_and_no_invalid_fallback():
    rows = [row('better_f1', 700, macro=.7), row('cheaper', 500, macro=.49),
            row('fails_weighted', 200, macro=.7, weighted=.56)]
    budget = Budget('joint', .75, 200, 800)
    selected = choose_minimum(rows, REFERENCE, 1000, 'rank_only', budget)
    assert isinstance(selected, QualifiedMinimum) and selected.candidate['candidate_id'] == 'cheaper'
    assert selected.qualified_count == 2
    result = choose_minimum(rows, REFERENCE, 1000, 'rank_only', Budget('joint', .5, 200, 400))
    assert isinstance(result, NoQualifiedMinimum) and result.feasible_count == 1
    assert isinstance(choose_minimum([], REFERENCE, 1000, 'rank_only', budget), NoQualifiedMinimum)


@settings(max_examples=80, deadline=None)
@given(specs=st.lists(st.tuples(st.integers(100, 1000), st.integers(10, 200),
          st.integers(40, 70), st.integers(90, 110), st.integers(50, 70)), min_size=0, max_size=25),
       memory_cap=st.integers(100, 1000), parameter_cap=st.integers(10, 200))
def test_selection_matches_exhaustive_oracle_and_is_permutation_invariant(specs, memory_cap, parameter_cap):
    rows = [row(i, b, p, f/100, loss/100, weighted/100)
            for i, (b, p, f, loss, weighted) in enumerate(specs)]
    budget = Budget('joint', .75, parameter_cap, memory_cap)
    eligible = [r for r in rows if r['payload_bytes'] <= memory_cap and r['parameters'] <= parameter_cap and qualified(r)]
    actual = choose_minimum(rows, REFERENCE, 1000, 'rank_only', budget)
    reverse = choose_minimum(list(reversed(rows)), REFERENCE, 1000, 'rank_only', budget)
    assert actual == reverse
    if not eligible:
        assert isinstance(actual, NoQualifiedMinimum)
    else:
        oracle = sorted(eligible, key=lambda r: (r['payload_bytes'], r['parameters'],
            -r['validation']['f1_macro'], r['validation']['loss'], r['candidate_id']))[0]
        assert actual.candidate == oracle
        assert actual.qualified_count == len(eligible)


@settings(max_examples=60, deadline=None)
@given(specs=st.lists(st.tuples(st.integers(100, 1000), st.integers(10, 200),
                              st.integers(48, 70)), min_size=1, max_size=25))
def test_frontier_matches_pairwise_dominance_and_preserves_cost_minimum(specs):
    rows = [row(i, b, p, f/100) for i, (b, p, f) in enumerate(specs)]
    def axes(r):
        return (r['payload_bytes'], r['parameters'], -r['validation']['f1_macro'])
    expected = {axes(r) for r in rows if not any(all(a <= b for a, b in zip(axes(other), axes(r)))
        and any(a < b for a, b in zip(axes(other), axes(r))) for other in rows)}
    actual = quality_frontier(rows)
    assert {axes(r) for r in actual} == expected
    assert len(actual) == len(expected)
    assert min(r['payload_bytes'] for r in actual) == min(r['payload_bytes'] for r in rows)


@given(memory_cap=st.integers(200, 1000), extra=st.integers(0, 500))
def test_relaxing_cap_or_adding_sparse_family_cannot_increase_validation_minimum(memory_cap, extra):
    rows = [row('rank', 400), row('sparse', 300, kind='sparse', nnz=15)]
    narrow = Budget('joint', .5, 200, memory_cap)
    wide = Budget('joint', .75, 200, memory_cap+extra)
    for family in ('rank_only', 'sparse_01', 'sparse_02', 'sparse_05'):
        a = choose_minimum(rows, REFERENCE, 1000, family, narrow)
        b = choose_minimum(rows, REFERENCE, 1000, family, wide)
        if isinstance(a, QualifiedMinimum):
            assert isinstance(b, QualifiedMinimum)
            assert b.candidate['payload_bytes'] <= a.candidate['payload_bytes']
    a = choose_minimum(rows, REFERENCE, 1000, 'rank_only', wide)
    b = choose_minimum(rows, REFERENCE, 1000, 'sparse_05', wide)
    if isinstance(a, QualifiedMinimum):
        assert b.candidate['payload_bytes'] <= a.candidate['payload_bytes']


def test_api_does_not_accept_test_metrics_and_all_quality_constraints_are_binding():
    assert not any('test' in p for p in inspect.signature(choose_minimum).parameters)
    budget = Budget('joint', .75, 100, 500)
    for bad in (row('loss', 200, loss=1.051), row('macro', 200, macro=.474),
                row('weighted', 200, weighted=.569)):
        assert isinstance(choose_minimum([bad], REFERENCE, 1000, 'rank_only', budget), NoQualifiedMinimum)


def test_plan_preserves_full_reference_and_keeps_missing_state_in_denominator():
    from experiments.minimum_budget_study import minimum_plan
    from dataclasses import asdict
    cheap, high = row('cheap', 300, macro=.49), row('high', 600, macro=.6)
    parent = {'validation_candidates': [cheap, high], 'validation_reference': REFERENCE,
              'reference_cost': {'parameters': 1000, 'payload_bytes': 1000},
              'selections': [{'candidate': high, 'budget': asdict(Budget('joint', .75, 750, 750)),
                'family': 'rank_only', 'residual_fraction_cap': 0, 'validation_quality_pass': True},
                {'candidate': cheap, 'budget': asdict(Budget('joint', .5, 50, 500)),
                 'family': 'rank_only', 'residual_fraction_cap': 0, 'validation_quality_pass': False}]}
    decisions = minimum_plan(parent)
    assert len(decisions) == 4
    assert decisions[0]['candidate'] == cheap and decisions[1]['candidate'] == high
    assert decisions[2]['candidate'] is None and decisions[2]['status'] == 'no_qualified_candidate'
    assert decisions[3]['candidate'] == cheap and not decisions[3]['validation_quality_pass']
    assert parent['selections'][0]['candidate'] == high
