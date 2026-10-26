"""Independent byte enumeration, not a duplicate of the matching formula."""
from dataclasses import replace
import inspect
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from experiments.rank_residual_map_core import representation_bytes, matched_allocations, choose_family
from experiments.prepared_leaf_inference import PreparedLeafInference
from experiments.uniform_threshold_core import ThresholdModel
from test_prepared_leaf_inference import model_for


@settings(max_examples=100, deadline=None)
@given(m=st.integers(1, 15), c=st.integers(2, 12), q0=st.integers(0, 100),
       cap0=st.integers(0, 200), storage=st.sampled_from(['coo', 'csr']))
def test_all_matches_equal_bruteforce_array_byte_oracle(m, c, q0, cap0, storage):
    q, cap, fixed = 1 + q0 % c, min(cap0, m*c), 76
    target = fixed + np.empty((m, q), np.float32).nbytes + np.empty((c, q), np.float32).nbytes
    expected = []
    for rank in range(1, q):
        candidates = []
        for count in range(1, cap + 1):
            value_bytes = np.empty(count, np.float32).nbytes
            index_bytes = (2*np.empty(count, np.int32).nbytes if storage == 'coo' else
                           np.empty(count, np.int32).nbytes + np.empty(m+1, np.int32).nbytes)
            cost = fixed + 4*m*rank + 4*c*rank + value_bytes + index_bytes
            if 0 <= target-cost < (12 if storage == 'coo' else 8):
                candidates.append((rank, count, cost, target-cost))
        expected.extend(candidates)
    actual = matched_allocations(fixed, m, c, q, cap, storage)
    assert [(a.rank, a.nnz, a.matching_bytes, a.slack_bytes) for a in actual] == expected


@pytest.mark.parametrize('role', ['rank', 'hybrid'])
@pytest.mark.parametrize('task', ['multilabel', 'multiclass'])
def test_cost_matches_real_coo_and_prepared_csr_arrays(role, task):
    model = model_for(task, role)
    m, rank = model.base.coefficients.shape
    c = len(model.base.bias)
    fixed = sum(getattr(model.base, k).nbytes for k in ('nodes', 'node_offsets', 'leaf_offsets', 'bias'))
    fixed += model.decision_shifts.nbytes
    assert representation_bytes(fixed, m, c, rank, len(model.values), 'coo') == model.payload_bytes
    assert representation_bytes(fixed, m, c, rank, len(model.values), 'csr') == PreparedLeafInference.prepare(model).resident_bytes


def test_csr_empty_has_no_pointer_and_nonempty_pays_pointer():
    empty = representation_bytes(100, 20, 10, 3, 0, 'csr')
    assert representation_bytes(100, 20, 10, 3, 1, 'csr')-empty == 4*21+8
    assert representation_bytes(100, 20, 10, 3, 1, 'coo')-empty == 12


@pytest.mark.parametrize('args', [(0, 0, 5, 1, 0, 'coo'), (0, 3, 5, 0, 0, 'coo'),
                                 (0, 3, 5, 6, 0, 'csr'), (0, 3, 5, 1, 16, 'coo'),
                                 (0, 3, 5, 1, 0, 'dense'), (0, 3, 5, 1, -1, 'csr')])
def test_invalid_domain_rejected(args):
    with pytest.raises(ValueError):
        representation_bytes(*args)


def row(cid, nnz, macro, loss=1., weighted=.8):
    return {'candidate_id': cid, 'nnz': nnz, 'payload_bytes': 1000, 'parameters': 100,
            'validation': {'f1_macro': macro, 'loss': loss, 'f1_weighted': weighted}}


def test_family_guard_and_zero_control_no_false_universal_win():
    reference = {'loss': 1., 'f1_macro': .8, 'f1_weighted': .8}
    anchor = row('anchor', 0, .8)
    good, too_large, bad_loss = row('good', 2, .82), row('large', 3, .84), row('bad', 1, .9, 1.06)
    chosen, passed = choose_family([good, too_large, bad_loss], anchor, reference, 2, 100)
    assert chosen is good and passed
    chosen, passed = choose_family([good], anchor, reference, 1, 100)
    assert chosen is anchor and passed
    chosen, passed = choose_family([], row('bad_anchor', 0, .4), reference, 5, 100)
    assert not passed and chosen['candidate_id'] == 'bad_anchor'
    assert 'test' not in inspect.signature(choose_family).parameters


@settings(max_examples=40, deadline=None)
@given(macros=st.lists(st.floats(.5, 1., allow_nan=False), min_size=1, max_size=10))
def test_more_residual_allowance_cannot_remove_validation_options(macros):
    reference = {'loss': 1., 'f1_macro': .8, 'f1_weighted': .8}
    anchor = row('anchor', 0, .8)
    rows = [row(str(i), i+1, v) for i, v in enumerate(macros)]
    choices = [choose_family(rows, anchor, reference, p, 100)[0] for p in (0, 1, 2, 5)]
    assert all(a['validation']['f1_macro'] <= b['validation']['f1_macro']
               for a, b in zip(choices, choices[1:]))


def test_entire_development_plan_does_not_observe_test_labels(tmp_path):
    """Synthetic frozen ensemble; poisoning test labels cannot change selection."""
    from experiments.shared_leaf_basis import LeafEnsemble, functional_basis, compress
    from experiments.a100_runner import write_report, sha256
    from experiments.sparse_leaf_core import leaf_design
    from experiments.uniform_threshold_core import threshold_variants
    from experiments.rank_residual_map_study import prepare_case, evaluate_case
    from experiments.output_calibration_core import calibrated_quality
    rng = np.random.default_rng(418)
    m, c, n = 32, 14, 128
    nodes = []
    def subtree(lo, hi):
        if hi-lo == 1:
            return -lo-1
        index = len(nodes)
        nodes.append(None)
        middle = (lo+hi)//2
        left, right = subtree(lo, middle), subtree(middle, hi)
        nodes[index] = [1, middle-.5, left, right]
        return index
    subtree(0, m)
    dense = LeafEnsemble(np.array(nodes, np.float32), np.array([0, m-1], np.int64),
        np.array([0, m], np.int64), rng.normal(size=(m, c)).astype(np.float32),
        np.empty((0, 0), np.float32), np.zeros(c, np.float32), 1, 'multilabel')
    x = np.tile(np.arange(m), 4).astype(np.float32)[:, None]
    y = rng.integers(0, 2, (n, c))
    parts = {'train': np.arange(64), 'calibration': np.arange(64, 88), 'selection': np.arange(88, 112)}
    source, previous, output = tmp_path/'source'/'toy_7', tmp_path/'old'/'toy_7', tmp_path/'output'
    source.mkdir(parents=True)
    previous.mkdir(parents=True)
    np.savez(source/'splits.npz', source_indices=np.arange(n), train=parts['train'],
             validation=np.r_[parts['calibration'], parts['selection']], test=np.arange(112, n))
    np.savez(previous/'development_splits.npz', **parts)
    dense.save(source/'dense_teacher.npz')
    dense.save(source/'reference.npz')
    mean, basis, _ = functional_basis(dense.contributions(dense.leaf_indices(x[parts['train']])), dense.task)
    np.savez(source/'calibration_diagnostics.npz', mean=mean, basis=basis)
    designs = {k: leaf_design(dense.leaf_indices(x[v]), m) for k, v in parts.items()}
    grid, shrinkages = [0, -1, 1], [.25, .5, 1.]
    def variants(base, kind, rank, selector):
        rows = threshold_variants(base.raw_from_leaves(base.leaf_indices(x[parts['calibration']])),
            y[parts['calibration']], base.raw_from_leaves(base.leaf_indices(x[parts['selection']])),
            y[parts['selection']], dense.task, grid, shrinkages)
        return [{'candidate_id': f'{kind}-r{rank}-{selector}-0-t'+str(v['threshold_fraction']).replace('.', 'p'),
                 'kind': kind, 'rank': rank, 'nnz': 0, 'selector': selector,
                 'parameters': base.parameter_count+c, 'payload_bytes': base.payload_bytes+4*c, **v} for v in rows]
    full_rows = variants(dense, 'full', c, 'full')
    # A deliberately small synthetic old grid fixes the anchor while retaining
    # real metrics. The scientific grid is not restricted this way.
    rank_rows = [row for rank in [8] for row in
                 variants(compress(dense, mean, basis[:, :rank]), 'rank_only', rank, 'rank_only')]
    record = {'reference_cost': {'parameters': dense.parameter_count, 'payload_bytes': dense.payload_bytes},
              'development_split_sha256': sha256(previous/'development_splits.npz'),
              'validation_reference': full_rows[0]['validation'], 'full': {'candidate': full_rows[0]},
              'validation_candidates': rank_rows, 'full_candidates': full_rows}
    write_report(previous/'selection.json', record)
    old_record = {k: v for k, v in record.items() if k not in ('validation_candidates', 'full_candidates')}
    config = {'source_dir': str(source.parent), 'inference_parent_dir': str(previous.parent),
              'tiny_selectors': ['balanced_mse', 'balanced_margin'], 'margin_scale': .5,
              'budget_fractions': [.9], 'map_storage_formats': ['coo', 'csr'],
              'decision_grid': grid, 'decision_shrinkage': shrinkages}
    data = (x, y, np.arange(n), {'sha256': 'synthetic'})
    selected = prepare_case('toy', 7, config, output, data, old_record)
    poisoned = y.copy()
    poisoned[112:] = 1-poisoned[112:]
    replay = prepare_case('toy', 7, config, tmp_path/'replay', (x, poisoned, np.arange(n), data[3]), old_record)
    assert selected['decisions'] == replay['decisions']
    assert selected['reuse']['threshold_fits'] > 0
    assert len(evaluate_case(selected, config, output, data)) == 8
