from dataclasses import replace
import pytest
from hypothesis import given, settings, strategies as st
from abl_stop_rule_core import (Quality, Anchor, Candidate, Phase, StopReason,
    Accepted, Stopped, initialize, decide, quality_failures)


def item(uid='c', raw=.8, loss=1., macro=.8, cost=80, mse=0., ordinal=0):
    return Candidate(uid, uid.split('/')[0], raw, Quality(loss, macro, macro), cost, mse, ordinal)


def test_two_phases_and_rollback_keep_original_anchor():
    anchor = Anchor('frozen-full', Quality(1., .8, .8))
    state = initialize([item('warm', raw=1.4, loss=1.4, macro=.6)], anchor, 100)
    assert state.phase is Phase.SEEKING
    first = decide(state, [item('first', raw=1.1, loss=1.1, macro=.7)])
    assert isinstance(first, Accepted) and first.state.certificate is None
    second = decide(first.state, [item('second', raw=1., macro=.78)])
    assert second.state.phase is Phase.QUALIFIED and second.entered_qualification
    rejected = decide(second.state, [item('third', raw=.9, loss=1., macro=.74)])
    assert isinstance(rejected, Stopped) and rejected.reason is StopReason.QUALITY
    assert rejected.state.certificate.uid == 'second'
    assert rejected.state.anchor is anchor


def test_five_percent_does_not_reset_against_previous_checkpoint():
    anchor = Anchor('full', Quality(1., 1., 1.))
    state = initialize([item('a', raw=1.2, macro=1.)], anchor, 100)
    one = decide(state, [item('b', raw=1.1, macro=.96)])
    assert isinstance(one, Accepted)
    two = decide(one.state, [item('c', raw=1., macro=.94)])
    assert .94 > .95*.96  # Would pass the old previous-model interpretation.
    assert isinstance(two, Stopped)


def test_better_temperature_cannot_fabricate_raw_optimization_gain():
    anchor = Anchor('full', Quality(1., .8, .8))
    state = initialize([item('old', raw=1.)], anchor, 100)
    rejected = decide(state, [item('temperature_only', raw=1., loss=.6, macro=.9)])
    assert rejected.reason is StopReason.PROGRESS


def test_all_candidates_are_considered_and_qualification_has_priority():
    anchor = Anchor('full', Quality(1., .8, .8))
    state = initialize([item('old', raw=2., loss=1.5, macro=.5)], anchor, 100)
    new = decide(state, [item('cheapest_bad', raw=1., macro=.5, mse=0.),
        item('qualifies', raw=1.1, macro=.78, mse=1.)])
    assert new.state.current.uid == 'qualifies'


def test_exact_memory_boundary_and_zero_reference_f1():
    anchor = Anchor('full', Quality(1., 0., 0.))
    state = initialize([item('old', raw=1.2, macro=0., cost=100)], anchor, 100)
    assert isinstance(decide(state, [item('fits', raw=1., macro=0., cost=100)]), Accepted)
    assert decide(state, [item('too_big', raw=.1, loss=.1, macro=1., cost=101)]).reason is StopReason.MEMORY


@settings(max_examples=70, deadline=None)
@given(st.lists(st.tuples(st.floats(.01, 2., allow_nan=False),
    st.floats(.0, 1., allow_nan=False), st.integers(1, 120)), min_size=1, max_size=15))
def test_qualified_sequence_preserves_fixed_quality_and_memory(commands):
    anchor = Anchor('unchanged', Quality(1., .8, .8))
    state = initialize([item('initial', raw=2.)], anchor, 100)
    for i, (loss, macro, cost) in enumerate(commands):
        before = state
        result = decide(state, [item(str(i), raw=max(.00001, state.current.raw_nll-.01),
            loss=loss, macro=macro, cost=cost)])
        if isinstance(result, Accepted):
            state = result.state
            assert cost <= 100 and loss <= 1.05 and macro >= .76
            assert state.current.raw_nll < before.current.raw_nll
        else:
            assert result.state is before
        assert state.anchor is anchor and state.budget_bytes == 100
        assert state.phase is Phase.QUALIFIED


@settings(max_examples=60, deadline=None)
@given(st.permutations((0, 1, 2)))
def test_candidate_order_does_not_change_decision(order):
    anchor = Anchor('full', Quality(1., .8, .8))
    state = initialize([item('old', raw=2.)], anchor, 100)
    candidates = [item('a', raw=1., mse=.1, ordinal=0),
        item('b', raw=1.1, mse=.01, ordinal=1), item('c', raw=.9, cost=101, ordinal=2)]
    result = decide(state, [candidates[i] for i in order])
    assert result.state.current.uid == 'b'


def test_cached_candidate_adapter_cannot_read_selection_labels():
    from abl_stop_rule_replay import typed_rows
    row = dict(percent=5, temperature=1., validation=dict(loss=.7, f1_macro=.8, f1_weighted=.9),
        cost_bytes=80, train_weighted_mse=.1, selection=dict(loss=100., f1_macro=0., f1_weighted=0.))
    first = typed_rows([row], 'saved')
    row['selection'] = dict(loss=0., f1_macro=1., f1_weighted=1.)
    assert typed_rows([row], 'saved') == first


def test_global_policy_does_not_change_raw_continuation_signature():
    import numpy as np
    from test_abl_quality_core import fixture
    from abl_protected_core import as_protected
    from abl_stop_rule_replay import raw_signature
    m, _, _, _ = fixture()
    full = as_protected(m)
    assert raw_signature(full) == raw_signature(full.with_shifts(np.ones(4, np.float32)))
    assert raw_signature(full) != raw_signature(replace(full, bias=full.bias+np.float32(.1)))


def test_replay_boundary_with_synthetic_saved_models_and_independent_audit(tmp_path):
    import json
    import numpy as np
    from budget_quality_worker import write, digest
    from test_abl_increment_diagnostic import test_saved_artifact_diagnostic_boundary_and_independent_audit
    from abl_protected_core import load_protected, block_cost
    from abl_increment_diagnostic_core import task_quality
    from abl_stop_rule_replay import case_replay
    from audit_abl_stop_rule import audit
    # Reuse the earlier synthetic no-fit archive, including accepted/rejected steps.
    test_saved_artifact_diagnostic_boundary_and_independent_audit(tmp_path)
    p = tmp_path/'case.json'; source = json.loads(p.read_text())
    with np.load(tmp_path/'data.npz') as f:
        x, y = f['features'], f['target']
    dev, ys = x[20:], y[32:]
    warm = json.loads((tmp_path/'warmup.json').read_text())
    metas = {}
    def meta(uid, item):
        m = load_protected(item['artifact']); z = m.raw(dev)
        cache = tmp_path/(uid.replace('/', '-')+'-dev.npz'); np.savez(cache, raw=z)
        return dict(item, uid=uid, source=dict(artifact=item['artifact'], sha256=item['sha256']),
            development_cache=dict(path=str(cache), sha256=digest(cache)), **block_cost(m))
    for ref in source['trajectories']:
        t = json.loads(__import__('pathlib').Path(ref['path']).read_text())
        for item_ in t['models']:
            uid = ref['mode']+f"/b{item_['block']:02d}"
            metas[uid] = meta(uid, item_)
    metas['full/b00'] = meta('full/b00', warm['models'][0])
    model = load_protected(warm['models'][0]['artifact'])
    q = task_quality(ys, model.raw(x[32:]), 'multiclass')
    row = dict(uid='full/b00', validation=dict(raw=q),
        refit=dict(temperature=1., raw_shifts=[0.]*4, temperature_shifts=[0.]*4))
    source.update(metas=metas, rows=[row], choices=[dict(mode='temperature_shifts', objective='f1_macro',
        group=g, chosen=dict(uid='full/b00', policy='raw', validation=q)) for g in
        ('full_unbounded', 'functional', 'current', 'mean', 'history')])
    write(p, source)
    hd = tmp_path/'out/cases/fixture_1.json'; d = json.loads(hd.read_text())
    d['input']['sha256'] = digest(p); write(hd, d)
    cfg = dict(decision_grid=[0, .5, -.5, 1, -1, 2, -2], decision_shrinkage=[.25, .5, 1.])
    value = case_replay(dict(case='fixture_1', path=str(hd), sha256=digest(hd)), cfg, tmp_path/'rule', {})
    checked = audit([value])
    assert checked['status'] == 'passed' and checked['cases'][0]['paths'] == 4
