"""Independent state/metric/storage audit of the bounded saved-path replay."""
import numpy as np
from budget_quality_worker import locked, digest
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected
from audit_abl_protected import physical_cost
from audit_abl_transfer import independent_task_quality
from audit_abl_quality import control


def passed(q, ref):
    return q['loss'] <= 1.05*ref['loss'] and q['f1_macro'] >= .95*ref['f1_macro'] and q['f1_weighted'] >= .95*ref['f1_weighted']


def same_raw(a, b):
    if len(a.blocks) != len(b.blocks) or not np.array_equal(a.bias, b.bias):
        return False
    for x, y in zip(a.blocks, b.blocks):
        if x.base.task != y.base.task or x.base.representation != y.base.representation:
            return False
        for name in ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias'):
            if not np.array_equal(getattr(x.base, name), getattr(y.base, name)):
                return False
        for name in ('rows', 'columns', 'values'):
            if not np.array_equal(getattr(x, name), getattr(y, name)):
                return False
    return True


def final_choice(rows):
    if not rows:
        return None
    top = max(r['selection']['f1_macro'] for r in rows)
    tied = [r for r in rows if top-r['selection']['f1_macro'] <= 1e-12]
    loss = min(r['selection']['loss'] for r in tied)
    return min((r for r in tied if r['selection']['loss']-loss <= 1e-6),
        key=lambda r: (r['cost_bytes'], r['uid']))


def audit(entries):
    checks = []
    for entry in entries:
        result = locked(entry['path'], entry['sha256'])
        source = locked(result['original']['path'], result['original']['sha256'])
        data = data_before_test(source['record']); task = source['task']
        nc, nt, _ = data['sizes']; x = data['dev_x']
        yc, ys = data['dev_y'][nc:nc+nt], data['dev_y'][nc+nt:]
        anchor = result['anchor']; mode = 'temperature_shifts' if task == 'multiclass' else 'shifts'
        choice = next(c for c in source['choices'] if c['mode'] == mode and
            c['objective'] == 'f1_macro' and c['group'] == 'full_unbounded')
        assert choice == anchor['reference_choice']
        uid = choice['chosen']['uid']; meta = source['metas'][uid]
        assert digest(meta['artifact']) == meta['sha256']
        cp = control(next(r for r in source['rows'] if r['uid'] == uid), choice['chosen']['policy'])
        assert cp['temperature'] == anchor['policy']['temperature']
        np.testing.assert_array_equal(cp['shifts'], anchor['policy']['shifts'])
        z = load_protected(meta['artifact']).raw(x)
        maximum = 0.
        for yy, zz, expected in ((yc, z[nc:nc+nt], anchor['control']), (ys, z[nc+nt:], anchor['selection'])):
            q = independent_task_quality(yy, zz, cp, task)
            maximum = max(maximum, *(abs(q[k]-expected[k]) for k in expected))
        assert maximum < 2e-8
        assert result['budget_bytes'] == source['budget_bytes']
        for one in result['one_step']:
            for c, g in zip(one['candidates'], one['gates']):
                assert g['uid'] == c['uid']
                assert g['memory'] == (c['cost_bytes'] <= source['budget_bytes'])
                assert g['useful'] == (one['previous_raw_nll']-c['raw_nll'] > 1e-6)
                assert g['fixed_quality'] == passed(c['quality'], anchor['control'])
        for path in result['paths']:
            phase = path['initial_state']['phase']
            current_raw = path['initial_state']['current']['raw_nll']
            snapshots = [path['initial']]
            for index, attempt in enumerate(path['attempts']):
                before = attempt['before']
                assert before['anchor']['token'] == anchor['token'] and before['anchor']['quality'] == anchor['control']
                assert before['budget_bytes'] == source['budget_bytes'] and before['phase'] == phase
                assert abs(before['current']['raw_nll']-current_raw) < 2e-8
                feasible = [c for c in attempt['candidate_rows'] if c['cost_bytes'] <= source['budget_bytes']]
                useful = [c for c in feasible if current_raw-c['raw_nll'] > 1e-6]
                ready = [c for c in useful if passed(c['quality'], anchor['control'])]
                reason = ('no_candidate_within_memory' if not feasible else
                    'no_useful_candidate' if not useful else
                    'no_useful_candidate_within_fixed_quality_guard' if phase == 'qualified' and not ready else None)
                if reason is not None:
                    assert attempt['action'] == 'stopped' and attempt['reason'] == reason
                    assert index == len(path['attempts'])-1
                    continue
                winner = min(ready or useful, key=lambda c: (c['reconstruction_mse'], -c['quality']['f1_macro'],
                    c['quality']['loss'], c['cost_bytes'], c['ordinal'], c['uid']))
                assert attempt['selected_uid'] == winner['uid']
                qualifies = passed(winner['quality'], anchor['control'])
                phase = 'qualified' if qualifies else 'seeking_quality'
                current_raw = winner['raw_nll']
                assert attempt['after']['phase'] == phase
                assert attempt['after']['anchor'] == before['anchor']
                assert attempt['after']['budget_bytes'] == before['budget_bytes']
                if before['phase'] == 'qualified':
                    assert qualifies
                snapshot = attempt['snapshot']; snapshots.append(snapshot)
                assert snapshot['control_guard'] == qualifies
                if attempt['original_action'] != 'rejected':
                    old = source['metas'][path['path']+f"/b{attempt['block']:02d}"]
                    actual_compatible = same_raw(load_protected(snapshot['artifact']), load_protected(old['artifact']))
                    assert attempt['raw_path_compatible'] == actual_compatible
                else:
                    assert not attempt['raw_path_compatible']
                if not attempt['raw_path_compatible']:
                    assert index == len(path['attempts'])-1 and path['censored']
            for snapshot in snapshots:
                assert digest(snapshot['artifact']) == snapshot['sha256']
                model = load_protected(snapshot['artifact']); price = physical_cost(model)
                assert price == snapshot['cost_bytes'] <= source['budget_bytes']
                np.testing.assert_array_equal(model.decision_shifts, snapshot['policy']['shifts'])
                z = model.raw(x); cp = snapshot['policy']
                for yy, zz, expected in ((yc, z[nc:nc+nt], snapshot['control']), (ys, z[nc+nt:], snapshot['selection'])):
                    q = independent_task_quality(yy, zz, cp, task)
                    maximum = max(maximum, *(abs(q[k]-expected[k]) for k in expected))
                raw = independent_task_quality(yc, z[nc:nc+nt],
                    dict(temperature=1., shifts=np.zeros(z.shape[1])), task)
                assert abs(raw['loss']-snapshot['raw_nll']) < 2e-8
                assert snapshot['selection_guard'] == passed(snapshot['selection'], anchor['selection'])
            assert path['final_observed'] == final_choice(path['qualified_snapshots'])
            for c in path['qualified_snapshots']:
                assert c['control_guard'] and c['cost_bytes'] <= source['budget_bytes']
            assert path['censored'] == path['terminal'].startswith('censored')
        assert maximum < 2e-8
        checks.append(dict(case=result['case'], status='passed', paths=len(result['paths']),
            maximum_metric_difference=maximum, conditional_steps=len(result['one_step'])))
    return dict(status='passed', cases=checks, new_tree_fits=0, gpu_execution=False,
        test_labels_used=False, checks='exact frozen full UID/policy; independent decisions and phase transitions; '
            'raw-NLL usefulness; stored real models and physical bytes; exact raw-path continuation; final selection')
