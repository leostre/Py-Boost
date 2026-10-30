"""Independent checks of geometry bounds, error algebra and frozen gates."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from budget_quality_worker import locked, digest, native_load
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected
from audit_abl_transfer import independent_task_quality
from experiments.shared_leaf_basis import reference_class_model


def independent_guard(q, r):
    return (q['loss'] <= r['loss']*1.05 and q['f1_macro'] >= r['f1_macro']*.95 and
        q['f1_weighted'] >= r['f1_weighted']*.95)


def audit(entries, cfg):
    checks = []
    for entry in entries:
        result = locked(entry['path'], entry['sha256'])
        source = locked(result['input']['path'], result['input']['sha256'])
        data = data_before_test(source['record'])
        start = sum(data['sizes'][:2])
        x, y = data['dev_x'][start:], data['dev_y'][start:]
        task = result['task']
        # Contrast error for softmax; sigmoid has no unidentifiable common shift.
        def error(a):
            return a-a.mean(1, keepdims=True) if task == 'multiclass' else a
        grouped = {}
        for row in result['basis_comparisons']:
            grouped.setdefault((row['path'], row['block']), []).append(row)
            for candidate in row['candidates']:
                assert candidate['control_guard']['passed'] == independent_guard(
                    candidate['validation'], row['reference_control'])
                assert candidate['selection_guard']['passed'] == independent_guard(
                    candidate['selection'], row['reference_selection'])
            feasible = [r for r in row['candidates'] if r['cost_bytes'] <= source['budget_bytes']]
            eligible = [r for r in feasible if independent_guard(r['validation'], row['reference_control'])]
            expected = 'memory' if not feasible else ('accepted' if eligible else 'local_quality')
            assert row['stop'] == expected
            if row['chosen'] is not None:
                winner = min(eligible, key=lambda r: (r['train_weighted_mse'], -r['validation']['f1_macro'],
                    r['validation']['loss'], r['cost_bytes'], r['percent']))
                assert row['chosen']['index'] == row['candidates'].index(winner)
        for key, rows in grouped.items():
            assert len(rows) == 4
            assert len({r['pure']['cost_bytes'] for r in rows}) == 1
            pca = next(r for r in rows if r['basis'] == 'functional')
            for row in rows:
                assert pca['increment_capture'] >= row['increment_capture']-3e-6
                assert abs(row['pca_retained']-pca['increment_capture']) < 3e-6
        warm = locked(source['warmup']['path'], source['warmup']['sha256'])
        warm_model = load_protected(warm['models'][0]['artifact'])
        worst_identity, worst_metrics = 0., 0.
        for ref in source['trajectories']:
            if ref['mode'] == 'full':
                continue
            trajectory = locked(ref['path'], ref['sha256'])
            shadow, actual = warm_model.raw(x), warm_model.raw(x)
            sum_local = 0.
            for cycle in trajectory['cycles']:
                if cycle['action'] == 'rejected':
                    break
                inc = native_load(cycle['fit']['artifact'])
                if task == 'multiclass':
                    inc = replace(inc, base=reference_class_model(inc.base))
                increment = inc.prepare_inference().raw(x)
                previous_error = error(actual-shadow)
                shadow = shadow+increment
                teacher = actual+increment
                item = next(m for m in trajectory['models'] if m['block'] == cycle['block'])
                actual = load_protected(item['artifact']).raw(x)
                local, total = error(actual-teacher), error(actual-shadow)
                sum_local += float(np.mean(local*local))
                measured = next(r for r in result['accumulation'] if r['path'] == ref['mode'] and r['block'] == cycle['block'])
                r = measured['roles']['selection']
                worst_identity = max(worst_identity, abs(r['cumulative_mse']-float(np.mean(total*total))),
                    abs(r['sum_local_mse']-sum_local),
                    abs(r['cross_term']-float(2*np.mean(previous_error*local))))
                cp = dict(temperature=cycle['projection']['chosen']['temperature'],
                    shifts=cycle['projection']['chosen']['decision_shifts'])
                for values, key in ((actual, 'selection_actual'), (shadow, 'selection_shadow')):
                    q = independent_task_quality(y, values, cp, task)
                    worst_metrics = max(worst_metrics, *(abs(q[k]-measured[key][k]) for k in measured[key]))
                if worst_identity > 2e-8 or worst_metrics > 2e-8:
                    raise RuntimeError('independent accumulated-error replay failed')
        checks.append(dict(case=result['case'], status='passed', paired_blocks=len(grouped),
            maximum_error_identity_difference=worst_identity, maximum_metric_difference=worst_metrics))
    return dict(status='passed', cases=checks, new_tree_fits=0, gpu_execution=False,
        test_labels_used=False, checks='equal pure-basis bytes; PCA bound; independent guards/ranking; '
            'independent prepared-inference dense shadow and accumulated cross terms; sklearn/sigmoid metrics')
