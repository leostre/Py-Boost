"""Independent scipy/sklearn replay; does not call the gate implementation."""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import f1_score


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(report_path, check_sources=True):
    report_path = Path(report_path); report = json.loads(report_path.read_text(encoding='utf-8'))
    root = report_path.parent; remote = str(Path(report['protocol']['path']).parent).replace('\\', '/')
    checks = dict(metrics=0, controls=0, decisions=0, bootstrap_explicit_draws=0, model_arrays=0, folds=0)
    maximum = 0.
    def resolve(path):
        text = str(path).replace('\\', '/')
        return root/Path(text[len(remote)+1:]) if text.startswith(remote+'/') else Path(path)
    def data(reference):
        p = resolve(reference['path']); assert sha(p) == reference['sha256'], str(p)
        return json.loads(p.read_text(encoding='utf-8'))
    def arrays(reference):
        p = resolve(reference['path']); assert sha(p) == reference['sha256'], str(p)
        with np.load(p, allow_pickle=False) as f: return {k:f[k].copy() for k in f.files}
    def evaluate(y, raw, task, control):
        if task == 'multilabel':
            assert control['temperature'] == 1.
            p = np.clip(expit(raw), 1e-7, 1-1e-7); pred = raw-np.asarray(control['shifts']) >= 0
            loss = np.mean(-y*np.log(p)-(1-y)*np.log1p(-p), axis=1); brier = np.mean((p-y)**2, axis=1)
        else:
            p = np.clip(softmax(raw/control['temperature'], axis=1), 1e-7, 1-1e-7)
            pred = (raw/control['temperature']-control['shifts']).argmax(axis=1)
            loss = -np.log(p[np.arange(len(y)), y]); brier = np.sum(p*p, axis=1)-2*p[np.arange(len(y)), y]+1
        return pred, loss, brier
    def q(y, parts, task, c):
        pred, loss, brier = parts; args = {'labels':np.arange(c)} if task == 'multiclass' else {}
        return dict(f1_macro=float(f1_score(y, pred, average='macro', zero_division=0, **args)),
            f1_weighted=float(f1_score(y, pred, average='weighted', zero_division=0, **args)),
            loss=float(np.mean(loss)), brier=float(np.mean(brier)))
    def compare(actual, expected):
        nonlocal maximum
        for k in actual:
            error = abs(actual[k]-expected[k]); maximum = max(maximum, error); assert error < 2e-12, (k, actual, expected)
        checks['metrics'] += 1
    def gate_check(a, cache):
        scores = a['scores']; ff = cache['boot_f1']; ll = cache['boot_loss']; n = len(scores)
        f = [s['f1_macro'] for s in scores]; l = [s['loss'] for s in scores]
        df = ff-ff[:, [0]]; dl = ll[:, [0]]-ll
        sf = np.std(df, axis=0, ddof=1); sl = np.std(dl, axis=0, ddof=1)
        eligible = [i for i in range(1, n) if f[i]-f[0] > sf[i] and l[0]-l[i] >= -sl[i]]
        best = sorted(eligible, key=lambda i:(-f[i], l[i], i))[0] if eligible else 0
        se = np.std(ff[:, [best]]-ff, axis=0, ddof=1)
        selected = min([i for i in [0]+eligible if f[best]-f[i] <= se[i]+1e-14]) if eligible else 0
        assert a['decision']['index'] == selected and a['decision']['best_index'] == best
        assert a['decision']['alpha'] == [0., .125, .25, .5, 1.][selected]
        for i, r in enumerate(a['decision']['candidates']):
            assert r['benefit_passed'] == bool(f[i]-f[0] > sf[i])
            assert r['nll_guard_passed'] == bool(l[0]-l[i] >= -sl[i])
        checks['decisions'] += 1
    def assessment_check(a, raws, y, task, available, forbidden):
        cache = arrays(a['arrays']); n, c = len(y), raws.shape[-1]; names = list(a['all_scores'][0]); held = []
        assert set(a['fitting_rows']) == set(available) and not set(available) & set(forbidden)
        reconstructed = {}
        for i in range(5):
            reconstructed[i] = {}
            for name in names:
                pred = np.empty_like(cache['prediction'][i]); loss = np.empty(n); brier = np.empty(n)
                for state in a['states']:
                    fit, val = state['fitting'], state['validation']
                    assert not set(fit) & set(val) and set(fit+val) == set(range(n))
                    assert not set(np.asarray(available)[fit]) & set(forbidden)
                    ctrl = state['controls'][i][name]
                    if a['mode'] == 'frozen_base': assert ctrl == state['controls'][0][name]
                    p, l, b = evaluate(y[val], raws[i, val], task, ctrl); pred[val] = p; loss[val] = l; brier[val] = b
                    checks['controls'] += 1
                score = q(y, (pred, loss, brier), task, c); compare(score, a['all_scores'][i][name])
                reconstructed[i][name] = (pred, loss, brier)
            ranking = a['all_scores'][0 if a['mode'] == 'frozen_base' else i]
            expected = min(names, key=lambda name: ((ranking[name]['loss'],) if task == 'multiclass' else
                (-ranking[name]['f1_macro'], ranking[name]['loss'])) + (int(name != 'raw'), name))
            assert expected == a['policies'][i]
            p, l, b = reconstructed[i][expected]
            np.testing.assert_array_equal(p, cache['prediction'][i]); np.testing.assert_allclose(l, cache['loss'][i], atol=2e-14)
            np.testing.assert_allclose(b, cache['brier'][i], atol=2e-14)
            compare(q(y, (p, l, b), task, c), a['scores'][i])
        for state in a['states']: held += state['validation']
        assert sorted(held) == list(range(n)); checks['folds'] += len(a['states'])
        assert cache['boot_f1'].shape == (500, 5) and cache['boot_loss'].shape == (500, 5)
        # Explicit original-row replay on first/last draws catches axis and pairing defects.
        counts = np.random.default_rng(a['bootstrap_seed']).multinomial(n, np.full(n, 1/n), size=500)
        for draw in (0, 1, 2, 497, 498, 499):
            ids = np.repeat(np.arange(n), counts[draw])
            for i in range(5):
                args = {'labels':np.arange(c)} if task == 'multiclass' else {}
                f = f1_score(y[ids], cache['prediction'][i, ids], average='macro', zero_division=0, **args)
                assert abs(f-cache['boot_f1'][draw, i]) < 2e-12
                assert abs(cache['loss'][i, ids].mean()-cache['boot_loss'][draw, i]) < 2e-12
                checks['bootstrap_explicit_draws'] += 1
        gate_check(a, cache)
    protocol = data(report['protocol']); seal = data(report['seal'])
    assert protocol['tree_training'] is False and protocol['coefficient_refit'] is False and protocol['alphas'] == [0, .125, .25, .5, 1]
    assert report['new_trees'] == report['coefficient_refits'] == 0 and seal['test_labels_used'] is False
    assert seal['cases'] == report['cases']
    test_entries = {e['case']:e for e in report['tests']}
    for entry in report['cases']:
        case = data(entry); common = arrays(case['development']); task = case['task']; y = common['y']; c = common['raw_base'].shape[1]; n = len(y)
        assert not set(common['train_rows']) & set(common['original_rows'])
        for family in case['families']:
            raw = arrays(family['raw']); raws = np.stack([common['raw_base']+a*(raw['raw']-common['raw_base']) for a in protocol['alphas']])
            # Preserve endpoint values bit-for-bit, as specified by the source contract.
            raws[0] = common['raw_base']; raws[-1] = raw['raw']
            with np.load(resolve(family['models'][-1]['artifact']), allow_pickle=False) as f: original = {k:f[k].copy() for k in f.files}
            with np.load(resolve(family['models'][0]['artifact']), allow_pickle=False) as f: base = {k:f[k].copy() for k in f.files}
            for i, model in enumerate(family['models']):
                p = resolve(model['artifact']); assert sha(p) == model['sha256']
                assert model['cost_bytes'] <= family['source']['byte_ceiling']
                if i > 0: assert model['cost_bytes'] == family['models'][-1]['cost_bytes']
                with np.load(p, allow_pickle=False) as f:
                    for k in f.files:
                        if k.endswith('_values'):
                            np.testing.assert_array_equal(f[k], original[k]*model['alpha'] if i else base[k])
                        elif k.endswith('_rows') or k.endswith('_columns'):
                            np.testing.assert_array_equal(f[k], original[k] if i else base[k])
                        else: np.testing.assert_array_equal(f[k], base[k])
                        checks['model_arrays'] += 1
            if check_sources:
                assert sha(family['source']['artifact']) == family['source']['sha256']
                assert sha(case['baseline']['artifact']) == case['baseline']['sha256']
            for mode in family['modes']:
                cache = arrays(mode['outer_arrays']); held = []
                for state in mode['states']:
                    a, b = np.asarray(state['fitting']), np.asarray(state['validation']); held += b.tolist()
                    assert not set(a) & set(b) and set(a)|set(b) == set(range(n))
                    assert common['original_rows'][a].tolist() == state['fitting_original_rows']
                    assert common['original_rows'][b].tolist() == state['validation_original_rows']
                    assessment_check(state['inner'], raws[:, a], y[a], task, a, b)
                    assert np.all(cache['selected'][b] == state['inner']['decision']['index'])
                    for i in range(5):
                        pp, ll, bb = evaluate(y[b], raws[i, b], task, state['controls'][i])
                        np.testing.assert_array_equal(pp, cache['prediction'][i, b]); np.testing.assert_allclose(ll, cache['loss'][i, b], atol=2e-14)
                        np.testing.assert_allclose(bb, cache['brier'][i, b], atol=2e-14)
                assert sorted(held) == list(range(n)); checks['folds'] += 3
                for i in range(5): compare(q(y, (cache['prediction'][i], cache['loss'][i], cache['brier'][i]), task, c), mode['outer_curves'][i]['quality'])
                sel = cache['selected']; ids = np.arange(n)
                compare(q(y, (cache['prediction'][sel, ids], cache['loss'][sel, ids], cache['brier'][sel, ids]), task, c), mode['outer_gate']['quality'])
                assessment_check(mode['final'], raws, y, task, np.arange(n), [])
        test = data(test_entries[case['case']]); yt = arrays(test['targets'])['y']
        assert not set(arrays(test['targets'])['original_rows']) & (set(common['train_rows']) | set(common['original_rows']))
        for family in test['families']:
            raw = arrays(family['raw']); raws = np.stack([raw['raw_base']+a*(raw['raw']-raw['raw_base']) for a in protocol['alphas']]); raws[-1] = raw['raw']
            source_family = next(f for f in case['families'] if f['family'] == family['family'])
            for mode in family['modes']:
                selected = next(m for m in source_family['modes'] if m['mode'] == mode['mode'])
                assert mode['final_decision'] == selected['final']['decision']
                assert mode['controls'] == selected['final']['controls']
                cache = arrays(mode['arrays'])
                for i, control in enumerate(mode['controls']):
                    parts = evaluate(yt, raws[i], task, control)
                    np.testing.assert_array_equal(parts[0], cache['prediction'][i]); np.testing.assert_allclose(parts[1], cache['loss'][i], atol=2e-14)
                    np.testing.assert_allclose(parts[2], cache['brier'][i], atol=2e-14)
                    compare(q(yt, parts, task, c), mode['curves'][i]['quality'])
                assert mode['gate'] == mode['curves'][selected['final']['decision']['index']]
    return dict(status='passed', checks=checks, maximum_metric_error=maximum, cases=len(report['cases']),
        new_trees=0, source_files_checked=check_sources,
        scope='all nested/test metrics and decisions; all500 stored draws used for decisions,6 explicitly resampled draws/assessment; model arrays/support/scale/budget and fold separation')
