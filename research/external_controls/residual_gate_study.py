"""R4: saved-model scalar attenuation, honest nested D and exploratory tests."""
from pathlib import Path
from dataclasses import asdict, replace
from time import perf_counter
import json, sys, gc
import numpy as np
from scipy import sparse
from scipy.special import expit, softmax, logsumexp
from budget_quality_worker import digest, locked, write, event
from abl_quality_worker import check_native_environment, memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected, block_cost
from unified_frontier_core import make_folds
from experiments.sparse_leaf_core import leaf_design
from residual_gate_core import (ALPHAS, CalibrationMode, blend_raw, sufficient, quality,
    fit_policies, assess, label_matrix)


def ref(path):
    return dict(path=str(path), sha256=digest(path))


def save_arrays(path, **arrays):
    np.savez_compressed(path, **arrays)
    return ref(path)


def checked_raw(meta):
    r = meta['raw']; assert digest(r['path']) == r['sha256']
    with np.load(r['path'], allow_pickle=False) as f:
        return f['raw'].copy() if 'raw' in f else np.concatenate([f[k] for k in ('calibration', 'control', 'selection')])


def folded(y, task, seed, count):
    return [(f.fitting, f.validation) for f in make_folds(y, task, seed, count)]


def assessment(raws, y, task, mode, cfg, seed, path, fit_rows):
    folds = folded(y, task, seed, 2)
    a = assess(raws, y, task, folds, mode, cfg['decision_grid'], cfg['decision_shrinkage'], seed+50009, 500)
    cache = save_arrays(path.with_suffix('.npz'), prediction=a['prediction'], loss=a['loss'], brier=a['brier'],
                        boot_f1=a['boot_f1'], boot_loss=a['boot_loss'])
    result = dict(mode=mode.value, seed=seed, bootstrap_seed=seed+50009, fitting_rows=list(map(int, fit_rows)),
        policies=a['policies'], scores=a['scores'], all_scores=a['all_scores'], states=a['states'],
        decision=asdict(a['decision']), arrays=cache)
    write(path, result)
    return result


def refit_controls(raws, y, task, mode, policies, cfg):
    controls = [fit_policies(raws[0], y, task, cfg['decision_grid'], cfg['decision_shrinkage'])]
    for i in range(1, len(ALPHAS)):
        controls.append(controls[0] if mode is CalibrationMode.FROZEN_BASE else
                        fit_policies(raws[i], y, task, cfg['decision_grid'], cfg['decision_shrinkage']))
    return [p[name] for p, name in zip(controls, policies)]


def raw_quality(y, raw, task):
    p = dict(temperature=1., shifts=[0.]*raw.shape[1])
    return quality(y, *sufficient(y, raw, task, p), task, raw.shape[1])


def probability(raw, task, temperature=1.):
    return np.clip(expit(raw) if task == 'multilabel' else softmax(raw/np.asarray(temperature)[..., None], axis=1), 1e-7, 1-1e-7)


def per_label_comparison(y, pred0, pred1, prob0, prob1, task):
    c = prob0.shape[1]; truth = label_matrix(y, task, c)
    p0 = label_matrix(pred0, task, c); p1 = label_matrix(pred1, task, c)
    target = truth.astype(float)
    nll0 = -target*np.log(prob0)-(1-target)*np.log1p(-prob0)
    nll1 = -target*np.log(prob1)-(1-target)*np.log1p(-prob1)
    rows = []
    for j in range(c):
        def parts(p):
            tp = int(np.sum(p & truth[:, j])); fp = int(np.sum(p & ~truth[:, j])); fn = int(np.sum(~p & truth[:, j]))
            return dict(tp=tp, fp=fp, fn=fn, f1=2*tp/max(2*tp+fp+fn, 1), precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1))
        a, b = parts(p0[:, j]), parts(p1[:, j])
        rows.append(dict(output=j, positive=int(truth[:, j].sum()), baseline=a, corrected=b, f1_change=b['f1']-a['f1'],
            nll_base=float(nll0[:, j].mean()), nll_corrected=float(nll1[:, j].mean()),
            repaired=int(np.sum((p0[:, j] != truth[:, j]) & (p1[:, j] == truth[:, j]))),
            damaged=int(np.sum((p0[:, j] == truth[:, j]) & (p1[:, j] != truth[:, j]))),
            interpretation='binary output NLL' if task == 'multilabel' else 'one-vs-rest diagnostic, not categorical NLL decomposition'))
    return rows


def scaled_model(baseline, corrected, alpha):
    if alpha == 0: return baseline
    return replace(corrected, blocks=tuple(replace(b, values=(b.values*alpha).astype(np.float32)) for b in corrected.blocks))


def assert_same_base(baseline, corrected):
    assert len(baseline.blocks) == len(corrected.blocks)
    for a, b in zip(baseline.blocks, corrected.blocks):
        for name in ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias'):
            np.testing.assert_array_equal(getattr(a.base, name), getattr(b.base, name))
    np.testing.assert_array_equal(baseline.bias, corrected.bias)
    assert sum(len(b.values) for b in baseline.blocks) == 0


def geometry(baseline, corrected, data, raw_train, corrected_train, raw_dev, corrected_dev, out):
    block = next(b for b in corrected.blocks if len(b.values)); c = len(corrected.bias)
    leaves = block.base.leaf_indices(data['train_x']); design = leaf_design(leaves, len(block.base.coefficients))
    counts = np.asarray(design.sum(axis=0)).ravel()
    coefficient_counts = np.bincount(block.columns, minlength=c)
    squared = sparse.coo_matrix((block.values.astype(float)**2, (block.rows, block.columns)), shape=(len(counts), c)).tocsr()
    individual_energy = np.asarray((design @ squared).mean(axis=0)).ravel()
    presence = sparse.coo_matrix((np.ones(len(block.values)), (block.rows, block.columns)), shape=squared.shape).tocsr()
    active = (design @ presence).toarray()
    delta = corrected_train-raw_train; delta_dev = corrected_dev-raw_dev
    function_energy = np.mean(delta**2, axis=0)
    per_label = []
    truth = label_matrix(data['train_y'], baseline.task, c)
    truth_dev = label_matrix(data['dev_y'], baseline.task, c)
    for j in range(c):
        use = block.columns == j; support = counts[block.rows[use]]
        per_label.append(dict(output=j, train_positive=int(truth[:, j].sum()), development_positive=int(truth_dev[:, j].sum()),
            coefficients=int(coefficient_counts[j]), median_leaf_support=float(np.median(support)) if len(support) else None,
            minimum_leaf_support=int(support.min()) if len(support) else None,
            correction_rms_train=float(np.sqrt(function_energy[j])), correction_rms_development=float(np.sqrt(np.mean(delta_dev[:, j]**2))),
            concurrent_corrections_mean=float(active[:, j].mean()), concurrent_corrections_max=int(active[:, j].max()),
            coherent_energy_ratio=float(function_energy[j]/individual_energy[j]) if individual_energy[j] > 0 else None))
    curves = []
    for alpha in ALPHAS:
        curves.append(dict(alpha=alpha, train=raw_quality(data['train_y'], blend_raw(raw_train, corrected_train, alpha), baseline.task),
                           development=raw_quality(data['dev_y'], blend_raw(raw_dev, corrected_dev, alpha), baseline.task)))
    zero = dict(temperature=1., shifts=[0.]*c)
    return dict(per_label=per_label, uncalibrated_curves=curves,
        per_label_train_raw=label_diagnostics(data['train_y'], raw_train, corrected_train, baseline.task, zero, zero),
        support_scope='train only; descriptions do not select alpha')


def nested_case(raws, y, task, mode, cfg, seed, out, original_rows, costs):
    n = len(y); c = raws.shape[-1]; outer = folded(y, task, seed, 3)
    pred_shape = (len(ALPHAS), n, c) if task == 'multilabel' else (len(ALPHAS), n)
    prediction = np.empty(pred_shape, bool if task == 'multilabel' else int)
    loss = np.empty((len(ALPHAS), n)); brier = np.empty_like(loss)
    selected = np.empty(n, int); states = []
    for fold, (fitting, validation) in enumerate(outer):
        a, b = np.asarray(fitting, int), np.asarray(validation, int)
        inner = assessment(raws[:, a], y[a], task, mode, cfg, seed+101+fold*1009,
                           out/f'outer-{fold}-inner.json', a)
        controls = refit_controls(raws[:, a], y[a], task, mode, inner['policies'], cfg)
        for i in range(len(ALPHAS)):
            prediction[i, b], loss[i, b], brier[i, b] = sufficient(y[b], raws[i, b], task, controls[i])
        selected[b] = inner['decision']['index']
        states.append(dict(fitting=a.tolist(), validation=b.tolist(), fitting_original_rows=original_rows[a].tolist(),
            validation_original_rows=original_rows[b].tolist(), inner=inner, controls=controls))
        event('residual_gate_outer', mode=mode.value, fold=fold, alpha=inner['decision']['alpha'],
              reason=inner['decision']['reason'], n_fit=len(a), n_validation=len(b))
    ar = save_arrays(out/'outer.npz', prediction=prediction, loss=loss, brier=brier, selected=selected)
    curves = [dict(alpha=alpha, cost_bytes=costs[i], quality=quality(y, prediction[i], loss[i], brier[i], task, c),
        uncalibrated=raw_quality(y, raws[i], task)) for i, alpha in enumerate(ALPHAS)]
    ids = np.arange(n)
    gate_quality = quality(y, prediction[selected, ids], loss[selected, ids], brier[selected, ids], task, c)
    temperatures = np.ones((len(ALPHAS), n))
    for state in states:
        for i, control in enumerate(state['controls']): temperatures[i, state['validation']] = control['temperature']
    prob0 = probability(raws[0], task, temperatures[0]); prob1 = probability(raws[-1], task, temperatures[-1])
    prob_gate = probability(raws[selected, ids], task, temperatures[selected, ids])
    final = assessment(raws, y, task, mode, cfg, seed+70001, out/'full-D-inner.json', np.arange(n))
    final['controls'] = refit_controls(raws, y, task, mode, final['policies'], cfg)
    chosen = final['decision']['index']
    result = dict(mode=mode.value, states=states, outer_arrays=ar, outer_curves=curves,
        outer_gate=dict(quality=gate_quality, selected_row_counts={str(a):int(np.sum(selected == i)) for i, a in enumerate(ALPHAS)},
                        fold_alphas=[s['inner']['decision']['alpha'] for s in states],
                        mean_cost_bytes=float(np.mean(np.asarray(costs)[selected]))),
        per_label_outer_alpha1=per_label_comparison(y, prediction[0], prediction[-1], prob0, prob1, task),
        per_label_outer_gate=per_label_comparison(y, prediction[0], prediction[selected, ids], prob0, prob_gate, task),
        final=final, final_cost_bytes=costs[chosen])
    write(out/'mode.json', result)
    return result


def before_test_case(cfg, meta, old_meta, out):
    case = locked(meta['path'], meta['sha256']); original = locked(old_meta['path'], old_meta['sha256'])
    cid = case['case']; task = case['task']; record = case['record']; data = data_before_test(record)
    dev_ids = np.concatenate([data['roles'][k] for k in ('calibration', 'control', 'selection')])
    baseline_meta = next(m for m in original['models'] if m['uid'] == 'baseline')
    assert digest(baseline_meta['artifact']) == baseline_meta['sha256']
    baseline = load_protected(baseline_meta['artifact']); base_dev = checked_raw(baseline_meta)
    base_train = baseline.raw(data['train_x']); probes = np.linspace(0, len(base_dev)-1, min(17, len(base_dev))).astype(int)
    np.testing.assert_allclose(base_dev[probes], baseline.raw(data['dev_x'][probes]), atol=2e-5)
    out.mkdir(parents=True, exist_ok=True)
    common = save_arrays(out/'development.npz', y=data['dev_y'], y_train=data['train_y'], raw_base=base_dev,
        raw_base_train=base_train, original_rows=dev_ids, train_rows=data['roles']['train'],
        probe_x=data['dev_x'][probes], probe_rows=probes)
    families = []
    for family in ('carry', 'balanced'):
        source = next(m for m in case['models'] if m['family'] == family)
        assert source['fraction'] == .02 and source['window'] == 40 and source['fit']['all_converged']
        assert digest(source['artifact']) == source['sha256']
        corrected = load_protected(source['artifact']); assert_same_base(baseline, corrected)
        corrected_dev = checked_raw(source); corrected_train = corrected.raw(data['train_x'])
        np.testing.assert_allclose(corrected_dev[probes], corrected.raw(data['dev_x'][probes]), atol=2e-5)
        directory = out/family; directory.mkdir(exist_ok=True)
        raw_cache = save_arrays(directory/'raw.npz', raw=corrected_dev, raw_train=corrected_train)
        models = []; costs = []
        for alpha in ALPHAS:
            candidate = scaled_model(baseline, corrected, alpha); price = block_cost(candidate)
            assert price['cost_bytes'] <= source['byte_ceiling']
            np.testing.assert_allclose(candidate.raw(data['dev_x'][probes]), blend_raw(base_dev[probes], corrected_dev[probes], alpha), atol=2e-5)
            model_file = directory/f'alpha-{alpha:g}.npz'; candidate.save(model_file)
            models.append(dict(alpha=alpha, artifact=str(model_file), sha256=digest(model_file), **price))
            costs.append(price['cost_bytes'])
        raws = np.stack([blend_raw(base_dev, corrected_dev, a) for a in ALPHAS])
        modes = []
        for mode in CalibrationMode:
            path = directory/mode.value; path.mkdir(exist_ok=True)
            modes.append(nested_case(raws, data['dev_y'], task, mode, cfg, record['case']['seed']+20261029,
                                      path, dev_ids, costs))
        diagnosis = geometry(baseline, corrected, data, base_train, corrected_train, base_dev, corrected_dev, directory)
        families.append(dict(family=family, source=source, raw=raw_cache, models=models, modes=modes, geometry=diagnosis))
        event('residual_gate_family_D', case=cid, family=family,
            final_alphas={m['mode']:m['final']['decision']['alpha'] for m in modes},
            nested_macroF1={m['mode']:m['outer_gate']['quality']['f1_macro'] for m in modes})
    result = dict(case=cid, task=task, record=record, source_case=meta, baseline=baseline_meta, development=common,
        families=families, new_trees=0, coefficient_refits=0, test_labels_used_for_selection=False,
        labels=record.get('source_label_domain', [str(i) for i in range(record['case']['outputs'])]))
    write(out/'development.json', result)
    return ref(out/'development.json') | dict(case=cid)


def label_diagnostics(y, raw_base, raw_corrected, task, control_base, control_corrected):
    pa, la, ba = sufficient(y, raw_base, task, control_base)
    pb, lb, bb = sufficient(y, raw_corrected, task, control_corrected)
    return per_label_comparison(y, pa, pb, probability(raw_base, task, control_base['temperature']),
                                probability(raw_corrected, task, control_corrected['temperature']), task)


def after_seal_case(entry, seal, out, historical_points):
    assert digest(seal['path']) == seal['sha256']
    case = locked(entry['path'], entry['sha256']); record = case['record']; task = case['task']; c = record['case']['outputs']
    with np.load(record['case']['split_file'], allow_pickle=False) as f: ids = f['test'].copy()
    with np.load(record['dataset']['path'], allow_pickle=False) as f: y = f['target'][ids].copy(); x = f['features'][ids].astype(np.float32)
    y_ref = save_arrays(out/'test-targets.npz', y=y, original_rows=ids)
    base = load_protected(case['baseline']['artifact']); raw_base = base.raw(x)
    families = []
    for family in case['families']:
        model = load_protected(family['source']['artifact']); raw_corrected = model.raw(x)
        raws = np.stack([blend_raw(raw_base, raw_corrected, a) for a in ALPHAS])
        path = out/family['family']; raw_ref = save_arrays(path/'test-raw.npz', raw_base=raw_base, raw=raw_corrected)
        modes = []
        for mode in family['modes']:
            controls = mode['final']['controls']; parts = [sufficient(y, z, task, p) for z, p in zip(raws, controls)]
            predictions, losses, briers = [np.stack([v[j] for v in parts]) for j in range(3)]
            cache = save_arrays(path/mode['mode']/'test.npz', prediction=predictions, loss=losses, brier=briers)
            curves = [dict(alpha=a, cost_bytes=family['models'][i]['cost_bytes'],
                quality=quality(y, predictions[i], losses[i], briers[i], task, c), uncalibrated=raw_quality(y, raws[i], task)) for i, a in enumerate(ALPHAS)]
            index = mode['final']['decision']['index']
            diag = label_diagnostics(y, raw_base, raw_corrected, task, controls[0], controls[-1])
            modes.append(dict(mode=mode['mode'], curves=curves, gate=curves[index], final_decision=mode['final']['decision'],
                controls=controls, arrays=cache, per_label_alpha1=diag,
                per_label_gate=label_diagnostics(y, raw_base, raws[index], task, controls[0], controls[index])))
            event('residual_gate_test', case=case['case'], family=family['family'], mode=mode['mode'],
                alpha=ALPHAS[index], basis=curves[0]['quality'], old_full_correction=curves[-1]['quality'],
                gate=curves[index]['quality'], bytes=curves[index]['cost_bytes'])
        families.append(dict(family=family['family'], raw=raw_ref, modes=modes))
    secondary = []
    if case['case'].startswith('emotions_'):
        old_base = next(p for p in historical_points if p['family'] == 'floor')
        for point in historical_points:
            if point['family'] in ('floor', 'rank', 'carry', 'balanced'): continue
            with np.load(point['raw']['path'], allow_pickle=False) as f: z = f['raw'].copy()
            assert digest(point['raw']['path']) == point['raw']['sha256']
            secondary.append(dict(uid=point['uid'], family=point['family'], quality=point['quality'],
                per_label=label_diagnostics(y, raw_base, z, task, old_base['control'], point['control']),
                calibration_scope='historical precision-stage controls; diagnostic only, not a new gate candidate'))
    result = dict(case=case['case'], task=task, targets=y_ref, development=entry, families=families, seal=seal,
                  secondary_historical=secondary,
                  interpretation='former official test, exploratory; not used for any current decisions')
    write(out/'test-report.json', result)
    return ref(out/'test-report.json') | dict(case=case['case'])


def run(cfg, out):
    out.mkdir(parents=True, exist_ok=True); started = perf_counter(); spec = cfg['residual_gate']
    assert spec['alphas'] == list(ALPHAS) and spec['bootstrap_repeats'] == 500 and spec['new_trees'] == 0
    environment = check_native_environment(cfg)
    precision = locked(spec['source']['path'], spec['source']['sha256'])
    coarse = locked(cfg['reserve_precision']['source']['path'], cfg['reserve_precision']['source']['sha256'])
    assert precision['status'] == 'reserve_precision_complete'
    protocol = dict(**spec, rule_source_sha256=digest(Path(__file__).with_name('residual_gate_core.py')),
        study_source_sha256=digest(__file__), command='python3 -u external_runner.py', tree_training=False,
        coefficient_refit=False, new_data=False, gpu_execution=False, calibration_grid=cfg['decision_grid'],
        calibration_shrinkages=cfg['decision_shrinkage'], outer_folds=3, inner_folds=2,
        admission='delta_macroF1 > paired_SE and delta_NLL <= paired_SE; prefer least alpha within paired_SE of best admitted F1',
        null='alpha0 always available; lambda/basis/support/trees unchanged',
        multiclass='same macroF1 gate; old NLL calibration policy objective preserved; report raw and calibrated NLL')
    write(out/'protocol_seal.json', protocol); event('residual_gate_protocol', **protocol)
    entries = []
    for cid in spec['cases']:
        meta = next(c for c in precision['cases'] if c['case'] == cid)
        original = next(c for c in coarse['cases'] if c['case'] == cid)
        entries.append(before_test_case(cfg, meta, original, out/cid))
        write(out/'progress.json', dict(cases=entries, new_trees=0, test_evaluated=False)); gc.collect(); memory_guard()
    write(out/'pretest_seal.json', dict(cases=entries, protocol=ref(out/'protocol_seal.json'), test_labels_used=False))
    seal = ref(out/'pretest_seal.json'); tests = []
    for entry in entries:
        historical_points = [p for p in precision['points'] if p['case'] == entry['case']]
        tests.append(after_seal_case(entry, seal, out/entry['case'], historical_points)); gc.collect(); memory_guard()
    report = dict(status='residual_gate_complete', cases=entries, tests=tests, protocol=ref(out/'protocol_seal.json'),
        seal=seal, new_trees=0, coefficient_refits=0, gpu_execution=False, environment=environment,
        seconds=perf_counter()-started, cpu_peak_rss_bytes=memory_guard(),
        limitations=['all former tests exploratory;3 Emotions/Satimage splits share test',
            '500 bootstrap draws conditional on inner OOF predictions, not full refitting uncertainty',
            'one-SE gate heuristic; no probabilistic no-harm guarantee',
            'rank/support/lambda fixed; no online or service-performance claim',
            'multiclass retains prior temperature calibration; raw NLL is reported separately'])
    write(out/'report.json', report)
    from audit_residual_gate import audit
    audit_result = audit(out/'report.json'); write(out/'independent_audit.json', audit_result)
    event('residual_gate_complete', report=ref(out/'report.json'), audit=audit_result, cases=len(entries),
          new_trees=0, coefficient_refits=0, seconds=perf_counter()-started)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2): run(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
