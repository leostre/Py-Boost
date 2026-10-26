"""H-CAL1: cached predictions, output-only fits, sealed exploratory evaluation.

The worker has no training entry point. New model inference, if a cached test
prediction is missing, is CPU-only on the already saved immutable PyBoost model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from budget_quality_worker import digest, write, locked, event, native_load
from budget_quality_core import quality, fit_temperature
from calibration_pilot_core import (POLICIES, Scheme, make_plans, evaluate_plan,
    choose_policy, fit_controls, control_for, choose_model, stratified_order)


ROLES = ('calibration', 'control', 'selection')


def load_development(record, locks):
    case, dataset = record['case'], record['dataset']
    for path, sha in ((case['split_file'], case['split_sha256']),
                      (dataset['path'], dataset['sha256'])):
        if digest(path) != sha:
            raise RuntimeError('frozen data or split changed')
        locks[path] = sha
    with np.load(case['split_file'], allow_pickle=False) as f:
        roles = {r: f[r].copy() for r in (*ROLES, 'train', 'test')}
    indices = np.concatenate(list(roles.values()))
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError('outer roles overlap')
    development = np.concatenate([roles[r] for r in ROLES])
    # The source NPZ stores one target array. Only development labels cross the
    # loader boundary; no features or train/test targets enter the fitting core.
    with np.load(dataset['path'], allow_pickle=False) as f:
        target = f['target'][development].copy()
    sizes = [len(roles[r]) for r in ROLES]
    return target, sizes, development


def model_inputs(records, cfg, locks):
    selected = []
    for r in records:
        if r['family'] not in ('pyboost', 'tabpfn8'):
            continue
        for m in r['models']:
            if m['origin'] == 'archival_control':
                continue
            if r['family'] == 'pyboost' and (m['depth'] not in (3, 5) or m['rounds'] not in cfg['prefixes']):
                continue
            for path, sha in ((m['artifact'], m['sha256']),
                              (m['development_file'], m['development_sha256'])):
                if digest(path) != sha:
                    raise RuntimeError('saved input model or logits changed')
                locks[path] = sha
            with np.load(m['development_file'], allow_pickle=False) as f:
                logits = np.concatenate([f[role] for role in ROLES]).astype(np.float64)
            selected.append((r['family'], m, logits))
    if Counter(f for f, _, _ in selected) != Counter(pyboost=12, tabpfn8=1):
        raise RuntimeError('pilot requires exactly twelve PyBoost models and one frozen TabPFN8')
    return selected


def fit_case(case_id, records, cfg, output, locks):
    native = next(r for r in records if r['family'] == 'pyboost')
    y, sizes, dev_ids = load_development(native, locks)
    b0 = next(m['ledger']['deployed_array_bytes'] for m in native['models']
              if m['identity'] == 'archival-d5-t200')
    models = model_inputs(records, cfg, locks)
    plans = [p for repeat in range(cfg['calibration_pilot']['repeats'])
             for p in make_plans(y, sizes, repeat, cfg['calibration_pilot']['fractions'],
                                 cfg['calibration_pilot']['seed'])]
    rows = []
    plan_rows = []
    # Store actual row IDs once per case, not repeated in every model record.
    for p in plans:
        plan_rows.append(dict(asdict(p), scheme=p.scheme.value,
            fold_unique_fit_rows=[len(set(f.temperature) | set(f.shifts)) for f in p.folds],
            unique_labels_used=len(set().union(*(set(f.temperature) | set(f.shifts) |
                set(f.validation) for f in p.folds)) | set(p.refit_temperature) | set(p.refit_shifts))))
    plan_path = output/'development'/case_id/'plans.json'
    write(plan_path, dict(case=case_id, original_sizes=sizes, development_indices=dev_ids.tolist(),
                         target_support=np.bincount(y, minlength=native['case']['outputs']).tolist(), plans=plan_rows))
    for family, model, z in models:
        started = perf_counter()
        if z.shape != (len(y), native['case']['outputs']):
            raise RuntimeError('development logits shape or class order mismatch')
        if family == 'pyboost':
            cost = model['ledger']['deployed_array_bytes']+4
        else:
            cost = None  # No fake per-task share of TabPFN weights.
        for p in plans:
            validation, diagnostics = evaluate_plan(z, y, p, cfg['decision_grid'])
            # Refit cannot influence validation scores or policy/model selection.
            fitted = fit_controls(z, y, p.refit_temperature, p.refit_shifts, cfg['decision_grid'])
            choices = {}
            for objective in ('loss', 'f1_macro'):
                policy = choose_policy(validation, objective)
                policy_spec = next(a for a in POLICIES if a.name == policy)
                choices[objective] = dict(policy=policy, validation=validation[policy],
                    control=asdict(control_for(policy_spec, fitted)))
            temp = control_for(next(a for a in POLICIES if a.name == 'temperature'), fitted)
            row = dict(case=case_id, family=family, model_id=model['identity'],
                scheme=p.scheme.value, repeat=p.repeat, fraction=p.fraction, cost_bytes=cost,
                validation=validation, fold_diagnostics=diagnostics,
                refit_temperature_rows=len(p.refit_temperature), refit_shifts_rows=len(p.refit_shifts),
                fitted_temperature=asdict(temp), choices=choices)
            # Matched refit sensitivity: same CV-selected policy, but final controls
            # use as many distinct fitting rows as A/B. No extra policy selection.
            if p.scheme == Scheme.CROSSFIT and model['identity'] in ('d5-t200', 'frozen8'):
                count = max(1, int(np.floor(sizes[0]*p.fraction)))+max(1, int(np.floor(sizes[1]*p.fraction)))
                order = stratified_order(y, cfg['calibration_pilot']['seed']+1009*p.repeat+701)
                matched = tuple(order[:count].tolist())
                matched_fit = fit_controls(z, y, matched, matched, cfg['decision_grid'])
                row['matched_refit'] = dict(rows=count,
                    fitted_temperature=asdict(control_for(next(a for a in POLICIES if a.name == 'temperature'), matched_fit)),
                    choices={objective: asdict(control_for(next(a for a in POLICIES if a.name == entry['policy']), matched_fit))
                             for objective, entry in choices.items()})
            # Historical temperature replay before any test access.
            if p.repeat == 0 and p.fraction == 1 and p.scheme == Scheme.SEPARATE:
                old = fit_temperature(z[:sizes[0]], y[:sizes[0]])['temperature']
                if abs(fitted[0]-old) > 2e-5*max(1., old):
                    raise RuntimeError('fast temperature differs from frozen numerical objective')
                row['historical_temperature_replay_error'] = abs(fitted[0]-old)
            rows.append(row)
        event('calibration_model_development_saved', case=case_id, family=family,
              model=model['identity'], scenarios=len(plans), seconds=perf_counter()-started,
              new_tree_fits=0, test_accessed=False)
        # Durable per-model checkpoint, recoverable even if a later model fails.
        write(output/'development'/case_id/(family+'-'+model['identity']+'.json'),
              dict(rows=rows[-len(plans):], model_sha256=model['sha256']))
    choices = []
    for p in plans:
        pool = [r for r in rows if r['family'] == 'pyboost' and r['scheme'] == p.scheme.value
                and r['repeat'] == p.repeat and r['fraction'] == p.fraction]
        for objective in ('loss', 'f1_macro'):
            candidates = [dict(model_id=r['model_id'], cost_bytes=r['cost_bytes'],
                validation=r['choices'][objective]['validation'], policy=r['choices'][objective]['policy']) for r in pool]
            for ratio in cfg['budget_ratios']:
                budget = int(np.floor(b0*ratio))
                chosen = choose_model(candidates, objective, budget)
                choices.append(dict(scheme=p.scheme.value, repeat=p.repeat, fraction=p.fraction,
                    objective=objective, budget_ratio=ratio, budget_bytes=budget, chosen=chosen))
    path = output/'development'/case_id/'case.json'
    write(path, dict(case=case_id, reference_b0_bytes=b0, rows=rows, budget_choices=choices,
        models=[dict(family=f, **{k:m[k] for k in ('identity','artifact','sha256','development_file','development_sha256')})
                for f,m,_ in models], plan_path=str(plan_path), plan_sha256=digest(plan_path)))
    return dict(case=case_id, path=str(path), sha256=digest(path), scenarios=len(rows), choices=len(choices))


def require_seal(path):
    seal = json.loads(Path(path).read_text(encoding='utf-8'))
    if seal.get('status') != 'calibration_choices_sealed_before_test':
        raise RuntimeError('test access requires a complete development seal')
    for item in seal['cases']:
        if digest(item['path']) != item['sha256']:
            raise RuntimeError('development choices changed after seal')
    return seal


def test_predictions(cfg, record, models, seal_path, output):
    require_seal(seal_path)
    case, dataset = record['case'], record['dataset']
    if digest(case['split_file']) != case['split_sha256'] or digest(dataset['path']) != dataset['sha256']:
        raise RuntimeError('frozen test data changed')
    with np.load(case['split_file'], allow_pickle=False) as f:
        ids = f['test'].copy()
    with np.load(dataset['path'], allow_pickle=False) as f:
        y = f['target'][ids].copy()
        x = f['features'][ids].astype(np.float32)
    root = Path(cfg['budget_quality_source']['report']).parent
    result, audit = {}, []
    manifests = {}
    for family, model, _ in models:
        if family not in manifests:
            path = root/'test'/record['case_id']/family/'test.json'
            manifests[family] = json.loads(path.read_text())['predictions']
        item = manifests[family].get(model['identity'])
        if item:
            if digest(item['path']) != item['sha256']:
                raise RuntimeError('saved test prediction changed')
            with np.load(item['path'], allow_pickle=False) as f:
                if not np.array_equal(f['indices'], ids) or not np.array_equal(f['target'], y):
                    raise RuntimeError('cached test row order or labels mismatch')
                raw = f['raw'].copy()
            origin = 'immutable_cached_prediction'
        elif family == 'pyboost':
            if digest(model['artifact']) != model['sha256']:
                raise RuntimeError('model changed before CPU prediction')
            loaded = native_load(model['artifact']).prepare_inference()
            # Replay saved development predictions before trusting new CPU output.
            with np.load(case['split_file'], allow_pickle=False) as f:
                selection = f['selection'].copy()
            with np.load(dataset['path'], allow_pickle=False) as f:
                sx = f['features'][selection].astype(np.float32)
            with np.load(model['development_file'], allow_pickle=False) as f:
                expected = f['selection'].copy()
            replay = loaded.raw(sx)
            np.testing.assert_allclose(replay, expected, atol=2e-4, rtol=2e-5)
            raw = loaded.raw(x)
            origin = 'unchanged_saved_model_cpu_inference'
        else:
            raise RuntimeError('TabPFN inference must reuse saved predictions, never reprepare a context')
        path = output/'test_predictions'/record['case_id']/(family+'-'+model['identity']+'.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, raw=raw, target=y, indices=ids)
        result[(family, model['identity'])] = raw
        audit.append(dict(family=family, model=model['identity'], origin=origin,
                          path=str(path), sha256=digest(path)))
    return y, result, audit


def quality_control(y, raw, control):
    return quality(y, raw, control['temperature'], control['shifts'])


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize_case(development, y, predictions):
    fixed = []; budgets = []
    rows = development['rows']
    index = {(r['family'],r['model_id'],r['scheme'],r['repeat'],r['fraction']):r for r in rows}
    raw_quality = {key:quality(y,z) for key,z in predictions.items()}
    for r in rows:
        if r['model_id'] not in ('d5-t200', 'frozen8'):
            continue
        raw = predictions[(r['family'],r['model_id'])]
        base = raw_quality[(r['family'],r['model_id'])]
        temperature = quality_control(y,raw,r['fitted_temperature'])
        item = {k:r[k] for k in ('case','family','model_id','scheme','repeat','fraction')}
        item.update(temperature=r['fitted_temperature']['temperature'],
            temperature_fit_rows=r['refit_temperature_rows'],shift_fit_rows=r['refit_shifts_rows'],
            raw_loss=base['loss'],temperature_loss=temperature['loss'],
            delta_temperature_loss=temperature['loss']-base['loss'],
            raw_brier=base['brier'],temperature_brier=temperature['brier'],
            raw_f1=base['f1_macro'])
        for objective, choice in r['choices'].items():
            q = quality_control(y,raw,choice['control'])
            item.update({objective+'_'+key:value for key,value in
                         dict(policy=choice['policy'],temperature=choice['control']['temperature'],
                              test_loss=q['loss'],test_f1=q['f1_macro'],test_brier=q['brier'],
                              validation_loss=choice['validation']['loss'],
                              validation_f1=choice['validation']['f1_macro']).items()})
        matched = r.get('matched_refit')
        item.update(matched_refit_rows=matched['rows'] if matched else None,
            matched_temperature_loss=(quality_control(y,raw,matched['fitted_temperature'])['loss'] if matched else None),
            matched_selected_loss=(quality_control(y,raw,matched['choices']['loss'])['loss'] if matched else None),
            matched_selected_f1=(quality_control(y,raw,matched['choices']['f1_macro'])['f1_macro'] if matched else None))
        fixed.append(item)
    for choice in development['budget_choices']:
        chosen = choice['chosen']
        item = dict(case=development['case'], **{k:v for k,v in choice.items() if k != 'chosen'})
        if chosen is None:
            item.update(status='infeasible',model_id=None,policy=None,cost_bytes=None,
                temperature=None,test_loss=None,test_f1=None,validation_loss=None,validation_f1=None)
        else:
            row = index[('pyboost',chosen['model_id'],choice['scheme'],choice['repeat'],choice['fraction'])]
            control = row['choices'][choice['objective']]['control']
            q = quality_control(y,predictions[('pyboost',chosen['model_id'])],control)
            item.update(status='evaluated',model_id=chosen['model_id'],policy=chosen['policy'],
                cost_bytes=chosen['cost_bytes'],temperature=control['temperature'],
                test_loss=q['loss'],test_f1=q['f1_macro'],
                validation_loss=chosen['validation']['loss'],validation_f1=chosen['validation']['f1_macro'])
        budgets.append(item)
    return fixed,budgets


def aggregate(fixed, budgets):
    fixed_groups=[]; budget_groups=[]
    for case in sorted({r['case'] for r in fixed}):
        for family in ('pyboost','tabpfn8'):
            for scheme in Scheme:
                for fraction in (.25,.5,1.):
                    group=[r for r in fixed if r['case']==case and r['family']==family and
                           r['scheme']==scheme.value and r['fraction']==fraction]
                    t=np.array([r['temperature'] for r in group])
                    delta=np.array([r['delta_temperature_loss'] for r in group])
                    fixed_groups.append(dict(case=case,family=family,scheme=scheme.value,fraction=fraction,
                        repeats=len(group),temperature_median=float(np.median(t)),temperature_min=float(t.min()),
                        temperature_max=float(t.max()),temperature_log_sd=float(np.std(np.log(t))),
                        raw_loss=group[0]['raw_loss'],temperature_delta_mean=float(delta.mean()),
                        temperature_harm_count=int(np.sum(delta>1e-12)),
                        selected_loss_mean=float(np.mean([r['loss_test_loss'] for r in group])),
                        selected_loss_harm_count=sum(r['loss_test_loss']>r['raw_loss']+1e-12 for r in group),
                        selected_f1_mean=float(np.mean([r['f1_macro_test_f1'] for r in group])),
                        selected_f1_harm_count=sum(r['f1_macro_test_f1']<r['raw_f1']-1e-12 for r in group)))
    keys=sorted({(r['case'],r['scheme'],r['fraction'],r['objective'],r['budget_ratio']) for r in budgets})
    for case,scheme,fraction,objective,ratio in keys:
        group=[r for r in budgets if (r['case'],r['scheme'],r['fraction'],r['objective'],r['budget_ratio'])==
               (case,scheme,fraction,objective,ratio) and r['status']=='evaluated']
        if not group:
            continue
        counts=Counter(r['model_id'] for r in group)
        budget_groups.append(dict(case=case,scheme=scheme,fraction=fraction,objective=objective,budget_ratio=ratio,
            repeats=len(group),unique_models=len(counts),modal_model=counts.most_common(1)[0][0],
            modal_fraction=counts.most_common(1)[0][1]/len(group),
            test_loss_mean=float(np.mean([r['test_loss'] for r in group])),
            test_loss_sd=float(np.std([r['test_loss'] for r in group])),
            test_f1_mean=float(np.mean([r['test_f1'] for r in group])),
            test_f1_sd=float(np.std([r['test_f1'] for r in group])),
            cost_bytes_mean=float(np.mean([r['cost_bytes'] for r in group]))))
    return fixed_groups,budget_groups


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());started=perf_counter()
    pilot=cfg['calibration_pilot']
    expected=dict(fractions=[.25,.5,1.],repeats=10,folds=3,schemes=['separate','pooled','crossfit'],
        minimum_positive_support=1,minimum_negative_support=1,
        tree_training_allowed=False,gpu_execution=False,stop_after_pilot=True,
        matched_refit_models=['d5-t200','frozen8'])
    for key,value in expected.items():
        if pilot[key]!=value:
            raise RuntimeError('pilot config is outside preregistered contract: '+key)
    source=cfg['calibration_pilot']['source_development']
    original=locked(source['path'],source['sha256'])
    locks={source['path']:source['sha256']}
    event('calibration_pilot_configuration',config=cfg['calibration_pilot'],
          new_tree_fits=0,new_boosting_rounds=0,gpu_execution=False,environment_changed=False)
    cases=[]
    for case in cfg['quality_cases']:
        records=[r for r in original['records'] if r['case_id']==case]
        cases.append(fit_case(case,records,cfg,args.output,locks))
    # All choices for all tasks must exist before the first test loader call.
    for path,sha in locks.items():
        if digest(path)!=sha:
            raise RuntimeError('input changed before final development seal')
    seal=args.output/'pretest_choices.json'
    write(seal,dict(status='calibration_choices_sealed_before_test',cases=cases,input_locks=locks,
        config_sha256=digest(args.config),new_tree_fits=0,test_used_for_selection=False,
        note='Repeated output fits on fixed models; test reused historically, exploratory evaluation only'))
    event('calibration_all_choices_sealed',sha256=digest(seal),cases=len(cases),test_accessed=False)
    fixed=[];budgets=[];audits=[]
    for item in require_seal(seal)['cases']:
        records=[r for r in original['records'] if r['case_id']==item['case']]
        native=next(r for r in records if r['family']=='pyboost')
        models=model_inputs(records,cfg,{})
        y,predictions,audit=test_predictions(cfg,native,models,seal,args.output)
        development=json.loads(Path(item['path']).read_text())
        f,b=summarize_case(development,y,predictions)
        fixed.extend(f);budgets.extend(b);audits.extend(audit)
        csv_write(args.output/'fixed_models.csv',fixed)
        csv_write(args.output/'budget_choices.csv',budgets)
        event('calibration_case_test_evaluated',case=item['case'],fixed_rows=len(f),budget_choices=len(b),new_tree_fits=0)
    fg,bg=aggregate(fixed,budgets)
    csv_write(args.output/'fixed_summary.csv',fg);csv_write(args.output/'budget_summary.csv',bg)
    write(args.output/'report.json',dict(status='complete_paused_for_discussion',config=cfg['calibration_pilot'],
        cases=cases,pretest_seal_sha256=digest(seal),new_tree_fits=0,new_boosting_rounds=0,
        test_used_for_selection=False,gpu_execution=False,environment_changed=False,
        fixed_summary=fg,budget_summary=bg,prediction_audit=audits,seconds=perf_counter()-started,
        caveats=['one historical outer split per task; repeated development partitions are not new training seeds',
                 'cross-fitting changes fit/validation/refit sample counts; see plans, not an equal-N causal comparison',
                 'selected CV scores are not unbiased; test was used historically; no independent confirmation']))
    for r in fg:
        if r['fraction']==1.:
            event('calibration_fixed_summary',**r)
    event('calibration_pilot_complete',report_sha256=digest(args.output/'report.json'),
        fixed_rows=len(fixed),budget_choices=len(budgets),source_models=65,
        new_tree_fits=0,new_boosting_rounds=0,stop_for_discussion=True,seconds=perf_counter()-started)


if __name__=='__main__':
    main()
