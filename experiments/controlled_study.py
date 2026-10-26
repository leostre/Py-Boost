"""Paired structural controllers; reference configuration selected without test fields."""
import gc
import hashlib
import json
import time
import numpy as np
from experiments.adaptive_callback import CompactnessController
from experiments.compactness_core import ValidationCandidate, select_compact
from experiments.mechanism_core import uniform_reference_settings


def reference_settings(records, dataset, seed):
    rows = [r for r in records if r['dataset'] == dataset and r['seed'] == seed and r['method'] == 'full']
    if not rows:
        raise ValueError('missing paired full reference')
    # Only explicitly allowed fields reach the choice, not test scores.
    ref = min(rows, key=lambda r: (r['validation']['loss'], r['parameters'], r['candidate_id']))
    return {'max_depth': ref['max_depth'], 'lambda_l2': ref['lambda_l2'],
            'parameters': ref['parameters'], 'validation_loss': ref['validation']['loss'],
            'candidate_id': ref['candidate_id']}


def transfer_reference_settings(records, dataset, seed):
    """Calibrate from the final full model's counts; never inspect any scores."""
    rows = [r for r in records if r['dataset'] == dataset and r['seed'] == seed
            and r['method'] == 'full_reference' and r.get('is_full_reference') is True]
    if not rows:
        raise ValueError('missing immutable transfer full reference')
    ref = max(rows, key=lambda r: r['boosting_steps'])
    if ref['boosting_steps'] != 200 or not isinstance(ref['parameters'], int) or ref['parameters'] <= 0:
        raise ValueError('expected positive final 200-tree transfer reference count')
    return {'max_depth': 5, 'lambda_l2': ref['lambda_l2'],
            'parameters': ref['parameters'], 'candidate_id': ref['candidate_id']}


def controlled_study(config, report, output):
    import cupy as cp
    from experiments.a100_runner import (read_dataset, make_partitions, build_model,
                                        model_complexity, quality, write_report, emit, sha256)
    # Actual GPU integration guard before scientific fits; synthetic labels only.
    rng = np.random.default_rng(917)
    smoke_x = rng.normal(size=(512, 8)).astype(np.float32)
    smoke_y = np.argmax(smoke_x[:, :4], axis=1).astype(np.int32)
    smoke_results = []
    smoke_variants = [('fixed', 'none', {}), ('adaptive', 'none', {}), ('adaptive', 'history', {})]
    if config['phase'] == 'mechanism_ablation':
        smoke_variants += [('schedule', 'none', {'static_fraction':.25, 'static_depth_ceiling':3}),
                           ('schedule','history',{'budget_channels':'outputs'}),
                           ('schedule','history',{'budget_channels':'depth'})]
    for mode, spectral, options in smoke_variants:
        check = CompactnessController(mode=mode, spectral=spectral, parameter_cap=10000,
                                      seed=917, task='multiclass', emit=emit, **options)
        check_model = build_model('full' if spectral=='none' and not options else 'sketchboost_topk_half',
                                  'multiclass', 4, config, 917, 3, 10., trees=80, callbacks=[check])
        check_model.fit(smoke_x[:400], smoke_y[:400],
                        eval_sets=[{'X':smoke_x[400:], 'y':smoke_y[400:]}])
        actual = model_complexity(check_model, [len(check_model.models)])[-1]['parameters']
        if actual != check.parameters or actual > check.cap or not check.trace:
            raise RuntimeError('GPU controller integration check failed')
        smoke_results.append(dict(mode=mode, spectral=spectral, parameters=actual,
                                  options=options, steps=len(check_model.models), checks=len(check.spectral_checks)))
        del check_model, check
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    report['gpu_controller_preflight'] = smoke_results
    emit('gpu_controller_preflight', results=smoke_results)
    baseline_path = config['baseline_report']
    if sha256(baseline_path) != config['baseline_sha256']:
        raise RuntimeError('immutable H1 reference fingerprint changed')
    baseline = json.loads(open(baseline_path, encoding='utf-8').read())
    if baseline['status'] != 'complete':
        raise ValueError('controller requires completed H1 reference')
    transfer_baseline = None
    uniform_records = []
    for spec in config.get('uniform_reference_reports', []):
        if sha256(spec['path']) != spec['sha256']:
            raise RuntimeError('immutable uniform reference fingerprint changed')
        source = json.loads(open(spec['path'], encoding='utf-8').read())
        if source['status'] != 'complete' or source['completed_fits'] != source['expected_fits']:
            raise ValueError('incomplete uniform reference')
        uniform_records.extend(source['records'])
    if config.get('transfer_reference_report'):
        if config['phase'] != 'h5_transfer':
            raise ValueError('realized transfer reference is only supported for H5')
        path = config['transfer_reference_report']
        if sha256(path) != config['transfer_reference_sha256']:
            raise RuntimeError('immutable transfer reference fingerprint changed')
        transfer_baseline = json.loads(open(path, encoding='utf-8').read())
        if transfer_baseline['status'] != 'complete' or transfer_baseline['completed_fits'] != transfer_baseline['expected_fits']:
            raise ValueError('incomplete transfer reference')
    records, references, traces = [], [], []
    def seeds_for(name):
        return config.get('seeds_by_dataset', {}).get(name, config.get('seeds', []))
    expected = sum(len(seeds_for(n)) for n in config['datasets']) * len(config['controllers'])
    report.update(status='running', records=records, references=references,
                  controller_traces=traces, expected_fits=expected, completed_fits=0)
    for name in config['datasets']:
        features, target, source, metadata = read_dataset(name, config)
        task = metadata['task']
        outputs = int(target.max()+1) if task == 'multiclass' else target.shape[1]
        for seed in seeds_for(name):
            train, validation, test = make_partitions(target, task, seed)
            output.joinpath('splits').mkdir(exist_ok=True)
            np.savez_compressed(output / 'splits' / f'{name}_{seed}.npz', train=train,
                                validation=validation, test=test, source_indices=source)
            split_hash = sha256(output / 'splits' / f'{name}_{seed}.npz')
            if config['phase'] == 'mechanism_ablation':
                reference = uniform_reference_settings(uniform_records, name, seed)
                if reference['data_sha256'] != metadata['sha256']:
                    raise RuntimeError('uniform reference data does not match')
            elif config['phase'] == 'h5_transfer':
                # New tasks/seeds: frozen universal rule, not H1 test-derived per-task knobs.
                reference = {'max_depth': 5, 'lambda_l2': 10.,
                             'parameters': outputs * (1 + 32 * config['ntrees']),
                             'candidate_id': 'fixed-transfer-d5-l210-t200'}
                if transfer_baseline is not None:
                    reference = transfer_reference_settings(transfer_baseline['records'], name, seed)
            else:
                reference = reference_settings(baseline['records'], name, seed)
            reference['anchor_lambda_l2'] = reference['lambda_l2']
            reference['lambda_l2'] = float(config.get('lambda_l2_override', reference['lambda_l2']))
            references.append({'dataset': name, 'seed': seed, **reference})
            cap = int(reference['parameters'] * config['parameter_fraction'])
            for variant in config['controllers']:
                controller = None
                if variant['mode'] != 'unconstrained':
                    controller = CompactnessController(mode=variant['mode'], spectral=variant['spectral'],
                                                       parameter_cap=cap, seed=seed, task=task, emit=emit,
                                                       improvement_floor=config.get('improvement_floor', .002),
                                                       budget_channels=variant.get('budget_channels', 'joint'),
                                                       static_fraction=variant.get('static_fraction'),
                                                       static_depth_ceiling=variant.get('static_depth_ceiling', 7),
                                                       activation_step=variant.get('activation_step', 60))
                method = variant.get('sketch', config['control_sketch'])
                depth = reference['max_depth'] if controller is None else 3
                model = build_model(method, task, outputs, config, seed, depth,
                                    reference['lambda_l2'], callbacks=[] if controller is None else [controller])
                cp.cuda.get_current_stream().synchronize()
                started = time.perf_counter()
                model.fit(features[train], target[train],
                          eval_sets=[{'X': features[validation], 'y': target[validation]}])
                cp.cuda.get_current_stream().synchronize()
                elapsed = time.perf_counter() - started
                checkpoints = sorted(set([c for c in config['checkpoints'] if c <= len(model.models)]
                                         + [len(model.models)]))
                complexity = model_complexity(model, checkpoints)
                if config['phase'] == 'mechanism_ablation' and controller is None and complexity[-1]['parameters'] != reference['parameters']:
                    raise RuntimeError('full reference count did not reproduce')
                if controller is not None and (complexity[-1]['parameters'] != controller.parameters or
                                                complexity[-1]['parameters'] > cap):
                    raise RuntimeError('independent parameter counter disagrees with controller')
                predictions = {split: model.predict_staged(features[idx], iterations=[c-1 for c in checkpoints])
                               for split, idx in [('train', train), ('validation', validation), ('test', test)]}
                for j, checkpoint in enumerate(checkpoints):
                    original_trees = model.models
                    timings = []
                    try:
                        model.models = original_trees[:checkpoint]
                        model.predict(features[test])
                        for _ in range(3):
                            cp.cuda.get_current_stream().synchronize()
                            inference_started = time.perf_counter()
                            model.predict(features[test])
                            cp.cuda.get_current_stream().synchronize()
                            timings.append((time.perf_counter()-inference_started)*1000)
                    finally:
                        model.models = original_trees
                    record = {'candidate_id': f'{name}/{seed}/{variant["name"]}/t{checkpoint}',
                              'dataset': name, 'seed': seed, 'method': variant['name'],
                              'is_full_reference':controller is None and method=='full',
                              'sketch': method, 'lambda_l2': reference['lambda_l2'],
                              'mechanism_variant': variant,
                              'parameter_cap': cap if controller is not None else None,
                              **complexity[j], 'full_fit_train_seconds': elapsed,
                              'controller_seconds': 0. if controller is None else controller.overhead,
                              'inference_median_ms':float(np.median(timings)), 'inference_repeats':3,
                              'retained_history_bytes':0 if controller is None else
                              sum(g.nbytes+h.nbytes for g,h in controller.frames),
                              'data_sha256': metadata['sha256'],
                              'split_sha256': split_hash,
                              **{split: quality(target[idx], predictions[split][j], task)
                                 for split, idx in [('train', train), ('validation', validation), ('test', test)]}}
                    records.append(record)
                if controller is not None:
                    traces.append({'dataset': name, 'seed': seed, 'method': variant['name'],
                                   'trace': controller.trace, 'spectral_checks':controller.spectral_checks})
                    if controller.frames:
                        output.joinpath('history').mkdir(exist_ok=True)
                        np.savez_compressed(output/'history'/f'{name}_{seed}_{variant["name"]}.npz',
                                            gradient=np.stack([g for g,h in controller.frames], axis=2),
                                            hessian=np.stack([h for g,h in controller.frames], axis=2),
                                            epochs=np.array(controller.frame_epochs),
                                            train_relative_rows=controller.probe)
                report['completed_fits'] += 1
                emit('controlled_fit_finished', dataset=name, seed=seed, method=variant['name'],
                     steps=len(model.models), parameters=complexity[-1]['parameters'], cap=cap,
                     validation_final=records[-1]['validation'], test_final=records[-1]['test'],
                     seconds=elapsed, controller_seconds=0. if controller is None else controller.overhead,
                     completed=report['completed_fits'], expected=expected)
                write_report(output / 'report.json', report)
                del model, predictions, controller
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
    selections = []
    for name in config['datasets']:
        for seed in seeds_for(name):
            for variant in config['controllers']:
                rows = [r for r in records if r['dataset']==name and r['seed']==seed and r['method']==variant['name']]
                choices = [ValidationCandidate(r['candidate_id'], r['parameters'], r['validation']['loss']) for r in rows]
                reference, compact = select_compact(choices, .05)
                selections.append({'dataset': name, 'seed': seed, 'method': variant['name'],
                                   'reference': reference.candidate_id, 'compact': compact.candidate_id})
    report.update(status='complete', selections=selections,
                  interpretation='Causal rules and validation-only prefixes; compare against frozen references offline.')
    emit('controlled_summary', completed_fits=report['completed_fits'], expected_fits=expected,
         selections=selections)
