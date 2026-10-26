"""One preregistered, reference-calibrated depth allocation hypothesis."""
import gc
import time
import numpy as np
from experiments.risk_budget_callback import QualityEfficiencyController
from experiments.matched_budget import budget_regime


def risk_budget_study(config, report, output):
    import cupy as cp
    from experiments.a100_runner import (read_dataset, make_partitions, build_model,
        model_complexity, quality, write_report, emit, sha256)

    def cleanup():
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    rng = np.random.default_rng(917)
    regime = budget_regime(config)
    x = rng.normal(size=(512, 8)).astype(np.float32)
    y = np.argmax(x[:, :4], axis=1).astype(np.int32)
    smoke = []
    methods = config.get('controller_methods', ['fixed','alternating','latest','history'])
    if len(methods) != len(set(methods)) or any(m not in ('fixed','alternating','latest','history','deceleration') for m in methods):
        raise ValueError('invalid controller methods')
    for mode, smoke_task in [(m,t) for m in methods
                            for t in ('multiclass','multilabel')]:
        smoke_y = y if smoke_task == 'multiclass' else (x[:,:4] > 0).astype(np.float32)
        c = QualityEfficiencyController(mode=mode, reference=dict(loss=.0001, f1_macro=1., f1_weighted=1.),
            parameter_cap=50000, seed=917, task=smoke_task, horizon=140, emit=lambda *a, **k: None,
            stop_on_quality=regime.stop_on_quality)
        m = build_model('full', smoke_task, 4, config, 917, 5, 1., trees=140, callbacks=[c])
        m.fit(x[:400], smoke_y[:400], eval_sets=[{'X':x[400:], 'y':smoke_y[400:]}])
        count = model_complexity(m, [len(m.models)])[-1]['parameters']
        measured = quality(smoke_y[400:], m.predict(x[400:]), smoke_task)
        if count != c.parameters or count > c.cap or len(m.models) != 140:
            raise RuntimeError('quality controller integration counter/horizon mismatch')
        if any(abs(measured[k]-c.checks[-1]['validation'][k]) > 1e-6 for k in measured):
            raise RuntimeError('callback and offline metrics differ')
        if mode in ('latest', 'history', 'deceleration') and not any(k['forecasts'] for k in c.checks):
            raise RuntimeError('smoke failed to exercise informed decisions')
        smoke.append(dict(mode=mode, task=smoke_task, parameters=count, steps=len(m.models), final_quality=measured,
            decay_forecasts=sum(f['block_decay']<1 for k in c.checks for f in k['forecasts'])))
        del m, c
        cleanup()
    report['gpu_controller_preflight'] = smoke
    emit('gpu_controller_preflight', results=smoke)
    records, traces = [], []
    variants = ['full_reference'] + methods
    expected = sum(len(config['seeds_by_dataset'][n]) for n in config['datasets']) * len(variants)
    report.update(status='running', expected_fits=expected, completed_fits=0, records=records,
                  controller_traces=traces, primary_model=config.get('primary_model', 'history'),
                  calibration_policy=f'one full d5/lambda1/t{regime.reference_trees} reference per split; cost reported separately',
                  selection_policy=('cap or horizon stop; no quality stop; final model only; no test input'
                                    if regime.matched else 'online validation stop; final model only; no test input'))
    for name in config['datasets']:
        features, target, source, metadata = read_dataset(name, config)
        task = metadata['task']
        outputs = int(target.max()+1) if task == 'multiclass' else target.shape[1]
        for seed in config['seeds_by_dataset'][name]:
            train, valid, test = make_partitions(target, task, seed)
            output.joinpath('splits').mkdir(exist_ok=True)
            split_file = output/'splits'/f'{name}_{seed}.npz'
            np.savez_compressed(split_file, train=train, validation=valid, test=test, source_indices=source)
            split_hash = sha256(split_file)
            reference_quality, reference_parameters, reference_seconds = None, None, None
            for mode in variants:
                controller = None if mode == 'full_reference' else QualityEfficiencyController(
                    mode=mode, reference=reference_quality,
                    parameter_cap=regime.cap(reference_parameters, outputs, config['parameter_fraction']),
                    seed=seed, task=task, horizon=regime.controller_horizon, emit=emit,
                    stop_on_quality=regime.stop_on_quality)
                m = build_model('full', task, outputs, config, seed, 5, 1.,
                                trees=regime.reference_trees if controller is None else regime.controller_horizon,
                                callbacks=[] if controller is None else [controller])
                cp.cuda.get_current_stream().synchronize()
                started = time.perf_counter()
                m.fit(features[train], target[train], eval_sets=[{'X':features[valid], 'y':target[valid]}])
                cp.cuda.get_current_stream().synchronize()
                elapsed = time.perf_counter()-started
                checkpoints = sorted(set([c for c in config['checkpoints'] if c <= len(m.models)] + [len(m.models)]))
                costs = model_complexity(m, checkpoints)
                if controller is None:
                    if len(m.models) != regime.reference_trees:
                        raise RuntimeError('full reference stopped early')
                    reference_quality = quality(target[valid], m.predict(features[valid]), task)
                    reference_parameters, reference_seconds = costs[-1]['parameters'], elapsed
                    if regime.matched:
                        warmup = model_complexity(m, [60])[-1]['parameters']
                        cap = regime.cap(reference_parameters, outputs, config['parameter_fraction'])
                        if cap < warmup + 32 * outputs:
                            raise RuntimeError('matched budget cannot safely admit common warmup')
                elif costs[-1]['parameters'] != controller.parameters or costs[-1]['parameters'] > controller.cap:
                    raise RuntimeError('independent coefficient count mismatch')
                predictions = m.predict_staged(features[valid], iterations=[c-1 for c in checkpoints])
                # Test is evaluated exactly once, on the trained final model only.
                final_test = quality(target[test], m.predict(features[test]), task)
                for j, checkpoint in enumerate(checkpoints):
                    records.append(dict(dataset=name, seed=seed, method=mode,
                        is_final=checkpoint==len(m.models), candidate_id=f'{name}/{seed}/{mode}/t{checkpoint}',
                        is_full_reference=controller is None, **costs[j],
                        data_sha256=metadata['sha256'], split_sha256=split_hash,
                        parameter_cap=None if controller is None else controller.cap,
                        outputs=outputs,
                        budget_match=(regime.endpoint(costs[j]['parameters'], controller.cap, outputs,
                                      controller.checks[-1]['stop_reason'])
                                      if regime.matched and controller is not None and checkpoint==len(m.models) else None),
                        reference_parameters=reference_parameters, reference_validation=reference_quality,
                        full_fit_train_seconds=elapsed, calibration_train_seconds=reference_seconds,
                        total_calibrated_train_seconds=elapsed if controller is None else elapsed+reference_seconds,
                        controller_seconds=0. if controller is None else controller.overhead,
                        retained_history_bytes=0 if controller is None else sum(g.nbytes+h.nbytes for g,h in controller.frames),
                        stop_reason='horizon' if controller is None else controller.checks[-1]['stop_reason'],
                        validation=quality(target[valid], predictions[j], task),
                        test=final_test if checkpoint==len(m.models) else None))
                if controller is not None:
                    measured = records[-1]['validation']
                    if any(abs(measured[k]-controller.checks[-1]['validation'][k]) > 1e-6 for k in measured):
                        raise RuntimeError('final validation did not match causal callback')
                    traces.append(dict(dataset=name, seed=seed, method=mode,
                                       trace=controller.trace, checks=controller.checks))
                    output.joinpath('history').mkdir(exist_ok=True)
                    np.savez_compressed(output/'history'/f'{name}_{seed}_{mode}.npz',
                        gradient=np.stack([g for g,h in controller.frames],axis=2),
                        hessian=np.stack([h for g,h in controller.frames],axis=2),
                        epochs=np.array(controller.epochs), train_relative_rows=controller.probe)
                report['completed_fits'] += 1
                write_report(output/'report.json',report)
                emit('quality_fit_finished', completed=report['completed_fits'], expected=expected,
                     dataset=name, seed=seed, method=mode, steps=len(m.models),
                     parameters=costs[-1]['parameters'], seconds=elapsed,
                     stop_reason=records[-1]['stop_reason'], validation=records[-1]['validation'], test=records[-1]['test'])
                del m, predictions, controller
                cleanup()
    report.update(status='complete', interpretation='Reference-calibrated quality-efficiency depth control; not RMT validation.')
    emit('quality_budget_final_summary', primary_model=report['primary_model'],completed_fits=report['completed_fits'],
         final_records=[{k:r[k] for k in ('dataset','seed','method','boosting_steps','parameters','stop_reason','validation','test')}
                        for r in records if r['is_final']])
