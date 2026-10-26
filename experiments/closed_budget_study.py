"""Bounded periodic vs joint-budget continuation; tests remain sealed."""
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.leaf_budget_runtime import expanded_manifest, load_expanded, load_roles, check_deadline
from experiments.shared_leaf_basis import LeafEnsemble, quality_guard, probabilities
from experiments.uniform_threshold_core import ThresholdModel, threshold_variants, choice_key
from experiments.sparse_leaf_core import Cost, budget_for, leaf_design
from experiments.closed_budget_core import Topology, plan_depth, independent_snapshot
from experiments.closed_budget_projection import compact_proposal
from experiments.parallel_leaf_core import average_increments, dense_parts, blocks_of, prefix
from experiments.online_leaf_core import stitch
from experiments.history_basis_study import raw
from experiments.parallel_leaf_worker import fit_increment


def empty_model(base):
    return ThresholdModel(base, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32),
                          np.zeros(len(base.bias), np.float32))


def combine_actual(checkpoint, increment, prior):
    if checkpoint is None:
        return replace(increment, base=replace(increment.base, bias=prior.copy()))
    merged = average_increments(checkpoint, [increment])
    _, bias, n_features = blocks_of(merged)
    # This reconstructs ONLY the actual compact checkpoint, never the original
    # unsqueezed historical teacher. Full values live only during this proposal.
    return stitch(dense_parts(merged), None, bias, n_features, increment.base.task)


def choose_final_checkpoints(checkpoints, x, y, roles, reference_quality, budget, config, directory, output):
    rows, models = [], {}
    for block, model in checkpoints:
        variants = threshold_variants(raw(model, x[roles['calibration']]), y[roles['calibration']],
            raw(model, x[roles['selection']]), y[roles['selection']], model.base.task,
            config['decision_grid'], config['decision_shrinkage'])
        for v in variants:
            cid = f"block{block}-t{v['threshold_fraction']}"
            rows.append(dict(candidate_id=cid, block=block, parameters=model.parameter_count,
                             payload_bytes=model.payload_bytes, rank=model.base.basis.shape[1],
                             nnz=len(model.values), trees=model.base.n_trees,
                             leaves=len(model.base.coefficients), internal_nodes=len(model.base.nodes), **v))
            models[cid] = model
    if not rows:
        return dict(state='no_accepted_checkpoint', candidate=None, candidates=[])
    if any(not budget.allows(Cost(r['parameters'], r['payload_bytes'])) for r in rows):
        raise RuntimeError('accepted checkpoint violated final resource cap')
    eligible = [r for r in rows if quality_guard(r['validation'], reference_quality)]
    chosen = min(eligible or rows, key=choice_key)
    model = replace(independent_snapshot(models[chosen['candidate_id']]),
                    decision_shifts=np.array(chosen['decision_shifts'], np.float32))
    path = directory/'selected.npz'
    model.save(path)
    loaded = ThresholdModel.load(path)
    np.testing.assert_allclose(loaded.predict(x[roles['selection']]), model.predict(x[roles['selection']]), atol=0, rtol=0)
    return dict(state='accepted' if eligible else 'no_quality_feasible', candidate=chosen, candidates=rows,
                artifact=path.relative_to(output).as_posix(), sha256=sha256(path),
                coo_bytes=loaded.payload_bytes, csr_bytes=loaded.prepare_inference().resident_bytes)


def run_path(method, case, full_record, static_root, config, output, x, y, roles, *, guard_policy=None):
    import cupy as cp
    import resource
    case_id = f"{case['dataset']}_{case['seed']}"
    from experiments.local_guard_core import (GuardPolicy, ActiveGuard, Outcome,
        StopAndRollback, advance_guard, state_record)
    policy = None if guard_policy is None else GuardPolicy(guard_policy)
    path_method = method if policy is None else f'{method}__{policy.value}'
    directory = output/case_id/path_method
    directory.mkdir(parents=True, exist_ok=True)
    reference = Cost(**full_record['reference_cost'])
    budget = budget_for(reference, 'joint', .75)
    full_path = static_root/next(f['artifact'] for f in full_record['fits'] if f['depth'] == 5)
    full_fit = next(f for f in full_record['fits'] if f['depth'] == 5)
    if sha256(full_path) != full_fit['sha256']:
        raise RuntimeError('static full control changed')
    full = LeafEnsemble.load(full_path)
    prior = full.bias.copy()
    c = len(prior)
    ceiling = c-1 if full.task == 'multiclass' else c
    fixed_rank = max(1, ceiling//2)
    fixed_depth = plan_depth(Topology(0, 0, 0), 200, c, fixed_rank,
                             5*reference.parameters//100, budget)
    np.savez(directory/'worker_data.npz', train_x=x[roles['train']], train_y=y[roles['train']],
             validation_x=x[roles['calibration']], validation_y=y[roles['calibration']])
    checkpoint, checkpoints, cycles, built = None, [], [], 0
    guard_state = ActiveGuard()
    started = time.perf_counter()
    for block in range(config.get('closed_max_blocks', 5)):
        check_deadline(config, 2700)
        topology = Topology(0, 0, 0) if checkpoint is None else Topology(
            checkpoint.base.n_trees, len(checkpoint.base.nodes), len(checkpoint.base.coefficients))
        if method in ('periodic_fixed', 'periodic_no_local_guard', 'adaptive_fixed_depth'):
            depth = fixed_depth
        else:
            minimum_rank = max(1, int((ceiling*.25) if checkpoint is None else checkpoint.base.basis.shape[1]*.75))
            depth = plan_depth(topology, 200-topology.trees, c, minimum_rank, 0, budget)
        if depth is None:
            failure = dict(block=block, state='no_depth_with_final_horizon_reserve')
            if policy is not None:
                transition = advance_guard(policy, guard_state, block, Outcome.NO_CANDIDATE)
                failure.update(guard_before=state_record(guard_state), guard_outcome=Outcome.NO_CANDIDATE.value,
                               guard_after=state_record(transition.state), guard_action='rollback')
                guard_state = transition.state
                checkpoints = [(b,m) for b,m in checkpoints if b <= guard_state.retained_through]
            cycles.append(failure)
            break
        cycle_dir = directory/f'block{block}'
        cycle_dir.mkdir()
        init_train = (np.broadcast_to(prior, (len(roles['train']), c)).copy() if checkpoint is None else
                      raw(checkpoint, x[roles['train']]).astype(np.float32))
        init_cal = (np.broadcast_to(prior, (len(roles['calibration']), c)).copy() if checkpoint is None else
                    raw(checkpoint, x[roles['calibration']]))
        initial_path = cycle_dir/'initial.npz'
        np.savez(initial_path, train=init_train, validation=init_cal)
        request = dict(data_path=str(directory/'worker_data.npz'), initial_path=str(initial_path),
            config={**config, 'max_depth': depth}, task=case['task'], seed=case['seed']+40000+40*block,
            trees=40, lr=config['lr'], subsample=1., colsample=1., output=str(cycle_dir/'increment.npz'),
            minimum_free_mib=16384 if case['dataset'] == 'bibtex' else 8192,
            own_pool_limit_bytes=config['own_pool_limit_bytes'])
        fit = fit_increment(request)
        built += 40
        increment = ThresholdModel.load(cycle_dir/'increment.npz')
        proposal = combine_actual(checkpoint, increment, prior)
        prefix_error = None
        if block == 0:
            match = next(f for f in full_record['fits'] if f['depth'] == depth)
            control_path = static_root/match['artifact']
            if sha256(control_path) != match['sha256']:
                raise RuntimeError('static depth control changed')
            control = prefix(empty_model(LeafEnsemble.load(control_path)), 40)
            probes = x[roles['control'][:min(32, len(roles['control']))]]
            prefix_error = float(np.max(np.abs(proposal.predict(probes)-control.predict(probes))))
            if prefix_error > 2e-5:
                raise RuntimeError('first continuation block differs from full-control prefix')
        before_bytes = proposal.payload_bytes
        projection_start = time.perf_counter()
        accepted, projection = compact_proposal(proposal.base, x[roles['train']], y[roles['train']],
            x[roles['calibration']], y[roles['calibration']], x[roles['control']], y[roles['control']],
            budget, reference.parameters, config, fixed=method in ('periodic_fixed', 'periodic_no_local_guard'),
            enforce_local_quality=method != 'periodic_no_local_guard' if policy is None else False,
            prefer_local_quality=policy is not None)
        row = dict(block=block, proposed_depth=depth, state=projection['state'], fit=fit,
                   projection=projection, proposal_full_bytes=before_bytes,
                   projection_seconds=time.perf_counter()-projection_start, prefix_probability_error=prefix_error,
                   cpu_process_cumulative_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   initial_sha256=sha256(initial_path), accepted_trees_before=topology.trees)
        del proposal, increment, init_train, init_cal
        if policy is not None:
            if accepted is not None and not budget.allows(Cost(accepted.parameter_count, accepted.payload_bytes)):
                raise RuntimeError('provisional checkpoint exceeded the hard resource cap')
            outcome = (Outcome.NO_CANDIDATE if accepted is None else
                       (Outcome.PASS if projection['local_quality_pass'] else Outcome.FAIL))
            transition = advance_guard(policy, guard_state, block, outcome)
            row.update(guard_before=state_record(guard_state), guard_outcome=outcome.value,
                       guard_after=state_record(transition.state),
                       guard_action=('rollback' if isinstance(transition, StopAndRollback) else
                                     ('confirmed' if transition.locally_confirmed else 'provisional')))
            guard_state = transition.state
            if isinstance(transition, StopAndRollback):
                checkpoints = [(b,m) for b,m in checkpoints if b <= guard_state.retained_through]
                checkpoint = independent_snapshot(checkpoints[-1][1]) if checkpoints else None
                accepted = None
                row['state'] = 'guard_quality_rejected' if outcome is Outcome.FAIL else projection['state']
        if accepted is None:
            retained = topology.trees if policy is None else (0 if checkpoint is None else checkpoint.base.n_trees)
            row['rollback_to_trees'] = retained
            cycles.append(row)
            write_report(cycle_dir/'decision.json', row)
            emit('closed_block_rejected', dataset=case['dataset'], seed=case['seed'], method=path_method,
                 block=block, state=row['state'], retained_trees=retained)
            break
        if not budget.allows(Cost(accepted.parameter_count, accepted.payload_bytes)):
            raise RuntimeError('hard budget broken at acceptance')
        checkpoint = independent_snapshot(accepted)
        path = cycle_dir/'accepted.npz'
        checkpoint.save(path)
        checkpoints.append((block, independent_snapshot(checkpoint)))
        row.update(artifact=path.relative_to(output).as_posix(), sha256=sha256(path),
                   parameters=checkpoint.parameter_count, payload_bytes=checkpoint.payload_bytes,
                   accepted_trees=checkpoint.base.n_trees, rank=checkpoint.base.basis.shape[1], nnz=len(checkpoint.values))
        cycles.append(row)
        write_report(cycle_dir/'decision.json', row)
        emit('closed_block_accepted', dataset=case['dataset'], seed=case['seed'], method=path_method,
             block=block, depth=depth, trees=checkpoint.base.n_trees, rank=row['rank'], nnz=row['nnz'],
             bytes=row['payload_bytes'], seconds=row['fit']['seconds']+row['projection_seconds'])
        del accepted
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    choice = choose_final_checkpoints(checkpoints, x, y, roles, full_record['validation_reference'],
                                     budget, config, directory, output)
    record = dict(dataset=case['dataset'], seed=case['seed'], method=path_method, budget=asdict(budget),
                  cycles=cycles, choice=choice, new_built_trees=built, native_fit_jobs=built//40,
                  fixed_depth=fixed_depth, test_evaluated=False, seconds=time.perf_counter()-started)
    if policy is not None:
        record.update(allocation_policy=method, guard_policy=policy.value, final_guard_state=state_record(guard_state))
    write_report(directory/'trajectory.json', record)
    return record


def closed_budget_study(config, report, output):
    from experiments.inference_gpu_guard import SharedGpuObserver
    started = time.perf_counter()
    tests = ['tests/test_closed_budget_core.py', 'tests/test_closed_budget_projection.py',
             'tests/test_parallel_leaf_core.py', 'tests/test_continuation_contract.py']
    if config.get('closed_gpu_regression', False):
        tests += ['tests/test_joint_budget_core.py', 'tests/test_uniform_threshold_core.py',
                  'tests/test_sparse_leaf_core.py', 'tests/test_expanded_budget_data.py',
                  'tests/test_leaf_budget_runtime.py', 'tests/test_joint_budget_audit.py',
                  'tests/test_joint_budget_audit_pipeline.py',
                  'tests/test_label_budget_diagnostics.py', 'tests/test_leaf_geometry_core.py',
                  'tests/test_leaf_geometry_study.py', 'tests/test_continuation_leaf_gpu.py']
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=ROOT, check=True, timeout=600)
    data_manifest = expanded_manifest(config)
    path = Path(config['joint_development_manifest'])
    if sha256(path) != config['joint_development_manifest_sha256']:
        raise RuntimeError('static development manifest changed')
    parent = json.loads(path.read_text())
    static_root = path.parent
    records, cases, complete = [], [], True
    methods = tuple(config.get('closed_methods', ('periodic_fixed', 'adaptive_fixed_depth', 'joint_adaptive')))
    allowed_methods = {'periodic_fixed', 'adaptive_fixed_depth', 'joint_adaptive', 'periodic_no_local_guard'}
    if not methods or len(set(methods)) != len(methods) or set(methods)-allowed_methods:
        raise ValueError('invalid or duplicate closed-budget policies')
    with SharedGpuObserver() as observer, threadpool_limits(limits=1):
        for name in config['datasets']:
            x, y = load_expanded(name, data_manifest)
            seeds = config['closed_seeds_by_dataset'][name]
            for case in (c for c in data_manifest['cases'] if c['dataset'] == name and c['seed'] in seeds):
                try:
                    check_deadline(config, 3600)
                except TimeoutError:
                    complete = False
                    break
                roles = load_roles(case, len(x))
                full_record = next(r for r in parent['records'] if r['dataset'] == name and r['seed'] == case['seed'])
                for method in methods:
                    records.append(run_path(method, case, full_record, static_root, config, output, x, y, roles))
                    write_report(output/'progress.json', dict(records=records, test_evaluated=False))
                cases.append(dict(dataset=name, seed=case['seed']))
            if not complete:
                break
    manifest_path = output/'closed_development_manifest.json'
    write_report(manifest_path, dict(records=records, cases=cases, test_evaluated=False,
                                    parent_manifest=str(path), parent_manifest_sha256=sha256(path)))
    report.update(status='development_complete' if complete else 'partial_deadline', records=records,
                  cases=cases, test_evaluated=False, native_fit_jobs=sum(r['native_fit_jobs'] for r in records),
                  new_built_trees=sum(r['new_built_trees'] for r in records),
                  development_manifest_sha256=sha256(manifest_path), gpu_observations=observer.samples,
                  gpu_regression_fit_jobs=4 if config.get('closed_gpu_regression', False) else 0,
                  gpu_regression_trees=16 if config.get('closed_gpu_regression', False) else 0,
                  elapsed_seconds=time.perf_counter()-started)
    emit('closed_development_complete', cases=len(cases), jobs=report['native_fit_jobs'],
         trees=report['new_built_trees'], test_evaluated=False, seconds=report['elapsed_seconds'])
