"""Preregistered paired sparse/rank comparison on frozen teachers, CPU only."""
from __future__ import annotations

from dataclasses import asdict
import gc
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from scipy import sparse

from experiments.a100_runner import ROOT, emit, quality, read_dataset, sha256, write_report
from experiments.shared_leaf_basis import LeafEnsemble, compress, probabilities, quality_guard
from experiments.sparse_leaf_core import (SELECTORS, Budget, Cost, SparseLeafModel,
    budget_for, contrast, coordinate_scores, leaf_design, make_sparse_model,
    residual_capacity, score_order, select_candidate, training_weights)


def base_at_rank(dense,mean,basis,rank,full):
    b=basis[:,:rank]
    return LeafEnsemble(dense.nodes,dense.node_offsets,dense.leaf_offsets,
        full.coefficients[:,:rank].copy(),full.basis[:,:rank].copy(),
        (dense.bias+mean-(mean@b)@b.T).astype(np.float32),dense.n_features,dense.task,'low_rank')


def development_targets(target, indices):
    # The selection layer receives no test rows, features, labels or metrics.
    return target[indices['train']].copy(), target[indices['validation']].copy()


def plan_candidates(dense, mean, basis, train_leaves, train_target,
                    validation_leaves, validation_target, reference_cost, config, progress=None):
    """Entire search input is development-only; returns plans, not test scores."""
    m, c = dense.coefficients.shape
    train = leaf_design(train_leaves,m)
    validation = leaf_design(validation_leaves,m)
    frequency = np.asarray(train.mean(axis=0)).ravel()
    train_teacher = contrast(train @ dense.coefficients + dense.bias, dense.task)
    validation_teacher = validation @ dense.coefficients + dense.bias
    validation_reference = quality(validation_target, probabilities(validation_teacher,dense.task), dense.task)
    full = compress(dense, mean, basis)
    train_latent = train @ full.coefficients
    valid_latent = validation @ full.coefficients
    budgets = [budget_for(reference_cost,kind,f) for kind in config['budget_kinds']
               for f in config['budget_fractions']]
    rank_rows, bases, orders = [], {}, {}
    for rank in range(1,basis.shape[1]+1):
        b = basis[:,:rank]
        bias = (dense.bias+mean-(mean@b)@b.T).astype(np.float32)
        raw = valid_latent[:,:rank] @ full.basis[:,:rank].T + bias
        cost = Cost(m*rank+c*rank+c, dense.payload_bytes-4*m*c+4*(m+c)*rank)
        if not any(budget.allows(cost) for budget in budgets):
            continue
        rank_rows.append({'candidate_id':f'r{rank}-rank_only-0','rank':rank,
            'selector':'rank_only','nnz':0,**asdict(cost),
            'validation':quality(validation_target,probabilities(raw,dense.task),dense.task)})
    if not rank_rows:
        raise RuntimeError('no rank fits the preregistered budgets')
    rows = list(rank_rows)
    sparse_requests = {}
    for budget in budgets:
        allowed = [r['rank'] for r in rank_rows if budget.allows(Cost(r['parameters'],r['payload_bytes']))]
        if not allowed:
            raise RuntimeError(f'no allowed rank for {budget}')
        rmax = max(allowed)
        for f in config['base_rank_fractions']:
            rank = max(1,int(np.floor(rmax*f)))
            sparse_requests.setdefault(rank,[]).append(budget)
    weights = {selector:training_weights(train_target,train_teacher,dense.task,selector,
                                        config['margin_scale'])
               for selector in SELECTORS if selector != 'energy'}
    dense_contrast = contrast(dense.coefficients,dense.task)
    for rank, requests in sorted(sparse_requests.items()):
        base = base_at_rank(dense,mean,basis,rank,full)
        bases[rank] = base
        residual = (dense_contrast-contrast(base.coefficients.astype(float)@base.basis.T,
                                            dense.task)).astype(np.float32)
        error = train_teacher-contrast(train_latent[:,:rank]@base.basis.T+base.bias,dense.task)
        base_validation = valid_latent[:,:rank]@base.basis.T+base.bias
        cost = Cost(base.parameter_count,base.payload_bytes)
        for selector in SELECTORS:
            scores = coordinate_scores(train,residual,error,selector,weights.get(selector))
            order = score_order(scores,residual)
            orders[(rank,selector)] = order
            counts = {min(len(order),int(np.floor(residual_capacity(cost,b)*f)))
                      for b in requests for f in config['residual_capacity_fractions']}
            for count in sorted(counts-{0}):
                flat = order[:count]
                rr,cc = (flat//c).astype(np.int32),(flat%c).astype(np.int32)
                values = residual.ravel()[flat]
                correction = sparse.coo_matrix((values,(rr,cc)),shape=(m,c)).tocsr()
                shift = np.bincount(cc,weights=values.astype(float)*frequency[rr],minlength=c)
                bias = (base.bias.astype(float)-shift).astype(np.float32)
                raw = valid_latent[:,:rank]@base.basis.T+bias+(validation@correction).toarray()
                rows.append({'candidate_id':f'r{rank}-{selector}-{count}','rank':rank,
                    'selector':selector,'nnz':count,'parameters':cost.parameters+count,
                    'payload_bytes':cost.payload_bytes+12*count,
                    'validation':quality(validation_target,probabilities(raw,dense.task),dense.task)})
        if progress:
            progress(rank,len(sparse_requests),len(rows))
    selections=[]
    for budget in budgets:
        for family in ('rank_only',*SELECTORS):
            candidates = [r for r in rows if r['selector'] in ('rank_only',family)]
            chosen,feasible = select_candidate(candidates,validation_reference,budget)
            selections.append({'budget':asdict(budget),'family':family,
                'validation_quality_pass':feasible,'candidate':chosen.copy()})
    return {'rows':rows,'selections':selections,'validation_reference':validation_reference,
            'bases':bases,'orders':orders,'frequency':frequency,'full':full}


def materialize(chosen,plan,dense,mean,basis):
    rank,selector,count = chosen['rank'],chosen['selector'],chosen['nnz']
    base=plan['bases'].get(rank)
    if base is None:
        base=base_at_rank(dense,mean,basis,rank,plan['full'])
    if count:
        residual=(contrast(dense.coefficients,dense.task)
                  -contrast(base.coefficients.astype(float)@base.basis.T,dense.task)).astype(np.float32)
        return make_sparse_model(base,residual,plan['orders'][(rank,selector)],count,plan['frequency'])
    return SparseLeafModel(base,np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32))


def verify_sources(source,manifest):
    for name,expected in manifest['files'].items():
        if sha256(source/name)!=expected:
            raise RuntimeError(f'frozen input changed: {name}')


def prepare_split(name,seed,config,output,data,manifest):
    features,target,source_indices,metadata=data
    source=Path(config['source_dir'])/f'{name}_{seed}'
    directory=output/f'{name}_{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    with np.load(source/'splits.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(source_indices,f['source_indices'])
        indices={k:f[k].copy() for k in ('train','validation','test')}
    dense=LeafEnsemble.load(source/'dense_teacher.npz')
    reference=LeafEnsemble.load(source/'reference.npz')
    with np.load(source/'calibration_diagnostics.npz',allow_pickle=False) as f:
        mean,basis=f['mean'].copy(),f['basis'].copy()
    leaves={k:dense.leaf_indices(features[indices[k]]) for k in ('train','validation')}
    actual_mean=dense.contributions(leaves['train']).mean(axis=0)
    np.testing.assert_allclose(actual_mean,mean,atol=1e-12,rtol=0)
    train_y,validation_y=development_targets(target,indices)
    started=time.perf_counter()
    plan=plan_candidates(dense,mean,basis,leaves['train'],train_y,leaves['validation'],validation_y,
        Cost(reference.parameter_count,reference.payload_bytes),config,
        lambda rank,total,count:emit('sparse_rank_prepared',dataset=name,seed=seed,rank=rank,
                                      grid_ranks=total,candidates=count))
    saved={}
    valid_design=leaf_design(leaves['validation'],len(dense.coefficients))
    for selection in plan['selections']:
        chosen=selection['candidate'];cid=chosen['candidate_id']
        if cid not in saved:
            model=materialize(chosen,plan,dense,mean,basis)
            assert model.parameter_count==chosen['parameters'] and model.payload_bytes==chosen['payload_bytes']
            path=directory/f'{cid}.npz'
            model.save(path)
            loaded=SparseLeafModel.load(path)
            np.testing.assert_array_equal(loaded.raw_from_design(valid_design),model.raw_from_design(valid_design))
            q=quality(validation_y,probabilities(loaded.raw_from_design(valid_design),dense.task),dense.task)
            for key in q:
                if abs(q[key]-chosen['validation'][key])>1e-7:
                    raise RuntimeError('standalone validation differs from search')
            saved[cid]={'artifact':str(path.relative_to(output)).replace('\\','/'),
                        'sha256':sha256(path),'file_bytes':path.stat().st_size}
        selection.update(saved[cid])
    record={'dataset':name,'seed':seed,'task':dense.task,'data_sha256':metadata['sha256'],
        'source_split_sha256':sha256(source/'splits.npz'),
        'reference_cost':{'parameters':reference.parameter_count,'payload_bytes':reference.payload_bytes},
        'validation_reference':plan['validation_reference'],'selections':plan['selections'],
        'validation_candidates':plan['rows'],'saved_models':len(saved),
        'prepare_seconds':time.perf_counter()-started}
    write_report(directory/'selection.json',record)
    emit('sparse_split_sealed',dataset=name,seed=seed,candidates=len(plan['rows']),
         saved_models=len(saved),selections=len(plan['selections']),seconds=record['prepare_seconds'])
    return {k:v for k,v in record.items() if k!='validation_candidates'}


def evaluate_sealed_split(record,config,output,data):
    features,target,_,_=data
    name,seed=record['dataset'],record['seed']
    source=Path(config['source_dir'])/f'{name}_{seed}'
    with np.load(source/'splits.npz',allow_pickle=False) as f:
        test=f['test'].copy()
    dense=LeafEnsemble.load(source/'dense_teacher.npz')
    design=leaf_design(dense.leaf_indices(features[test]),len(dense.coefficients))
    teacher=quality(target[test],probabilities(design@dense.coefficients+dense.bias,dense.task),dense.task)
    results=[];cache={}
    for selection in record['selections']:
        path=output/selection['artifact']
        if sha256(path)!=selection['sha256']:
            raise RuntimeError('sealed model changed')
        if selection['artifact'] not in cache:
            model=SparseLeafModel.load(path)
            prediction=probabilities(model.raw_from_design(design),dense.task)
            cache[selection['artifact']]=quality(target[test],prediction,dense.task)
        q=cache[selection['artifact']]
        candidate=selection['candidate']
        cost=Cost(candidate['parameters'],candidate['payload_bytes'])
        budget=Budget(**selection['budget'])
        assert budget.allows(cost)
        results.append({'dataset':name,'seed':seed,'task':dense.task,
            **selection,'test':q,'test_reference':teacher,
            'macro_retained':q['f1_macro']/teacher['f1_macro'],
            'loss_ratio':q['loss']/teacher['loss'],
            'test_quality_pass':quality_guard(q,teacher),
            'joint_success':selection['validation_quality_pass'] and quality_guard(q,teacher),
            'parameter_reduction':1-cost.parameters/record['reference_cost']['parameters'],
            'payload_reduction':1-cost.payload_bytes/record['reference_cost']['payload_bytes']})
    emit('sparse_test_finished',dataset=name,seed=seed,selections=len(results))
    return results


def summarize(records):
    summaries=[]
    for dataset,kind,fraction in sorted({(r['dataset'],r['budget']['kind'],r['budget']['fraction']) for r in records}):
        group=[r for r in records if (r['dataset'],r['budget']['kind'],r['budget']['fraction'])==(dataset,kind,fraction)]
        control={r['seed']:r for r in group if r['family']=='rank_only'}
        for family in ('rank_only',*SELECTORS):
            rows=[r for r in group if r['family']==family]
            deltas=[r['test']['f1_macro']-control[r['seed']]['test']['f1_macro'] for r in rows]
            summaries.append({'dataset':dataset,'kind':kind,'fraction':fraction,'family':family,
                'splits':len(rows),'quality_passes':sum(r['test_quality_pass'] for r in rows),
                'joint_successes':sum(r['joint_success'] for r in rows),
                'nonzero_residual_models':sum(r['candidate']['nnz']>0 for r in rows),
                'median_macro_retained':float(np.median([r['macro_retained'] for r in rows])),
                'mean_macro_delta_vs_rank':float(np.mean(deltas)),
                'wins_ties_losses_vs_rank':[sum(d>1e-12 for d in deltas),sum(abs(d)<=1e-12 for d in deltas),sum(d < -1e-12 for d in deltas)],
                'median_parameter_reduction':float(np.median([r['parameter_reduction'] for r in rows])),
                'median_payload_reduction':float(np.median([r['payload_reduction'] for r in rows]))})
    return summaries


def sparse_leaf_study(config,report,output):
    def expired(signum,frame):
        raise TimeoutError('approved one-hour CPU study budget exhausted')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.alarm(config['walltime_seconds'])
    started=time.perf_counter()
    report.update(status='running',training_fits=0,gpu_runs=0,
        interpretation='Exploratory reuse of previously inspected tests; not independent confirmation.',
        selection_policy='train-only coordinate scores, validation-only model choice, all models sealed before test scoring')
    try:
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_sparse_leaf_core.py',
                        'tests/test_compactness_core.py'],cwd=ROOT,check=True)
        manifest=json.loads((ROOT/'experiments/sparse_source_manifest.json').read_text())
        verify_sources(Path(config['source_dir']),manifest)
        datasets={name:read_dataset(name,config) for name in config['datasets']}
        selections=[]
        for name in config['datasets']:
            for seed in config['seeds_by_dataset'][name]:
                selections.append(prepare_split(name,seed,config,output,datasets[name],manifest))
                gc.collect()
        selection_path=output/'selection_manifest.json'
        write_report(selection_path,{'source_run':manifest['source_run'],'selections':selections,
                                    'test_labels_used_in_selection':False})
        seal=sha256(selection_path)
        emit('all_selections_sealed',sha256=seal,model_decisions=sum(len(s['selections']) for s in selections),
             completed_teacher_fits=0,test_scoring_started=False)
        records=[]
        for selection in selections:
            if sha256(selection_path)!=seal:
                raise RuntimeError('selection manifest changed after test scoring started')
            records.extend(evaluate_sealed_split(selection,config,output,datasets[selection['dataset']]))
        summaries=summarize(records)
        report.update(status='complete',records=records,summaries=summaries,
            selection_manifest_sha256=seal,source_manifest_sha256=sha256(ROOT/'experiments/sparse_source_manifest.json'),
            standalone_replays='passed',elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json',report)
        emit('sparse_comparison_summary',training_fits=0,gpu_runs=0,decisions=len(records),
             elapsed_seconds=report['elapsed_seconds'],summaries=summaries)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json',report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
