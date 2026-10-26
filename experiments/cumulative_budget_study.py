"""Effect shell for H-B3C; only registered orx command invokes scientific work."""
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits

from experiments.a100_runner import ROOT,emit,sha256,write_report
from experiments.leaf_budget_runtime import expanded_manifest,load_expanded,load_roles
from experiments.cumulative_budget_core import widen_roles


def cumulative_preflight(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_cumulative_budget_core.py',
        'tests/test_closed_budget_core.py','tests/test_uniform_threshold_core.py'],
        cwd=ROOT,check=True,timeout=600)
    source=expanded_manifest(config)
    cases=[]
    for name in config['cumulative_datasets']:
        x,y=load_expanded(name,source)
        seed=config['cumulative_seeds'][name][0]
        old=next(c for c in source['cases'] if c['dataset']==name and c['seed']==seed)
        original=load_roles(old,len(x))
        roles=widen_roles(y[original['train']],original,old['outputs'],old['task'],seed)
        if old['task']=='multiclass' and set(map(int,y[roles['train']]))!=set(range(old['outputs'])):
            raise RuntimeError('new training role lacks multiclass outputs; do not silently relabel')
        directory=output/f'{name}_{seed}'
        directory.mkdir()
        path=directory/'splits.npz'
        np.savez(path,**roles)
        # Support diagnostics use only development roles, not test targets.
        def supports(indices):
            labels=y[indices]
            return ((np.eye(old['outputs'])[labels.astype(int)] if old['task']=='multiclass' else labels)
                    .sum(axis=0).astype(int).tolist())
        case=dict(old,split_file=str(path),split_sha256=sha256(path),
                  source_split_file=old['split_file'],source_split_sha256=old['split_sha256'],
                  sizes={k:len(v) for k,v in roles.items()},
                  supports={k:supports(v) for k,v in roles.items() if k!='test'},
                  absolute_budget_bytes=config['cumulative_absolute_bytes'][name],protocol_version=3,
                  old_calibration_preserved=True,old_test_preserved=True,
                  test_labels_used=False,full_teacher_required=False)
        cases.append(case)
        emit('cumulative_roles_ready',dataset=name,seed=seed,sizes=case['sizes'],
             budget_bytes=case['absolute_budget_bytes'],split_sha256=case['split_sha256'],
             test_labels_used=False,old_calibration_preserved=True,old_test_preserved=True)
    manifest=dict(version=3,datasets=source['datasets'],cases=cases,
                  source_manifest=config['expanded_manifest'],source_sha256=config['expanded_manifest_sha256'],
                  tree_fits=0,test_labels_used=False,
                  protocol='45/10/10/15/20 approximately; old train only repartitioned; calibration and test exact')
    write_report(output/'cumulative_roles_manifest.json',manifest)
    report.update(status='complete',tree_fits=0,scientific_fits=0,test_labels_used=False,
        cases=len(cases),roles_manifest_sha256=sha256(output/'cumulative_roles_manifest.json'))
    emit('cumulative_preflight_complete',cases=len(cases),tree_fits=0,
         roles_manifest_sha256=report['roles_manifest_sha256'],test_labels_used=False)


def model_bytes(model):
    return max(model.payload_bytes,model.prepare_inference().resident_bytes)


def initial_prior(labels,outputs,task):
    if task=='multiclass': return np.zeros(outputs,np.float32)
    p=np.clip(labels.mean(axis=0),1e-7,1-1e-7)
    return np.log(p/(1-p)).astype(np.float32)


def train_cumulative_path(method,case,development,config,directory,ledger,*,execution_pool=None):
    from experiments.cumulative_budget_core import CumulativeError,absolute_depth
    from experiments.cumulative_budget_projection import cumulative_proposal
    from experiments.closed_budget_core import Topology
    from experiments.closed_budget_study import combine_actual
    from experiments.history_basis_study import raw
    from experiments.parallel_leaf_worker import fit_increment
    from experiments.uniform_threshold_core import ThresholdModel
    from experiments.shared_leaf_basis import probabilities
    directory.mkdir()
    train_x,train_y=development['train']
    cal_x,cal_y=development['calibration']
    c,budget=case['outputs'],case['absolute_budget_bytes']
    ceiling=c-(case['task']=='multiclass')
    horizon=config.get('serving_tree_horizon',200)
    if isinstance(horizon,bool) or not isinstance(horizon,int) or horizon<40 or horizon%40:
        raise ValueError('tree horizon must be a positive multiple of40')
    prior=initial_prior(train_y,c,case['task'])
    error=CumulativeError(np.zeros((len(train_x),c)),case['task'])
    np.savez(directory/'worker_data.npz',train_x=train_x,train_y=train_y,validation_x=cal_x,validation_y=cal_y)
    checkpoint,snapshots,cycles=None,[],[]
    fixed_rank=max(1,ceiling//2)
    fixed_support=5*(budget//4)//100
    fixed_depth=absolute_depth(Topology(0,0,0),horizon,c,fixed_rank,fixed_support,budget)
    history=None
    from experiments.parallel_budget_core import VARIANTS,branch_plans,mean_increment,union_estimate
    parallel_mode=method in VARIANTS
    if method.startswith('gh_'):
        from experiments.absolute_history_core import AbsoluteHistory
        history=AbsoluteHistory(method[3:],c,case['task'],config['history_beta'])
    for block in range(horizon//40):
        topology=Topology(0,0,0) if checkpoint is None else Topology(
            checkpoint.base.n_trees,len(checkpoint.base.nodes),len(checkpoint.base.coefficients))
        minimum=max(1,int(ceiling*.25)) if checkpoint is None else max(1,int(checkpoint.base.basis.shape[1]*.75))
        depth=fixed_depth if method=='periodic' or history is not None else absolute_depth(topology,horizon-topology.trees,c,minimum,0,budget)
        if depth is None:
            cycles.append(dict(block=block,state='no_depth_with_final_horizon_reserve'))
            break
        blockdir=directory/f'block{block}'
        blockdir.mkdir()
        initial_train=np.broadcast_to(prior,(len(train_x),c)).copy() if checkpoint is None else raw(checkpoint,train_x).astype(np.float32)
        initial_cal=np.broadcast_to(prior,(len(cal_x),c)).copy() if checkpoint is None else raw(checkpoint,cal_x)
        np.savez(blockdir/'initial.npz',train=initial_train,validation=initial_cal)
        if ledger['trees']+40>config['cumulative_pilot_tree_cap']:
            raise RuntimeError('preregistered tree cap exceeded before fit')
        request=dict(data_path=str(directory/'worker_data.npz'),initial_path=str(blockdir/'initial.npz'),
            config={**config,'max_depth':depth},task=case['task'],seed=case['seed']+40000+40*block,
            trees=40,lr=config['lr'],subsample=1.,colsample=1.,output=str(blockdir/'increment.npz'),
            minimum_free_mib=8192,own_pool_limit_bytes=config['own_pool_limit_bytes'])
        branch_models=None
        if parallel_mode:
            plans=branch_plans(method,block)
            requests=[dict(request,trees=p.trees,lr=p.lr,seed=request['seed']+p.seed_offset,
                           subsample=p.sample,colsample=p.sample,
                           output=str(blockdir/f'branch{j}.npz') if len(plans)>1 else request['output'])
                      for j,p in enumerate(plans)]
            if len(requests)>1:
                if execution_pool is None: raise ValueError('two-branch execution pool required')
                from experiments.parallel_leaf_study import execute_requests
                branch_fits,execution=execute_requests(requests,execution_pool)
                branch_models=[ThresholdModel.load(r['output']) for r in requests]
                combined=mean_increment(branch_models)
                combined.save(request['output'])
                fit=dict(trees=sum(r['trees'] for r in branch_fits),seconds=execution['sum_worker_fit_seconds'],
                    artifact=request['output'],sha256=sha256(request['output']),
                    initial_sha256=sha256(blockdir/'initial.npz'),data_sha256=sha256(directory/'worker_data.npz'),
                    branch_sources=branch_fits,execution=execution,
                    training_pool_peak_bytes=execution['sum_individual_pool_peaks'],
                    peak_definition='sum of individual pool peaks, not measured simultaneous whole-device peak')
            else: fit=fit_increment(requests[0])
        else: fit=fit_increment(request)
        ledger['trees']+=40
        increment=ThresholdModel.load(blockdir/'increment.npz')
        proposal=combine_actual(checkpoint,increment,prior)
        np.testing.assert_allclose(proposal.predict(train_x[:17]),
            probabilities(initial_train[:17].astype(float)+raw(increment,train_x[:17]),case['task']),
            atol=2e-5,rtol=0)
        start=time.perf_counter()
        basis_estimate=None
        if history is not None:
            basis_estimate=history.observe(train_y,raw(proposal,train_x),config['lambda_l2'],fixed_rank)
        elif method=='parallel2_union' and branch_models is not None:
            candidates=[combine_actual(checkpoint,m,prior) for m in branch_models]
            basis_estimate=union_estimate(candidates,branch_models,train_x)
        observer_seconds=time.perf_counter()-start if basis_estimate is not None else 0.
        accepted,information,new_error=cumulative_proposal(proposal.base,development,budget,error,config,
            fixed=method=='periodic' or history is not None,criterion='cumulative' if method in ('cumulative','cumulative_reserve',*VARIANTS) else 'local',
            basis_estimate=basis_estimate)
        cycle=dict(block=block,depth=depth,fit=fit,projection=information,
                   projection_seconds=time.perf_counter()-start,trained_trees=40,
                   error_state_bytes=error.values.nbytes,basis_observer_seconds=observer_seconds,
                   observer_extra_gpu_transfer_bytes=0)
        if accepted is None:
            cycles.append(cycle)
            write_report(blockdir/'cycle.json',cycle)
            emit('cumulative_block_rejected',dataset=case['dataset'],seed=case['seed'],method=method,
                 block=block,reason=information['state'],trees_built=ledger['trees'])
            break
        if model_bytes(accepted)>budget: raise RuntimeError('accepted checkpoint exceeds serving budget')
        path=blockdir/'accepted.npz'
        accepted.save(path)
        loaded=ThresholdModel.load(path)
        np.testing.assert_allclose(loaded.predict(train_x[:17]),accepted.predict(train_x[:17]),atol=0,rtol=0)
        cycle.update(artifact=str(path),sha256=sha256(path),coo_bytes=loaded.payload_bytes,
                     csr_bytes=loaded.prepare_inference().resident_bytes,rank=loaded.base.basis.shape[1],
                     nnz=len(loaded.values),trees=loaded.base.n_trees,leaves=len(loaded.base.coefficients))
        np.savez(blockdir/'training_error.npz',error=new_error.values,updates=np.array(new_error.updates))
        checkpoint,error=loaded,new_error
        snapshots.append((block,loaded))
        cycles.append(cycle)
        write_report(blockdir/'cycle.json',cycle)
        emit('cumulative_block_accepted',dataset=case['dataset'],seed=case['seed'],method=method,
             block=block,depth=depth,rank=cycle['rank'],nnz=cycle['nnz'],trees=cycle['trees'],
             bytes=model_bytes(loaded),budget_bytes=budget,
             cumulative_error_rms=information['cumulative_error_rms'],trees_built=ledger['trees'])
    result=dict(method=method,cycles=cycles,accepted_checkpoints=len(snapshots),
                trees_built=sum(c.get('trained_trees',0) for c in cycles),
                full_teacher_used=False,gradient_source='actual compact checkpoint',
                error_state_bytes=error.values.nbytes)
    write_report(directory/'trajectory.json',result)
    return snapshots,result


def choose_absolute_models(method,models,calibration,selection,reference,budget,config,directory):
    from experiments.history_basis_study import raw
    from experiments.uniform_threshold_core import threshold_variants,choice_key,ThresholdModel
    from experiments.shared_leaf_basis import quality_guard
    xcal,ycal=calibration
    xsel,ysel=selection
    rows,live=[],{}
    for name,model in models:
        if model_bytes(model)>budget: continue
        variants=threshold_variants(raw(model,xcal),ycal,raw(model,xsel),ysel,model.base.task,
                                     config['decision_grid'],config['decision_shrinkage'])
        for v in variants:
            cid=f'{name}-s{v["threshold_fraction"]}'
            row=dict(candidate_id=cid,rank=model.base.coefficients.shape[1],nnz=len(model.values),
                trees=model.base.n_trees,leaves=len(model.base.coefficients),internal_nodes=len(model.base.nodes),
                parameters=model.parameter_count,payload_bytes=model.payload_bytes,
                csr_bytes=model.prepare_inference().resident_bytes,max_serving_bytes=model_bytes(model),**v)
            rows.append(row)
            live[cid]=model
    if not rows:
        return dict(method=method,state='no_memory_feasible',candidate=None,candidates=[])
    eligible=[r for r in rows if quality_guard(r['validation'],reference)]
    chosen=min(eligible or rows,key=choice_key)
    model=replace(live[chosen['candidate_id']],decision_shifts=np.array(chosen['decision_shifts'],np.float32))
    p=directory/f'{method}-selected.npz'
    model.save(p)
    loaded=ThresholdModel.load(p)
    np.testing.assert_allclose(loaded.predict(xsel),model.predict(xsel),atol=0,rtol=0)
    return dict(method=method,state='accepted' if eligible else 'no_quality_feasible',candidate=chosen,
                candidates=rows,artifact=str(p),sha256=sha256(p),coo_bytes=loaded.payload_bytes,
                csr_bytes=loaded.prepare_inference().resident_bytes)


def cumulative_training(config,report,output):
    import gc
    from experiments.cumulative_budget_core import CumulativeError
    from experiments.cumulative_budget_projection import cumulative_proposal
    from experiments.leaf_budget_runtime import fit_dense
    from experiments.closed_budget_study import empty_model
    from experiments.shared_leaf_basis import reference_class_model
    from experiments.uniform_threshold_core import threshold_variants,ThresholdModel
    from experiments.history_basis_study import raw
    from experiments.parallel_leaf_core import prefix
    from experiments.inference_gpu_guard import SharedGpuObserver
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_cumulative_budget_core.py',
        'tests/test_cumulative_budget_projection.py','tests/test_leaf_budget_runtime.py',
        'tests/test_continuation_contract.py'],cwd=ROOT,check=True,timeout=600)
    source_path=Path(config['cumulative_roles_manifest'])
    if sha256(source_path)!=config['cumulative_roles_manifest_sha256']:
        raise RuntimeError('new development roles changed')
    manifest=json.loads(source_path.read_text())
    results,ledger=[],dict(trees=0)
    with threadpool_limits(limits=1),SharedGpuObserver() as observer:
        for case in manifest['cases']:
            directory=output/f'{case["dataset"]}_{case["seed"]}'
            directory.mkdir()
            x,y=load_expanded(case['dataset'],manifest)
            roles=load_roles(case,len(x))
            # A narrow capability boundary: training/projection never receive test or final selection.
            development={k:(x[roles[k]].copy(),y[roles[k]].copy()) for k in ('train','calibration','control')}
            selection=(x[roles['selection']].copy(),y[roles['selection']].copy())
            paths,trajectories={},[]
            for method in config['cumulative_methods']:
                snapshots,trajectory=train_cumulative_path(method,case,development,config,directory/method,ledger)
                paths[method]=[(f'block{block}',model) for block,model in snapshots]
                trajectories.append(trajectory)
            emit('cumulative_compact_paths_complete',dataset=case['dataset'],trees_built=ledger['trees'],
                 full_controls_started=False)
            dense_by_depth,full_fits={},[]
            fitting_target=np.zeros_like(y)
            for role in ('train','calibration'):
                fitting_target[roles[role]]=y[roles[role]]
            for depth in (5,3,4):
                if ledger['trees']+200>config['cumulative_pilot_tree_cap']:
                    raise RuntimeError('preregistered tree cap exceeded before full control')
                dense,fit=fit_dense(case,depth,config,x,fitting_target,roles)
                if dense.n_trees!=200:
                    raise RuntimeError('full control tree count differs from fixed horizon')
                ledger['trees']+=200
                p=directory/f'dense-depth{depth}.npz'
                dense.save(p)
                fit.update(artifact=str(p),sha256=sha256(p))
                dense_by_depth[depth]=dense
                full_fits.append(fit)
                emit('cumulative_dense_control',dataset=case['dataset'],trees_built=ledger['trees'],**fit)
            full=empty_model(reference_class_model(dense_by_depth[5]) if case['task']=='multiclass' else dense_by_depth[5])
            variants=threshold_variants(raw(full,development['calibration'][0]),development['calibration'][1],
                raw(full,selection[0]),selection[1],case['task'],config['decision_grid'],config['decision_shrinkage'])
            choice=min(variants,key=lambda r:(-r['validation']['f1_macro'],r['validation']['loss'],r['threshold_fraction']))
            full=replace(full,decision_shifts=np.array(choice['decision_shifts'],np.float32))
            fullpath=directory/'full-reference.npz'
            full.save(fullpath)
            reference=choice['validation']
            native,posthoc=[],[]
            for depth,dense in dense_by_depth.items():
                base=empty_model(reference_class_model(dense) if case['task']=='multiclass' else dense)
                native.extend((f'd{depth}-t{trees}',prefix(base,trees)) for trees in (40,80,120,160,200))
                error=CumulativeError(np.zeros((len(development['train'][0]),case['outputs'])),case['task'])
                model,info,_=cumulative_proposal(dense,development,case['absolute_budget_bytes'],error,config)
                write_report(directory/f'posthoc-depth{depth}.json',info)
                if model is not None: posthoc.append((f'd{depth}',model))
            paths.update(native_prefix=native,posthoc=posthoc)
            decisions=[choose_absolute_models(method,models,development['calibration'],selection,reference,
                       case['absolute_budget_bytes'],config,directory) for method,models in paths.items()]
            record=dict(dataset=case['dataset'],seed=case['seed'],task=case['task'],outputs=case['outputs'],
                split_file=case['split_file'],split_sha256=case['split_sha256'],protocol_version=3,
                budget_bytes=case['absolute_budget_bytes'],reference_validation=reference,
                full=dict(artifact=str(fullpath),sha256=sha256(fullpath),coo_bytes=full.payload_bytes,
                          csr_bytes=full.prepare_inference().resident_bytes,threshold_choice=choice),
                trajectories=trajectories,full_fits=full_fits,decisions=decisions,test_evaluated=False)
            results.append(record)
            write_report(directory/'case.json',record)
            write_report(output/'cumulative_development_manifest.json',dict(records=results,
                source_roles_manifest=config['cumulative_roles_manifest'],
                source_roles_sha256=config['cumulative_roles_manifest_sha256'],tree_fits=ledger['trees'],test_evaluated=False))
            emit('cumulative_case_sealed',dataset=case['dataset'],seed=case['seed'],trees_built=ledger['trees'],
                 decisions=[dict(method=d['method'],state=d['state'],candidate=d['candidate']) for d in decisions])
            del paths,dense_by_depth,development,x,y
            gc.collect()
    report.update(status='development_complete',records=results,tree_fits=ledger['trees'],test_evaluated=False,
        environment_unchanged=True,gpu_observations=observer.samples,
        development_manifest_sha256=sha256(output/'cumulative_development_manifest.json'))
    emit('cumulative_development_complete',cases=len(results),tree_fits=ledger['trees'],test_evaluated=False,
         development_manifest_sha256=report['development_manifest_sha256'])
