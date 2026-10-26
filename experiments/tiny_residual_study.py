"""Small residual caps vs folded calibration, sealed selection, no tree fits."""
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
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from scipy import sparse

from experiments.a100_runner import ROOT,emit,read_dataset,sha256,write_report
from experiments.shared_leaf_basis import LeafEnsemble,compress,quality_guard
from experiments.sparse_leaf_core import (Cost,Budget,SparseLeafModel,budget_for,contrast,
    leaf_design,coordinate_scores,score_order,training_weights,residual_capacity,
    select_candidate,make_sparse_model)
from experiments.sparse_leaf_study import base_at_rank,verify_sources
from experiments.output_calibration_core import (DecisionLeafModel,fit_affine,fold_affine,
    fit_decision_shifts,calibrated_quality,indicator)

EMPTY=np.empty(0,dtype=np.float32)
FAMILIES=('rank_only','calibration','sparse_01','sparse_02','sparse_05')
SMALL_CAPS={'sparse_01':.01,'sparse_02':.02,'sparse_05':.05}


def calibration_partition(validation_target,task,c,seed):
    rows=np.arange(len(validation_target))
    y=indicator(validation_target,c,task)
    splitter=MultilabelStratifiedShuffleSplit(n_splits=1,test_size=.5,random_state=seed+90000)
    fit,select=next(splitter.split(rows[:,None],y))
    if len(np.intersect1d(fit,select)) or sorted(np.r_[fit,select].tolist())!=rows.tolist():
        raise RuntimeError('calibration partition is not disjoint and exhaustive')
    return fit,select


def eligible_family(row,family,reference_parameters):
    if row['kind']=='rank_only':
        return True
    if family=='calibration':
        return row['kind'] in ('affine','decision')
    if family in SMALL_CAPS:
        return row['kind']=='sparse' and row['nnz']<=int(np.floor(SMALL_CAPS[family]*reference_parameters))
    return False


def tiny_plan(dense,mean,basis,train_leaves,train_target,calibration_leaves,
              calibration_target,selection_leaves,selection_target,reference_cost,config,progress=None):
    """No test data or labels can enter any coordinate/calibration decision."""
    m,c=dense.coefficients.shape
    designs={k:leaf_design(v,m) for k,v in [('train',train_leaves),('calibration',calibration_leaves),
                                          ('selection',selection_leaves)]}
    full=compress(dense,mean,basis)
    latents={k:d@full.coefficients for k,d in designs.items()}
    teacher_train=contrast(designs['train']@dense.coefficients+dense.bias,dense.task)
    teacher_selection=designs['selection']@dense.coefficients+dense.bias
    reference_quality=calibrated_quality(selection_target,teacher_selection,dense.task,EMPTY)
    frequency=np.asarray(designs['train'].mean(axis=0)).ravel()
    budgets=[budget_for(reference_cost,k,f) for k in config['budget_kinds'] for f in config['budget_fractions']]
    rows=[];bases={};orders={};fits=[]
    rank_costs={r:Cost((m+c)*r+c,dense.payload_bytes-4*m*c+4*(m+c)*r)
                for r in range(1,basis.shape[1]+1)}
    allowed_ranks=[r for r,cost in rank_costs.items() if any(b.allows(cost) for b in budgets)]
    for rank in allowed_ranks:
        base=base_at_rank(dense,mean,basis,rank,full);cost=rank_costs[rank]
        raw_calibration=latents['calibration'][:,:rank]@base.basis.T+base.bias
        raw_selection=latents['selection'][:,:rank]@base.basis.T+base.bias
        common={'rank':rank,'nnz':0,**asdict(cost),'fit':None,'decision_shifts':[]}
        rows.append({**common,'candidate_id':f'r{rank}-rank_only-0','kind':'rank_only','selector':'rank_only',
            'validation':calibrated_quality(selection_target,raw_selection,dense.task,EMPTY)})
        for balanced in (False,True):
            for regularization in config['calibration_regularization']:
                fit=fit_affine(raw_calibration,calibration_target,dense.task,balanced,regularization)
                fits.append({k:fit[k] for k in ('converged','iterations','initial_objective','final_objective')})
                calibrated=fold_affine(base,fit['scale'],fit['shift'])
                assert calibrated.parameter_count==cost.parameters and calibrated.payload_bytes==cost.payload_bytes
                raw=(designs['selection']@calibrated.coefficients)@calibrated.basis.T+calibrated.bias
                tag='balanced' if balanced else 'nll'
                reg=str(regularization).replace('.','p')
                rows.append({**common,'candidate_id':f'r{rank}-affine_{tag}-{reg}',
                    'kind':'affine','selector':f'affine_{tag}','fit':fit,
                    'regularization':regularization,'validation':calibrated_quality(selection_target,raw,dense.task,EMPTY)})
        decision_cost=Cost(cost.parameters+c,cost.payload_bytes+4*c)
        if any(b.allows(decision_cost) for b in budgets):
            shifts=fit_decision_shifts(raw_calibration,calibration_target,dense.task,config['decision_grid'])
            for fraction in config['decision_shrinkage']:
                thresholds=(fraction*shifts).astype(np.float32)
                rows.append({**common,**asdict(decision_cost),
                    'candidate_id':f'r{rank}-decision-{str(fraction).replace(".","p")}',
                    'kind':'decision','selector':'decision','decision_shifts':thresholds.tolist(),
                    'validation':calibrated_quality(selection_target,raw_selection,dense.task,thresholds)})
        if progress and (rank%10==0 or rank==allowed_ranks[-1]):
            progress('calibration',rank,len(rows))
    if not allowed_ranks:
        raise RuntimeError('no rank fits the fixed budgets')
    # Include ranks spending the entire small residual cap, not only old r/rmax fractions.
    sparse_ranks=set()
    for budget in budgets:
        feasible=[r for r in allowed_ranks if budget.allows(rank_costs[r])]
        if not feasible:raise RuntimeError('no base rank fits a budget')
        maximum=max(feasible)
        sparse_ranks.update(max(1,int(maximum*f)) for f in config['base_rank_fractions'])
        sparse_ranks.add(maximum)
        for q in SMALL_CAPS.values():
            count=int(np.floor(q*reference_cost.parameters))
            fits_cap=[r for r in feasible if budget.allows(Cost(rank_costs[r].parameters+count,
                                                              rank_costs[r].payload_bytes+12*count))]
            if fits_cap:sparse_ranks.add(max(fits_cap))
    weights={s:training_weights(train_target,teacher_train,dense.task,s,config['margin_scale'])
             for s in config['tiny_selectors']}
    for rank in sorted(sparse_ranks):
        base=base_at_rank(dense,mean,basis,rank,full);bases[rank]=base;cost=rank_costs[rank]
        residual=(contrast(dense.coefficients,dense.task)
                  -contrast(base.coefficients.astype(float)@base.basis.T,dense.task)).astype(np.float32)
        error=teacher_train-contrast(latents['train'][:,:rank]@base.basis.T+base.bias,dense.task)
        for selector in config['tiny_selectors']:
            scores=coordinate_scores(designs['train'],residual,error,selector,weights[selector])
            order=score_order(scores,residual);orders[(rank,selector)]=order
            counts={min(len(order),residual_capacity(cost,b),int(np.floor(q*reference_cost.parameters*f)))
                    for b in budgets if b.allows(cost) for q in SMALL_CAPS.values()
                    for f in config['residual_capacity_fractions']}
            for count in sorted(counts-{0}):
                flat=order[:count];rr=(flat//c).astype(np.int32);cc=(flat%c).astype(np.int32)
                values=residual.ravel()[flat]
                correction=sparse.coo_matrix((values,(rr,cc)),shape=(m,c)).tocsr()
                shift=np.bincount(cc,weights=values.astype(float)*frequency[rr],minlength=c)
                bias=(base.bias.astype(float)-shift).astype(np.float32)
                raw=latents['selection'][:,:rank]@base.basis.T+bias+(designs['selection']@correction).toarray()
                rows.append({'candidate_id':f'r{rank}-{selector}-{count}','rank':rank,'nnz':count,
                    'parameters':cost.parameters+count,'payload_bytes':cost.payload_bytes+12*count,
                    'kind':'sparse','selector':selector,'fit':None,'decision_shifts':[],
                    'validation':calibrated_quality(selection_target,raw,dense.task,EMPTY)})
        if progress:progress('sparse',rank,len(rows))
    selections=[]
    for budget in budgets:
        for family in FAMILIES:
            candidates=[r for r in rows if eligible_family(r,family,reference_cost.parameters)]
            chosen,passed=select_candidate(candidates,reference_quality,budget)
            selections.append({'budget':asdict(budget),'family':family,
                'residual_fraction_cap':SMALL_CAPS.get(family),'validation_quality_pass':passed,
                'candidate':chosen.copy()})
    return {'rows':rows,'selections':selections,'bases':bases,'full':full,'orders':orders,'frequency':frequency,
            'validation_reference':reference_quality,'calibration_fits':fits}


def materialize_tiny(chosen,plan,dense,mean,basis):
    rank=chosen['rank'];base=plan['bases'].get(rank)
    if base is None:base=base_at_rank(dense,mean,basis,rank,plan['full'])
    if chosen['kind']=='affine':
        base=fold_affine(base,chosen['fit']['scale'],chosen['fit']['shift'])
    if chosen['nnz']:
        residual=(contrast(dense.coefficients,dense.task)
                  -contrast(base.coefficients.astype(float)@base.basis.T,dense.task)).astype(np.float32)
        leaf=make_sparse_model(base,residual,plan['orders'][(rank,chosen['selector'])],chosen['nnz'],plan['frequency'])
    else:
        leaf=SparseLeafModel(base,np.empty(0,np.int32),np.empty(0,np.int32),EMPTY.copy())
    return DecisionLeafModel(leaf,np.array(chosen['decision_shifts'],dtype=np.float32))


def prepare_tiny(name,seed,config,output,data):
    x,y,source_indices,metadata=data;started=time.perf_counter()
    source=Path(config['source_dir'])/f'{name}_{seed}';directory=output/f'{name}_{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    with np.load(source/'splits.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(source_indices,f['source_indices'])
        train,validation=f['train'].copy(),f['validation'].copy()
    dense=LeafEnsemble.load(source/'dense_teacher.npz');reference=LeafEnsemble.load(source/'reference.npz')
    with np.load(source/'calibration_diagnostics.npz',allow_pickle=False) as f:
        mean,basis=f['mean'].copy(),f['basis'].copy()
    a,b=calibration_partition(y[validation],dense.task,len(dense.bias),seed)
    calibration,selection=validation[a],validation[b]
    split_path=directory/'development_splits.npz'
    np.savez(split_path,train=train,calibration=calibration,selection=selection)
    leaves={k:dense.leaf_indices(x[v]) for k,v in [('train',train),('calibration',calibration),('selection',selection)]}
    np.testing.assert_allclose(dense.contributions(leaves['train']).mean(axis=0),mean,atol=1e-12,rtol=0)
    plan=tiny_plan(dense,mean,basis,leaves['train'],y[train],leaves['calibration'],y[calibration],
        leaves['selection'],y[selection],Cost(reference.parameter_count,reference.payload_bytes),config,
        lambda stage,rank,count:emit('tiny_preparing',dataset=name,seed=seed,stage=stage,rank=rank,candidates=count))
    design=leaf_design(leaves['selection'],len(dense.coefficients));saved={}
    for selected in plan['selections']:
        chosen=selected['candidate'];cid=chosen['candidate_id']
        if cid not in saved:
            model=materialize_tiny(chosen,plan,dense,mean,basis)
            assert model.parameter_count==chosen['parameters'] and model.payload_bytes==chosen['payload_bytes']
            path=directory/f'{cid}.npz';model.save(path);loaded=DecisionLeafModel.load(path)
            np.testing.assert_array_equal(loaded.raw_from_design(design),model.raw_from_design(design))
            q=calibrated_quality(y[selection],loaded.raw_from_design(design),dense.task,loaded.decision_shifts)
            for key in q:
                if abs(q[key]-chosen['validation'][key])>1e-7:raise RuntimeError('serialized selection changed')
            saved[cid]={'artifact':str(path.relative_to(output)).replace('\\','/'),
                        'sha256':sha256(path),'file_bytes':path.stat().st_size}
        selected.update(saved[cid])
    fits=plan['calibration_fits']
    record={'dataset':name,'seed':seed,'task':dense.task,'data_sha256':metadata['sha256'],
        'development_split_sha256':sha256(split_path),'calibration_rows':len(calibration),'selection_rows':len(selection),
        'reference_cost':{'parameters':reference.parameter_count,'payload_bytes':reference.payload_bytes},
        'validation_reference':plan['validation_reference'],'selections':plan['selections'],
        'validation_candidates':plan['rows'],'calibration_optimization':{'fits':len(fits),
            'converged':sum(f['converged'] for f in fits),'max_iterations':max(f['iterations'] for f in fits),
            'objective_worsenings':sum(f['final_objective']>f['initial_objective']+1e-9 for f in fits)},
        'prepare_seconds':time.perf_counter()-started}
    write_report(directory/'selection.json',record)
    emit('tiny_split_sealed',dataset=name,seed=seed,candidates=len(plan['rows']),saved_models=len(saved),
         decisions=len(plan['selections']),calibration_optimization=record['calibration_optimization'],seconds=record['prepare_seconds'])
    return {k:v for k,v in record.items() if k!='validation_candidates'}


def test_tiny(record,config,output,data):
    x,y,_,_=data;name,seed=record['dataset'],record['seed'];source=Path(config['source_dir'])/f'{name}_{seed}'
    with np.load(source/'splits.npz',allow_pickle=False) as f:test=f['test'].copy()
    dense=LeafEnsemble.load(source/'dense_teacher.npz')
    design=leaf_design(dense.leaf_indices(x[test]),len(dense.coefficients))
    reference=calibrated_quality(y[test],design@dense.coefficients+dense.bias,dense.task,EMPTY)
    records=[];cache={}
    for selected in record['selections']:
        path=output/selected['artifact']
        if sha256(path)!=selected['sha256']:raise RuntimeError('sealed model changed')
        if selected['artifact'] not in cache:
            model=DecisionLeafModel.load(path)
            cache[selected['artifact']]=calibrated_quality(y[test],model.raw_from_design(design),dense.task,model.decision_shifts)
        q=cache[selected['artifact']];chosen=selected['candidate']
        assert Budget(**selected['budget']).allows(Cost(chosen['parameters'],chosen['payload_bytes']))
        if selected['family'] in SMALL_CAPS:
            assert chosen['nnz']<=int(np.floor(SMALL_CAPS[selected['family']]*record['reference_cost']['parameters']))
        records.append({'dataset':name,'seed':seed,'task':dense.task,**selected,'test':q,'test_reference':reference,
            'test_quality_pass':quality_guard(q,reference),
            'joint_success':selected['validation_quality_pass'] and quality_guard(q,reference),
            'macro_retained':q['f1_macro']/reference['f1_macro'],'loss_ratio':q['loss']/reference['loss'],
            'actual_residual_fraction':chosen['nnz']/record['reference_cost']['parameters'],
            'parameter_reduction':1-chosen['parameters']/record['reference_cost']['parameters'],
            'payload_reduction':1-chosen['payload_bytes']/record['reference_cost']['payload_bytes']})
    emit('tiny_test_finished',dataset=name,seed=seed,decisions=len(records))
    return records


def summarize_tiny(records):
    summaries=[]
    for dataset,kind,fraction in sorted({(r['dataset'],r['budget']['kind'],r['budget']['fraction']) for r in records}):
        group=[r for r in records if (r['dataset'],r['budget']['kind'],r['budget']['fraction'])==(dataset,kind,fraction)]
        controls={f:{r['seed']:r for r in group if r['family']==f} for f in ('rank_only','calibration')}
        for family in FAMILIES:
            rows=[r for r in group if r['family']==family]
            summary={'dataset':dataset,'kind':kind,'fraction':fraction,'family':family,'splits':len(rows),
                'quality_passes':sum(r['test_quality_pass'] for r in rows),'joint_successes':sum(r['joint_success'] for r in rows),
                'nonzero_residual_models':sum(r['candidate']['nnz']>0 for r in rows),
                'median_macro_retained':float(np.median([r['macro_retained'] for r in rows])),
                'median_parameter_reduction':float(np.median([r['parameter_reduction'] for r in rows])),
                'median_payload_reduction':float(np.median([r['payload_reduction'] for r in rows]))}
            for control,lookup in controls.items():
                deltas=[r['test']['f1_macro']-lookup[r['seed']]['test']['f1_macro'] for r in rows]
                summary[f'mean_macro_delta_vs_{control}']=float(np.mean(deltas))
                summary[f'wins_ties_losses_vs_{control}']=[sum(d>1e-12 for d in deltas),sum(abs(d)<=1e-12 for d in deltas),sum(d < -1e-12 for d in deltas)]
            summaries.append(summary)
    return summaries


def tiny_residual_study(config,report,output):
    def expired(signum,frame):raise TimeoutError('one-hour approved CPU comparison exhausted')
    previous=signal.signal(signal.SIGALRM,expired);signal.alarm(config['walltime_seconds']);started=time.perf_counter()
    report.update(status='running',training_fits=0,gpu_runs=0,
        teacher_fits=0,model_cost='persistent stored factors+bias+tree arrays+COO indices+decision vector; affine folded by QR; calibration fit scalars not retained',
        interpretation='Exploratory comparison on previously inspected tests; no fresh data.',
        selection_policy='coordinate scores train-only; calibration fits calibration-half-only; choices selection-half-only; all sealed before test scoring')
    try:
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_output_calibration_core.py',
                        'tests/test_sparse_leaf_core.py','tests/test_compactness_core.py'],cwd=ROOT,check=True)
        manifest=json.loads((ROOT/'experiments/sparse_source_manifest.json').read_text())
        verify_sources(Path(config['source_dir']),manifest)
        datasets={name:read_dataset(name,config) for name in config['datasets']}
        selections=[]
        for name in config['datasets']:
            for seed in config['seeds_by_dataset'][name]:
                selections.append(prepare_tiny(name,seed,config,output,datasets[name]));gc.collect()
        seal_path=output/'selection_manifest.json'
        write_report(seal_path,{'source_run':manifest['source_run'],'selections':selections,'test_labels_used_in_selection':False})
        seal=sha256(seal_path)
        emit('tiny_all_selections_sealed',sha256=seal,decisions=sum(len(s['selections']) for s in selections),test_scoring_started=False)
        records=[]
        for selection in selections:
            if sha256(seal_path)!=seal:raise RuntimeError('selection manifest changed')
            records.extend(test_tiny(selection,config,output,datasets[selection['dataset']]))
        summaries=summarize_tiny(records)
        report.update(status='complete',records=records,summaries=summaries,selection_manifest_sha256=seal,
            elapsed_seconds=time.perf_counter()-started,source_manifest_sha256=sha256(ROOT/'experiments/sparse_source_manifest.json'))
        write_report(output/'report.json',report)
        emit('tiny_comparison_summary',training_fits=0,gpu_runs=0,decisions=len(records),elapsed_seconds=report['elapsed_seconds'],summaries=summaries)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-started)
        write_report(output/'report.json',report);raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
