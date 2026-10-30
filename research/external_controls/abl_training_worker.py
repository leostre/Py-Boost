"""One compact-training trajectory in the pinned native interpreter."""
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from budget_quality_worker import digest,write,event,native_load
from abl_quality_worker import before_test_data
from abl_quality_core import array_ledger,full_output_base
from abl_training_core import reserved_depth
from abl_training_projection import training_proposal
from abl_training_increment import fit_training_increment
from experiments.closed_budget_study import combine_actual,empty_model
from experiments.closed_budget_core import Topology
from experiments.cumulative_budget_core import CumulativeError
from experiments.shared_leaf_basis import reference_class_model
from budget_quality_core import probabilities


def cpu_peak():
    if sys.platform=='win32':return None
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def canonical_full(model):
    return empty_model(reference_class_model(full_output_base(model)))


def cost(model):
    ledger=array_ledger(model)
    return ledger['cost_bytes'],max(ledger['coo_bytes'],ledger['prepared_bytes'])


def initial_scores(model,x,outputs):
    return np.zeros((len(x),outputs),np.float64) if model is None else model.prepare_inference().raw(x)


def reserve_fit(path,case_id,mode,block,depth):
    ledger=json.loads(path.read_text())
    if ledger['reserved_rounds']+40>min(21600,ledger.get('phase_round_limit',21600)):
        raise RuntimeError('T1 global/phase tree cap exhausted')
    ledger['reserved_rounds']+=40
    ledger['attempts'].append(dict(case=case_id,mode=mode,block=block,depth=depth,state='reserved',rounds=40))
    write(path,ledger)
    return len(ledger['attempts'])-1


def completed_fit(path,index,fit,leaf_values):
    ledger=json.loads(path.read_text());ledger['attempts'][index].update(state='complete',fit=fit,
        fitted_full_output_leaf_values=int(leaf_values))
    write(path,ledger)


def snapshot(model,directory,name,case_id,mode,block,extra):
    directory.mkdir(parents=True,exist_ok=True);path=directory/(name+'.npz');model.save(path)
    ledger=array_ledger(model)
    return dict(uid=f'{mode}/b{block:02d}',family=mode,artifact=str(path),sha256=digest(path),
        case_id=case_id,block=block,rounds=model.base.n_trees,rank=model.base.basis.shape[1] or None,
        nnz=len(model.values),cost_bytes=ledger['cost_bytes'],ledger=ledger,
        leaves=len(model.base.coefficients),internal_nodes=len(model.base.nodes),
        representation=model.base.representation,**extra)


def train_path(request,output):
    record=request['record'];case=record['case'];case_id=request['case_id'];mode=request['mode']
    if mode not in ('full','fixed','joint'):raise ValueError('unknown T1 training mode')
    if request['hyper']['depth'] not in (3,5) or request['hyper']['minimum_leaf'] not in (10,50):
        raise ValueError('frozen T1 topology choice required')
    budget=request['budget_bytes'];envelope=budget-4;c=case['outputs'];ceiling=c-1
    cfg=request['training_config'];output.mkdir(parents=True,exist_ok=True)
    locks={};data=before_test_data(record,locks);ncal,nctrl=data['sizes'][:2]
    development=dict(train=(data['train_x'],data['train_y']),
        calibration=(data['dev_x'][:ncal],data['dev_y'][:ncal]),
        control=(data['dev_x'][ncal:ncal+nctrl],data['dev_y'][ncal:ncal+nctrl]))
    # The native fitter never receives selection/test features or targets.
    work=output/'worker_data.npz'
    np.savez(work,train_x=development['train'][0],train_y=development['train'][1],
        validation_x=development['calibration'][0],validation_y=development['calibration'][1])
    error=CumulativeError(np.zeros((len(data['train_y']),c)),'multiclass')
    checkpoint=None;models=[];posthoc=[];cycles=[];started=perf_counter();stop_reason='horizon'
    minimum=max(1,math.ceil(ceiling*.25));maximum_depth=request['hyper']['depth']
    horizon=800
    if mode=='joint':
        horizon=0
        for candidate in (800,400,200,80,40):
            d=reserved_depth(Topology(0,0,0),candidate,c,minimum,envelope)
            if d is not None and d<=maximum_depth:horizon=candidate;break
            if maximum_depth==3:
                # The generic helper prefers5 when both fit;3 also fits then.
                from experiments.closed_budget_core import extend_bound
                from experiments.cumulative_budget_core import serving_cost
                if serving_cost(extend_bound(Topology(0,0,0),candidate,3),c,minimum,0).maximum_bytes<=envelope:
                    horizon=candidate;break
        if not horizon:stop_reason='no_affordable_minimum_horizon'
    for block in range(horizon//40):
        topology=Topology(0,0,0) if checkpoint is None else Topology(checkpoint.base.n_trees,
            len(checkpoint.base.nodes),len(checkpoint.base.coefficients))
        depth=maximum_depth
        if mode=='joint':
            depth=reserved_depth(topology,horizon-topology.trees,c,minimum,envelope)
            if depth is not None and depth>maximum_depth:depth=maximum_depth
            if depth is None:stop_reason='future_topology_reserve_exhausted';break
        blockdir=output/f'block{block:02d}';blockdir.mkdir(parents=True,exist_ok=True)
        before_kind=None if checkpoint is None else checkpoint.base.representation
        source_hash=None if checkpoint is None else models[-1]['sha256']
        initial=blockdir/'initial.npz'
        train_initial=initial_scores(checkpoint,development['train'][0],c)
        cal_initial=initial_scores(checkpoint,development['calibration'][0],c)
        np.savez(initial,train=train_initial.astype(np.float32),validation=cal_initial)
        warmup=request.get('warmups',{}).get(str(depth)) if block==0 else None
        fit=None;reused=False
        if warmup is not None:
            if digest(warmup['artifact'])!=warmup['sha256']:raise RuntimeError('frozen first40 source changed')
            increment=native_load(warmup['artifact']);reused=True
            if increment.base.n_trees!=40:raise RuntimeError('warmup must contain exactly40trees')
        else:
            index=reserve_fit(Path(request['ledger_path']),case_id,mode,block,depth)
            fit=fit_training_increment(dict(data_path=str(work),initial_path=str(initial),
                config={**cfg,'max_depth':depth},task='multiclass',seed=case['seed']+40000+40*block,
                trees=40,lr=cfg['lr'],subsample=1.,colsample=1.,minimum_leaf=request['hyper']['minimum_leaf'],
                output=str(blockdir/'increment.npz'),minimum_free_mib=8192,own_pool_limit_bytes=16*1024**3))
            increment=native_load(blockdir/'increment.npz')
            completed_fit(Path(request['ledger_path']),index,fit,len(increment.base.coefficients)*c)
        proposal=combine_actual(checkpoint,increment,
            increment.base.bias.copy() if reused else np.zeros(c,np.float32))
        probe=np.unique(np.r_[np.arange(min(64,len(data['train_y']))),
            [np.flatnonzero(data['train_y']==label)[0] for label in np.unique(data['train_y'])]])
        expected=train_initial[probe]+increment.prepare_inference().raw(data['train_x'][probe])
        actual=proposal.prepare_inference().raw(data['train_x'][probe])
        proposal_error=float(np.max(np.abs(probabilities(expected)-probabilities(actual))))
        if proposal_error>2e-5:raise RuntimeError('actual-checkpoint composition changed probability beyond2e-5')
        ordinary=canonical_full(proposal);price,construction=cost(ordinary)
        projection=None;projection_seconds=0.;accepted=None;next_error=error;action='ordinary'
        t=perf_counter()
        if mode=='full':accepted=ordinary
        elif mode=='joint' and price<=budget:
            accepted=ordinary
            before=initial_scores(proposal,development['train'][0],c)
            after=initial_scores(accepted,development['train'][0],c);next_error=error.advance(before,after)
        else:
            accepted,projection,next_error=training_proposal(full_output_base(proposal),development,
                envelope,error,cfg,fixed=mode=='fixed',criterion='local' if mode=='fixed' else 'cumulative')
            action='projection' if accepted is not None else 'projection_rejected'
        projection_seconds=perf_counter()-t
        cycle=dict(block=block,depth=depth,reused_first40=reused,fit=fit,
            initial_from_representation=before_kind,initial_checkpoint_sha256=source_hash,
            initial_sha256=digest(initial),action=action,projection=projection,
            projection_seconds=projection_seconds,full_proposal_bytes=price,
            proposal_probability_replay_error=proposal_error,
            error_state_bytes=error.values.nbytes,full_output_statistic_width=c,
            cpu_process_peak_rss_bytes=cpu_peak())
        if accepted is None:
            cycles.append(cycle);write(blockdir/'cycle.json',cycle);stop_reason='local_quality_or_capacity_rejected';break
        accepted_price,accepted_envelope=cost(accepted)
        if mode!='full' and (accepted_price>budget or accepted_envelope>envelope):
            raise RuntimeError('accepted training checkpoint exceeded hard byte budget')
        item=snapshot(accepted,blockdir,'accepted',case_id,mode,block,dict(
            local_temperature=(projection['chosen'].get('temperature',1.) if projection else 1.),
            budget_bytes=None if mode=='full' else budget))
        models.append(item);checkpoint=accepted;error=next_error
        error_path=blockdir/'error_state.npz'
        np.savez_compressed(error_path,error=error.values,updates=np.array(error.updates))
        cycle.update(artifact=item['artifact'],sha256=item['sha256'],accepted_bytes=accepted_price,
            rank=item['rank'],nnz=item['nnz'],trees=item['rounds'],
            error_state_path=str(error_path),error_state_sha256=digest(error_path))
        if mode=='full':
            t=perf_counter();post_model=ordinary;post_info=None
            if price>budget:
                zero=CumulativeError(np.zeros_like(error.values),'multiclass')
                post_model,post_info,_=training_proposal(full_output_base(proposal),development,
                    envelope,zero,cfg,fixed=False,criterion='local')
            cycle['posthoc_projection_seconds']=perf_counter()-t
            cycle['posthoc_projection']=post_info
            if post_model is not None:
                post_price,post_envelope=cost(post_model)
                if post_price>budget or post_envelope>envelope:raise RuntimeError('posthoc model exceeded budget')
                posthoc.append(snapshot(post_model,blockdir,'posthoc',case_id,'posthoc',block,
                    dict(local_temperature=(post_info['chosen'].get('temperature',1.) if post_info else 1.),
                        ordinary_fallback=post_info is None,budget_bytes=budget)))
        cycles.append(cycle);write(blockdir/'cycle.json',cycle)
        # Persist only the current working checkpoint in RAM, all history on disk.
        result=dict(status='in_progress',case_id=case_id,mode=mode,models=models,posthoc=posthoc,
            cycles=cycles,planned_horizon=horizon,stop_reason=stop_reason,config=request)
        write(output/'trajectory.json',result)
        event('abl_training_checkpoint',case=case_id,mode=mode,block=block,depth=depth,
            action=action,rounds=item['rounds'],cost_bytes=accepted_price,rank=item['rank'],nnz=item['nnz'])
        del proposal,ordinary,increment,train_initial,cal_initial;gc.collect()
    for path,sha in locks.items():
        if digest(path)!=sha:raise RuntimeError('training source changed')
    if max((r['cpu_process_peak_rss_bytes'] or 0 for r in cycles),default=0)>32*1024**3:
        raise MemoryError('native trajectory exceeded32GiB CPU working limit')
    result=dict(status='complete',case_id=case_id,mode=mode,models=models,posthoc=posthoc,cycles=cycles,
        planned_horizon=horizon,stop_reason=stop_reason,config=request,input_locks=locks,
        elapsed_seconds=perf_counter()-started,projection_events=sum(r['action']=='projection' for r in cycles),
        blocks_initialized_from_compact=sum(r['initial_from_representation']=='low_rank' for r in cycles),
        gradient_source='actual persisted unscaled checkpoint; no hidden historical full model',
        controller_used_roles=['train','calibration','control'],selection_and_test_used_by_controller=False)
    write(output/'trajectory.json',result)
    event('abl_training_trajectory_complete',case=case_id,mode=mode,checkpoints=len(models),
        stop_reason=stop_reason,sha256=digest(output/'trajectory.json'))


def main():
    request=json.loads(Path(sys.argv[1]).read_text())
    with threadpool_limits(limits=1):train_path(request,Path(sys.argv[2]))


if __name__=='__main__':main()
