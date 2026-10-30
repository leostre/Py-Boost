"""One native protected-increment trajectory with an explicit paid fit ledger."""
from dataclasses import replace
from pathlib import Path
import gc,json,sys
from time import perf_counter
import numpy as np
from budget_quality_worker import locked,write,digest,event,native_load
from abl_training_increment import fit_training_increment
from abl_protected_core import as_protected,append_protected,block_cost,project_new_increment,load_protected
from abl_transfer_study import data_before_test
from abl_training_worker import cpu_peak
from experiments.shared_leaf_basis import reference_class_model,probabilities
from experiments.parallel_leaf_core import prefix
from experiments.absolute_history_core import AbsoluteHistory


def canonical(model):
    return replace(model,base=reference_class_model(model.base)) if model.base.task=='multiclass' else model


def save_snapshot(model,path,uid,block,extra):
    model.save(path);price=block_cost(model)
    return dict(uid=uid,artifact=str(path),sha256=digest(path),block=block,
        ranks=[b.base.basis.shape[1] or None for b in model.blocks],**price,**extra)


def fit(request,output):
    output.mkdir(parents=True,exist_ok=True);data=data_before_test(request['record']);task=request['record']['case']['task']
    c=request['record']['case']['outputs'];cid=request['case_id'];mode=request['mode'];cfg=request['training_config']
    ncal,nctrl=data['sizes'][:2];development=dict(train=(data['train_x'],data['train_y']),
        calibration=(data['dev_x'][:ncal],data['dev_y'][:ncal]),
        control=(data['dev_x'][ncal:ncal+nctrl],data['dev_y'][ncal:ncal+nctrl]))
    work=output/'worker_data.npz';np.savez(work,train_x=development['train'][0],train_y=development['train'][1],
        validation_x=development['calibration'][0],validation_y=development['calibration'][1])
    ledger_path=Path(request['ledger_path']);cycles=[];models=[];started=perf_counter()
    def build(block,initial_model,prior=None):
        directory=output/f'block{block:02d}';directory.mkdir(exist_ok=True)
        initial=directory/'initial.npz'
        train=(np.broadcast_to(prior,(len(data['train_y']),c)).copy() if initial_model is None else initial_model.raw(data['train_x']))
        cal=(np.broadcast_to(prior,(ncal,c)).copy() if initial_model is None else initial_model.raw(data['dev_x'][:ncal]))
        np.savez(initial,train=train.astype(np.float32),validation=cal)
        ledger=json.loads(ledger_path.read_text())
        if ledger['reserved_rounds']+40>min(8000,ledger['phase_limit']):raise RuntimeError('protected fit cap exhausted')
        ledger['reserved_rounds']+=40;index=len(ledger['attempts']);ledger['attempts'].append(dict(case=cid,mode=mode,block=block,rounds=40,state='reserved'))
        write(ledger_path,ledger)
        fitted=fit_training_increment(dict(data_path=str(work),initial_path=str(initial),config={**cfg,'max_depth':3},
            task=task,seed=request['record']['case']['seed']+50000+40*block,trees=40,lr=cfg['lr'],subsample=1.,colsample=1.,
            minimum_leaf=10,output=str(directory/'increment.npz'),minimum_free_mib=8192,own_pool_limit_bytes=16*1024**3))
        increment=native_load(fitted['artifact']);ledger=json.loads(ledger_path.read_text());ledger['attempts'][index].update(state='complete',fit=fitted)
        write(ledger_path,ledger)
        if prior is not None:increment=replace(increment,base=replace(increment.base,bias=np.asarray(prior,np.float32)))
        return canonical(increment),dict(block=block,fit=fitted,initial_path=str(initial),initial_sha256=digest(initial),
            initial_model_sha256=None if initial_model is None else models[-1]['sha256'])
    if mode=='warmup':
        prior=np.zeros(c,np.float32)
        if task=='multilabel':
            p=np.clip(data['train_y'].mean(0),1e-7,1-1e-7);prior=np.log(p/(1-p)).astype(np.float32)
        increment,cycle=build(0,None,prior);model=as_protected(increment)
        item=save_snapshot(model,output/'warmup.npz','warmup/b00',0,dict(compressed_blocks=0))
        threshold=output/'warmup_threshold.npz';increment.save(threshold)
        result=dict(status='complete',case=cid,mode=mode,models=[item],cycles=[cycle],
            warmup_threshold=dict(artifact=str(threshold),sha256=digest(threshold)),elapsed_seconds=perf_counter()-started)
        write(output/'trajectory.json',result);return
    warm=locked(request['warmup']['path'],request['warmup']['sha256']);warm_item=warm['models'][0]
    if digest(warm_item['artifact'])!=warm_item['sha256'] or digest(warm['warmup_threshold']['artifact'])!=warm['warmup_threshold']['sha256']:
        raise RuntimeError('shared warmup source changed')
    checkpoint=load_protected(warm_item['artifact']);models=[dict(warm_item,uid=mode+'/b00')]
    history=None;estimate=None
    if mode in ('current','mean','history'):
        history=AbsoluteHistory(mode,c,task,.8);teacher=native_load(warm['warmup_threshold']['artifact'])
        prior=teacher.base.bias.copy()
        # Temporal rows remain aligned. Four actual warmup prefixes and the
        # initial score give history something to accumulate before compression.
        for trees in (0,10,20,30,40):
            scores=np.broadcast_to(prior,(len(data['train_y']),c)).copy() if not trees else prefix(teacher,trees).prepare_inference().raw(data['train_x'])
            estimate=history.observe(data['train_y'],scores,cfg['lambda_l2'],max(1,(c-int(task=='multiclass'))//2))
    stop='horizon';compressed=0
    for block in range(1,5):
        increment,cycle=build(block,checkpoint)
        if mode=='full':
            accepted=append_protected(checkpoint,increment);info=dict(status='ordinary')
        else:
            accepted,info=project_new_increment(checkpoint,increment,development,request['budget_bytes'],cfg,estimate)
        cycle.update(projection=info,history_diagnostics=None if estimate is None else estimate.diagnostics,
            protected_initial_blocks=len(checkpoint.blocks),initial_model_sha256=models[-1]['sha256'],
            algorithm_state_bytes=0 if estimate is None else estimate.diagnostics['algorithm_state_bytes'],
            diagnostic_state_bytes=0 if estimate is None else estimate.diagnostics['diagnostic_state_bytes'],
            previous_basis_bytes=0 if estimate is None else estimate.diagnostics['previous_basis_bytes'],
            cpu_process_peak_rss_bytes=cpu_peak())
        if history is not None:
            path=output/f'block{block:02d}'/'history_before_projection.npz'
            state=history.state
            np.savez(path,mean=np.empty((0,0)) if state.mean is None else state.mean,
                gram=np.empty((0,0)) if state.gram is None else state.gram,
                halves=np.empty((0,0,0)) if state.half_grams is None else np.asarray(state.half_grams),
                mass=np.array(state.mass),observations=np.array(state.observations),basis=estimate.basis)
            cycle.update(history_state_path=str(path),history_state_sha256=digest(path))
        if accepted is None:
            cycle['action']='rejected';cycles.append(cycle);stop=info['status'];break
        cycle['action']='ordinary' if mode=='full' else 'projected_increment'
        if mode!='full':compressed+=1
        model_path=output/f'block{block:02d}'/'accepted.npz'
        item=save_snapshot(accepted,model_path,mode+f'/b{block:02d}',block,dict(compressed_blocks=compressed))
        if mode!='full' and item['cost_bytes']>request['budget_bytes']:raise RuntimeError('protected checkpoint exceeds budget')
        checkpoint=accepted;models.append(item);cycle.update(artifact=str(model_path),sha256=item['sha256']);cycles.append(cycle)
        if history is not None:estimate=history.observe(data['train_y'],checkpoint.raw(data['train_x']),cfg['lambda_l2'],max(1,(c-int(task=='multiclass'))//2))
        write(output/'trajectory.json',dict(status='in_progress',case=cid,mode=mode,models=models,cycles=cycles))
        event('abl_protected_checkpoint',case=cid,mode=mode,block=block,cost_bytes=item['cost_bytes'],compressed_blocks=compressed)
        del increment,accepted;gc.collect()
    if max((v['cpu_process_peak_rss_bytes'] or 0 for v in cycles),default=0)>32*1024**3:raise MemoryError('protected CPU limit exceeded')
    write(output/'trajectory.json',dict(status='complete',case=cid,mode=mode,models=models,cycles=cycles,stop_reason=stop,
        elapsed_seconds=perf_counter()-started,controller_used_roles=['train','calibration','control'],test_accessed=False,
        warmup=request['warmup'],gradient_source='actual protected-block checkpoint; full temporary derivatives remain'))
    event('abl_protected_trajectory_complete',case=cid,mode=mode,accepted_models=len(models),stop_reason=stop)


def main():
    request=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):fit(request,output)


if __name__=='__main__':main()
