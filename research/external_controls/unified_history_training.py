"""U3: history x real branch-process factorial, protected old function."""
from pathlib import Path
from dataclasses import replace
import gc,hashlib,json,subprocess,sys,time,os,signal
import numpy as np
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,as_protected,append_protected,block_cost
from abl_training_increment import fit_training_increment
from abl_budget_revision_core import decode_model,blank
from unified_training_core import candidate_grid,merge_increment_views,canonical
from unified_training import evaluate,serialize,quality_row
from unified_recalibration import audit_case
from unified_frontier_core import make_folds,frontier
from unified_history_core import observe_branches,replay_moments
from unified_data_core import expanded_record
from experiments.absolute_history_core import AbsoluteHistory
from experiments.parallel_leaf_core import prefix
from experiments.shared_leaf_basis import functional_basis
from experiments.gradient_basis_core import projector_overlap

METHODS=(('one_full','current',1,1.),('one_functional','current',1,1.),('one_current','current',1,1.),('one_mean','mean',1,1.),
    ('one_history','history',1,1.),('two_current_sum','current',2,1.),('two_history_sum','history',2,1.),
    ('two_current_average','current',2,.5),('two_history_average','history',2,.5))


def save_history(history,estimate,scores,path):
    s=history.state
    np.savez_compressed(path,scores=scores,mean=np.empty((0,0)) if s.mean is None else s.mean,
        gram=np.empty((0,0)) if s.gram is None else s.gram,
        mass=np.array(s.mass),observations=np.array(s.observations),basis=estimate.basis)
    return dict(path=str(path),sha256=digest(path),diagnostics=estimate.diagnostics)


def fit_child(request_path):
    request=json.loads(Path(request_path).read_text());result=fit_training_increment(request)
    write(request['result_path'],result)


def run(cfg,out):
    started=time.perf_counter();environment=check_native_environment(cfg);spec=cfg['unified_history']
    prerequisite=locked(spec['u2']['path'],spec['u2']['sha256']);assert prerequisite['status']=='unified_training_complete'
    source=locked(cfg['absolute_roles_manifest'],cfg['absolute_roles_manifest_sha256'])
    old=locked(cfg['abl_protected_development_completed']['path'],cfg['abl_protected_development_completed']['sha256'])
    cases={r['case_id']:locked(r['path'],r['sha256']) for r in old['cases']}
    native=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    out.mkdir(parents=True,exist_ok=True);ledger=[];paid=0;cache={};entries=[];completed={}
    warm_cases={}
    if spec.get('warm_completed'):
        prior=locked(spec['warm_completed']['path'],spec['warm_completed']['sha256'])
        assert prior['status']=='unified_training_complete'
        warm_cases={r['case']:locked(r['path'],r['sha256']) for r in prior['cases']}
    if spec.get('prior_completed'):
        prior=locked(spec['prior_completed']['path'],spec['prior_completed']['sha256'])
        assert prior['status']=='unified_history_complete'
        completed={r['case']:locked(r['path'],r['sha256']) for r in prior['cases']};paid=prior['paid_new_trees']
        ledger=list(locked(prior['ledger']['path'],prior['ledger']['sha256'])['attempts'])
        for attempt in ledger:
            fit=attempt.get('fit')
            if not fit or 'initial_path' not in fit:continue
            assert digest(fit['artifact'])==fit['sha256']
            with np.load(fit['initial_path'],allow_pickle=False) as f:tr,dv=f['train'].copy(),f['validation'].copy()
            key=hashlib.sha256(tr.tobytes()+dv.tobytes()+str((attempt['case'],fit['seed'],fit['trees'])).encode()).hexdigest()
            cache[key]=fit
    write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger))
    for name,seed in spec['cases']:
        cid=f'{name}_{seed}';directory=out/cid;directory.mkdir(exist_ok=True)
        record=dict(case=next(c for c in source['cases'] if c['dataset']==name and c['seed']==seed),dataset=source['datasets'][name])
        if cid in completed:
            record=completed[cid]['record']
        elif cid in warm_cases:
            record=warm_cases[cid]['record']
        elif spec.get('reclaim_training_fraction'):
            record=expanded_record(record,directory,spec['reclaim_training_fraction'])
        data=data_before_test(record);task=record['case']['task'];c=record['case']['outputs'];rank=max(1,(c-int(task=='multiclass'))//2)
        data_path=directory/'fit_data.npz';np.savez(data_path,train_x=data['train_x'],train_y=data['train_y'],validation_x=data['dev_x'],validation_y=data['dev_y'])
        warm_fit=None
        if cid in completed:
            warm_ref=completed[cid]['warmup'];assert digest(warm_ref['artifact'])==warm_ref['sha256'];warm=load_protected(warm_ref['artifact'])
        elif cid in warm_cases:
            warm_ref=warm_cases[cid]['warmup'];assert digest(warm_ref['artifact'])==warm_ref['sha256'];warm=load_protected(warm_ref['artifact'])
        elif spec.get('reclaim_training_fraction'):
            prior=np.zeros(c,np.float32)
            if task=='multilabel':
                prob=np.clip(data['train_y'].mean(0),1e-7,1-1e-7);prior=np.log(prob/(1-prob)).astype(np.float32)
            initial=directory/'warm_initial.npz';np.savez(initial,train=np.broadcast_to(prior,(len(data['train_y']),c)).copy(),validation=np.broadcast_to(prior,(len(data['dev_y']),c)).copy())
            if paid+40>spec['maximum_new_trees']:raise RuntimeError('U3 warm work cap exhausted')
            paid+=40;ledger.append(dict(case=cid,method='shared_warm',trees=40,reused=False,status='reserved'))
            write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger))
            warm_fit=fit_training_increment(dict(data_path=str(data_path),initial_path=str(initial),config={**cfg,'max_depth':3},task=task,seed=seed+81000,trees=40,lr=cfg['lr'],subsample=1.,colsample=1.,minimum_leaf=10,output=str(directory/'warm_increment.npz'),minimum_free_mib=8192,own_pool_limit_bytes=16*1024**3))
            inc=native_load(warm_fit['artifact']);inc=replace(inc,base=replace(inc.base,bias=prior));warm=canonical(as_protected(inc))
            ledger[-1].update(status='complete',fit=warm_fit);write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger))
        else:
            warm_ref=cases[cid]['metas']['full/b00'];assert digest(warm_ref['artifact'])==warm_ref['sha256'];warm=load_protected(warm_ref['artifact'])
        warm_path=directory/'warmup.npz';warm.save(warm_path)
        view,_=decode_model(warm);warm_dense=blank(replace(view.base(),bias=view.bias.copy()))
        data_path=directory/'fit_data.npz';np.savez(data_path,train_x=data['train_x'],train_y=data['train_y'],validation_x=data['dev_x'],validation_y=data['dev_y'])
        features={'train':data['train_x'],'development':data['dev_x']};labels={'train':data['train_y']}
        folds=make_folds(data['dev_y'],task,seed+20261028);models=list(completed[cid]['models']) if cid in completed else [];paths=list(completed[cid]['paths']) if cid in completed else []
        for method,kind,branches,weight in METHODS:
            if any(p['method']==method for p in paths):continue
            work=directory/method;work.mkdir(exist_ok=True);checkpoint=warm;initial_checkpoint=str(warm_path)
            history=AbsoluteHistory(kind,c,task);states=[];steps=[]
            for t in (0,10,20,30,40):
                raw=np.broadcast_to(warm.bias,(len(data['train_y']),c)).copy() if t==0 else as_protected(prefix(warm_dense,t)).raw(data['train_x'])
                estimate,joined,_=observe_branches(history,data['train_y'],[raw]*branches,cfg['lambda_l2'],rank)
                states.append(save_history(history,estimate,joined,work/f'history-warm{t}.npz'))
            for block in range(4):
                step=work/f'b{block}';step.mkdir(exist_ok=True)
                tr=checkpoint.raw(data['train_x']).astype(np.float32);dv=checkpoint.raw(data['dev_x'])
                initial=step/'initial.npz';np.savez(initial,train=tr,validation=dv)
                increments=[None]*branches;fits=[None]*branches;running=[]
                for branch in range(branches):
                    local_seed=seed+92000+40*block+1000*branch;trees=40//branches
                    key=hashlib.sha256(tr.tobytes()+dv.tobytes()+str((cid,local_seed,trees)).encode()).hexdigest()
                    if key in cache:
                        fits[branch]=cache[key];increments[branch]=native_load(fits[branch]['artifact'])
                        ledger.append(dict(case=cid,method=method,block=block,branch=branch,trees=trees,reused=True,fit=fits[branch]));continue
                    if paid+trees>spec['maximum_new_trees']:raise RuntimeError('U3 work cap exhausted')
                    paid+=trees;request=dict(data_path=str(data_path),initial_path=str(initial),config={**cfg,'max_depth':3},
                        task=task,seed=local_seed,trees=trees,lr=cfg['lr'],subsample=.8,colsample=.8,minimum_leaf=10,
                        output=str(step/f'increment{branch}.npz'),minimum_free_mib=8192,own_pool_limit_bytes=16*1024**3,
                        result_path=str(step/f'fit{branch}.json'))
                    index=len(ledger);ledger.append(dict(case=cid,method=method,block=block,branch=branch,trees=trees,reused=False,status='reserved'))
                    write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger))
                    request_file=step/f'request{branch}.json';write(request_file,request)
                    log=(step/f'fit{branch}.log').open('w')
                    process=subprocess.Popen([native,__file__,'--fit',str(request_file)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    running.append((branch,key,index,request,process,log))
                # Both branch subprocesses have been dispatched before waiting.
                try:
                    for branch,key,index,request,process,log in running:
                        code=process.wait(timeout=3600);log.close()
                        if code:raise RuntimeError('owned native branch failed; see '+str(step/f'fit{branch}.log'))
                        result=json.loads(Path(request['result_path']).read_text());result['initial_path']=str(initial)
                        cache[key]=result;fits[branch]=result;increments[branch]=native_load(result['artifact'])
                        ledger[index].update(status='complete',fit=result)
                except BaseException:
                    for _,_,_,_,process,log in running:
                        if process.poll() is None:
                            os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                        log.close()
                    raise
                write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger))
                scores=[checkpoint.raw(data['train_x'])+as_protected(inc).raw(data['train_x']) for inc in increments]
                estimate,joined,_=observe_branches(history,data['train_y'],scores,cfg['lambda_l2'],rank)
                history_ref=save_history(history,estimate,joined,step/'history.npz');states.append(history_ref)
                merged,_=merge_increment_views(increments,[weight]*branches)
                if method=='one_full':
                    accepted=canonical(append_protected(checkpoint,merged));metadata=dict(rank=None,residual_fraction=0.)
                else:
                    proposals=list(candidate_grid(checkpoint,merged,features,labels,None,protected=True,
                        basis=None if method=='one_functional' else estimate.basis,ranks=(rank,),caps=(8,),residuals=(spec['residual_fraction'],)))
                    assert len(proposals)==1;accepted=proposals[0]['model'];metadata=dict(rank=rank,residual_fraction=spec['residual_fraction'])
                raw=accepted.raw(data['dev_x']);evaluations=evaluate(raw,data['dev_y'],task,folds,cfg,step)
                overlap=None
                if branches==2:
                    bases=[functional_basis(as_protected(inc).raw(data['train_x']),task)[1][:,:rank] for inc in increments]
                    overlap=projector_overlap(*bases)
                item=serialize(accepted,raw,evaluations,step,f'{method}/{block}',dict(family=method,block=block,
                    history_kind=kind,branches=branches,branch_weight=weight,fits=fits,initial_checkpoint=initial_checkpoint,
                    history=history_ref,branch_subspace_overlap=overlap,**metadata))
                models.append(item);steps.append(dict(uid=item['uid'],artifact=item['artifact'],sha256=item['sha256'],
                    quality=quality_row(evaluations,task)['quality'],cost_bytes=item['cost_bytes'],
                    fit_intervals_overlap=branches==2 and max(f['fit_started_ns'] for f in fits)<min(f['fit_finished_ns'] for f in fits),
                    new_fit_processes=len(running),new_concurrent_pair=branches==2 and len(running)==2 and max(f['fit_started_ns'] for f in fits)<min(f['fit_finished_ns'] for f in fits),
                    worker_pids=[f['pid'] for f in fits],initial_checkpoint=initial_checkpoint))
                checkpoint=accepted;initial_checkpoint=item['artifact'];gc.collect();memory_guard()
                event('unified_history_checkpoint',case=cid,method=method,block=block,quality=steps[-1]['quality'],bytes=item['cost_bytes'],paid_new_trees=paid)
            paths.append(dict(method=method,kind=kind,branches=branches,weight=weight,steps=steps,history_states=states))
        case=dict(case=cid,task=task,record=record,models=models,paths=paths,sizes=data['sizes'],warmup=dict(artifact=str(warm_path),sha256=digest(warm_path),fit=warm_fit))
        case['audit']=audit_case(case,data);moment_audits=[];initial_checks=0
        for path in paths:
            def frames():
                for ref in path['history_states']:
                    assert digest(ref['path'])==ref['sha256']
                    with np.load(ref['path'],allow_pickle=False) as f:
                        yield f['scores'].copy(),{k:f[k].copy() for k in ('mean','gram','mass','basis')}
            moment_audits.append(replay_moments(frames(),data['train_y'],task,path['kind'],path['branches'],cfg['lambda_l2']))
        for m in models:
            initial_model=load_protected(m['initial_checkpoint'])
            for fit in m['fits']:
                assert digest(fit['artifact'])==fit['sha256']
                with np.load(fit['initial_path'],allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['train'],initial_model.raw(data['train_x']).astype(np.float32))
                    np.testing.assert_array_equal(f['validation'],initial_model.raw(data['dev_x']))
                initial_checks+=1
        case['audit'].update(moment_audits=moment_audits,actual_checkpoint_initializations=initial_checks,
            overlapping_new_branch_epochs=sum(s['new_concurrent_pair'] for p in paths for s in p['steps']))
        file=directory/'case.json';write(file,case);entries.append(dict(case=cid,path=str(file),sha256=digest(file),audit=case['audit']))
        write(out/'progress.json',dict(cases=entries,paid_new_trees=paid));event('unified_history_case_complete',case=cid,paid_new_trees=paid,audit=case['audit'])
    result=dict(status='unified_history_complete',cases=entries,config=spec,environment=environment,paid_new_trees=paid,
        ledger=dict(path=str(out/'training_ledger.json'),sha256=digest(out/'training_ledger.json')),test_evaluated=False,
        elapsed_seconds=time.perf_counter()-started,cpu_peak_rss_bytes=memory_guard(),limitations=[
            'fixed half-rank and 2percent residual isolate history/branch effect; not full rank optimization',
            'history of moments; not CP/Tucker tensor decomposition','development reused adaptively',
            'two processes measure real overlap, not a guarantee of GPU speedup'])
    write(out/'report.json',result);event('unified_history_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'),cases=len(entries),paid_new_trees=paid)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if sys.argv[1]=='--fit':fit_child(sys.argv[2])
        else:run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
