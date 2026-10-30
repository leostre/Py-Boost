"""A3 canonical cohort: direct compact versus native and coupled full controls."""
from pathlib import Path
from time import perf_counter
import json,sys,gc
import numpy as np
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,append_protected
from abl_training_increment import fit_training_increment
from direct_leaf_worker import fit_direct
from unified_training import evaluate,serialize,quality_row
from unified_training_core import canonical
from unified_frontier_core import make_folds
from unified_recalibration import audit_case
from adaptive_basis_core import risk_basis,risk_weights


def grouped_gate():
    import cupy as cp
    from direct_leaf_gpu import leaf_statistics_gpu,leaf_statistics_loop
    from direct_leaf_core import leaf_statistics_reference
    rng=np.random.default_rng(2931);checks=[]
    for task in ('multiclass','multilabel'):
        raw=rng.normal(size=(21,5));y=rng.integers(0,5,21) if task=='multiclass' else rng.integers(0,2,(21,5));ids=np.arange(21)%4
        for identity in (False,True):
            b=np.eye(5) if identity else np.linalg.qr(rng.normal(size=(5,3)))[0]
            args=[cp.asarray(a) for a in (raw,y,ids,b)]+[task,8,identity]
            actual=leaf_statistics_gpu(*args);loop=leaf_statistics_loop(*args);ref=leaf_statistics_reference(raw,y,ids,b,task)
            errors={k:float(np.max(np.abs(v.get()-ref[k]))) for k,v in actual.items() if v is not None}
            for k,v in actual.items():
                if v is not None:np.testing.assert_allclose(v.get(),loop[k].get(),atol=1e-10,rtol=1e-10)
            assert max(errors.values())<1e-10
            checks.append(dict(task=task,identity=identity,errors=errors))
    cp.get_default_memory_pool().free_all_blocks();return checks


def basis_for(model,data,task,rank):
    raw=model.raw(data['train_x']);rows,columns=risk_weights(data['train_y'],raw,task)
    basis,meta=risk_basis(raw-model.bias,task,rows,columns)
    return basis[:,:rank],columns,meta


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);started=perf_counter();environment=check_native_environment(cfg);spec=cfg['direct_training']
    prior=locked(spec['preflight']['path'],spec['preflight']['sha256']);assert prior['status']=='direct_preflight_complete'
    checks=grouped_gate();write(out/'grouped_gate.json',checks)
    source=locked(spec['source']['path'],spec['source']['sha256']);entries=[];ledger=[];paid=spec['prior_paid_new_trees']
    for cid,rank,escape in spec['cases']:
        oldref=next(r for r in source['cases'] if r['case']==cid);old=locked(oldref['path'],oldref['sha256']);record=old['record'];task=old['task']
        data=data_before_test(record);folds=make_folds(data['dev_y'],task,record['case']['seed']+20261028)
        directory=out/cid;directory.mkdir(exist_ok=True);data_path=directory/'fit_data.npz'
        np.savez(data_path,train_x=data['train_x'],train_y=data['train_y'],validation_x=data['dev_x'],validation_y=data['dev_y'])
        assert digest(old['warmup']['artifact'])==old['warmup']['sha256'];warm=load_protected(old['warmup']['artifact'])
        warmbasis,importance,bmeta=basis_for(warm,data,task,rank);models=[];traces=[]
        for mode in spec['modes']:
            checkpoint=warm;mode_dir=directory/mode;mode_dir.mkdir(exist_ok=True);steps=[];mode_started=perf_counter()
            for block in range(4 if mode=='refresh_risk' else 1):
                step=mode_dir/f'b{block}';step.mkdir(exist_ok=True)
                initial_path=step/'initial.npz';np.savez(initial_path,train=checkpoint.raw(data['train_x']).astype(np.float32),validation=checkpoint.raw(data['dev_x']))
                trees=40 if mode=='refresh_risk' else 160
                if paid+trees>spec['maximum_new_trees']:raise RuntimeError('A3 work budget exhausted')
                paid+=trees;attempt=dict(case=cid,mode=mode,block=block,trees=trees,status='reserved');ledger.append(attempt)
                write(out/'training_ledger.json',dict(paid_new_trees=paid,prior_paid_new_trees=spec['prior_paid_new_trees'],attempts=ledger))
                request=dict(data_path=str(data_path),initial_path=str(initial_path),config={**cfg,'max_depth':3},task=task,
                    seed=record['case']['seed']+91040+40*block,trees=trees,lr=cfg['lr'],subsample=1.,colsample=1.,minimum_leaf=10,
                    output=str(step/'increment.npz'),minimum_free_mib=8192,own_pool_limit_bytes=16*1024**3)
                if mode=='native_full':fit=fit_training_increment(request);inc=native_load(fit['artifact'])
                else:
                    t=perf_counter();b,imp,meta=(basis_for(checkpoint,data,task,rank) if mode=='refresh_risk' else (warmbasis,importance,bmeta))
                    if mode=='direct_full':b=np.eye(len(warm.bias),dtype=np.float32)
                    basis_seconds=perf_counter()-t
                    request.update(basis=b.tolist(),residual_fraction=escape if mode=='escape_risk' else 0.,
                        full_coordinates=mode=='direct_full',class_importance=imp.tolist(),chunk_rows=1024)
                    fit=fit_direct(request);fit.update(basis_seconds=basis_seconds,basis_metadata=meta)
                    inc=native_load(fit['artifact'])
                fit['initial_path']=str(initial_path);attempt.update(status='complete',fit=fit)
                write(out/'training_ledger.json',dict(paid_new_trees=paid,prior_paid_new_trees=spec['prior_paid_new_trees'],attempts=ledger))
                checkpoint=canonical(append_protected(checkpoint,inc));raw=checkpoint.raw(data['dev_x']);ev=evaluate(raw,data['dev_y'],task,folds,cfg,step)
                item=serialize(checkpoint,raw,ev,step,f'{mode}/{block}',dict(family=mode,block=block,fit=fit))
                models.append(item);steps.append(dict(model=item['uid'],cost_bytes=item['cost_bytes'],quality=quality_row(ev,task)['quality']))
                event('direct_training_step',case=cid,mode=mode,block=block,trees=trees,seconds=fit['seconds'],quality=steps[-1]['quality'])
                gc.collect();memory_guard()
            traces.append(dict(mode=mode,steps=steps,total_wall_seconds=perf_counter()-mode_started))
        case=dict(case=cid,task=task,record=record,models=models,traces=traces,rank=rank,escape=escape,warmup=old['warmup'])
        case['audit']=audit_case(case,data);path=directory/'case.json';write(path,case)
        entries.append(dict(case=cid,path=str(path),sha256=digest(path),audit=case['audit']))
        write(out/'progress.json',dict(cases=entries,paid_new_trees=paid))
    result=dict(status='direct_training_complete',cases=entries,checks=checks,paid_new_trees=paid,environment=environment,
        elapsed_seconds=perf_counter()-started,ledger=dict(path=str(out/'training_ledger.json'),sha256=digest(out/'training_ledger.json')),
        test_evaluated=False,limitations=['adaptive D reused','split scoring uses diagonal projected curvature; leaf solve uses full projected curvature',
        'refresh mode pays4 quantizations; other modes one','full raw C-output state retained; no optimizer-state savings claim'])
    write(out/'report.json',result);event('direct_training_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
