"""U2: real compact-feedback training, memory grid, no old quality gate."""
from pathlib import Path
from dataclasses import replace
import gc,hashlib,json,sys
from time import perf_counter
import numpy as np
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import as_protected,append_protected,load_protected,block_cost
from abl_training_increment import fit_training_increment
from unified_training_core import candidate_grid,shortlist,canonical
from unified_recalibration import store_replay,audit_case
from unified_frontier_core import make_folds,replay_calibration,choose_policy,variants,frontier
from abl_budget_revision_core import decode_model,blank
from unified_data_core import expanded_record


def evaluate(raw,y,task,folds,cfg,out):
    replay=replay_calibration(raw,y,task,folds,cfg['decision_grid'],cfg['decision_shrinkage'])
    saved=store_replay(replay,out/'evaluation.npz');fitted=variants(raw,y,raw[:1],y[:1],task,cfg['decision_grid'],cfg['decision_shrinkage'])
    return [dict(protocol='crossfit',repeat=0,objective=obj,policy=(name:=choose_policy(replay['scores'],obj)),
        quality=replay['scores'][name],control=replay['scores'],refit_control=fitted[name],evaluation=saved,
        evaluation_scope='adaptive use of D; external evaluation not performed here') for obj in ('f1_macro','loss')]


def quality_row(evaluations,task):
    objective='loss' if task=='multiclass' else 'f1_macro'
    return next(r for r in evaluations if r['objective']==objective)


def serialize(model,raw,evaluations,path,uid,metadata):
    model_path=path/'model.npz';model.save(model_path);cache=path/'raw.npz';np.savez_compressed(cache,raw=raw)
    result=dict(uid=uid,artifact=str(model_path),sha256=digest(model_path),raw=dict(path=str(cache),sha256=digest(cache)),
        evaluations=evaluations,**block_cost(model),**metadata)
    write(path/'model.json',result);return result


def run(cfg,out):
    started=perf_counter();environment=check_native_environment(cfg);spec=cfg['unified_training']
    prerequisite=locked(spec['u1']['path'],spec['u1']['sha256'])
    assert prerequisite['status']=='unified_recalibration_complete' and all(c['audit']['status']=='passed' for c in prerequisite['cases'])
    source_ref=spec.get('source_manifest',dict(path=cfg['absolute_roles_manifest'],sha256=cfg['absolute_roles_manifest_sha256']))
    source=locked(source_ref['path'],source_ref['sha256'])
    old=locked(cfg['abl_protected_development_completed']['path'],cfg['abl_protected_development_completed']['sha256'])
    old_cases={r['case_id']:locked(r['path'],r['sha256']) for r in old['cases']}
    cache={};ledger=[];entries=[];paid=0
    if spec.get('prior_completed'):
        prior=locked(spec['prior_completed']['path'],spec['prior_completed']['sha256'])
        assert prior['status']=='unified_training_complete'
        entries=list(prior['cases']);paid=prior['paid_new_trees']
        ledger=list(locked(prior['ledger']['path'],prior['ledger']['sha256'])['attempts'])
    out.mkdir(parents=True,exist_ok=True)
    write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger))
    def fit_increment(record,data,initial,seed,trees,directory):
        nonlocal paid
        directory.mkdir(parents=True,exist_ok=True);c=record['case']['outputs'];task=record['case']['task']
        prior=np.zeros(c,np.float32)
        if initial is None and task=='multilabel':
            p=np.clip(data['train_y'].mean(0),1e-7,1-1e-7);prior=np.log(p/(1-p)).astype(np.float32)
        tr=np.broadcast_to(prior,(len(data['train_y']),c)).copy() if initial is None else initial.raw(data['train_x']).astype(np.float32)
        dv=np.broadcast_to(prior,(len(data['dev_y']),c)).copy() if initial is None else initial.raw(data['dev_x'])
        key=hashlib.sha256(tr.tobytes()+dv.tobytes()+str((record['case']['dataset'],record['case']['seed'],seed,trees)).encode()).hexdigest()
        if key in cache:
            result=cache[key];ledger.append(dict(case=record['case']['dataset'],seed=seed,trees=trees,reused=True,fit=result))
            increment=native_load(result['artifact'])
            if result.get('added_initial_bias') is not None:
                increment=replace(increment,base=replace(increment.base,bias=np.asarray(result['added_initial_bias'],np.float32)))
            return increment,result
        initial_path=directory/'initial.npz';np.savez(initial_path,train=tr,validation=dv)
        data_path=directory.parent/'fit_data.npz'
        if not data_path.exists():np.savez(data_path,train_x=data['train_x'],train_y=data['train_y'],validation_x=data['dev_x'],validation_y=data['dev_y'])
        if paid+trees>spec['maximum_new_trees']:raise RuntimeError('registered U2 tree-work limit exhausted')
        paid+=trees;ledger.append(dict(case=record['case']['dataset'],seed=seed,trees=trees,reused=False,status='reserved'))
        write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger))
        result=fit_training_increment(dict(data_path=str(data_path),initial_path=str(initial_path),
            config={**cfg,'max_depth':3},task=task,seed=seed,trees=trees,lr=cfg['lr'],subsample=1.,colsample=1.,
            minimum_leaf=10,output=str(directory/'increment.npz'),minimum_free_mib=8192,own_pool_limit_bytes=16*1024**3))
        result.update(initial_path=str(initial_path),initial_function_sha256=hashlib.sha256(tr.tobytes()+dv.tobytes()).hexdigest())
        increment=native_load(result['artifact'])
        if initial is None:
            increment=replace(increment,base=replace(increment.base,bias=prior));result['added_initial_bias']=prior.tolist()
        cache[key]=result;ledger[-1].update(status='complete',fit=result)
        write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger));return increment,result

    for name,seed in spec['cases']:
        cid=f'{name}_{seed}';directory=out/cid;directory.mkdir(parents=True,exist_ok=True)
        if any(e['case']==cid for e in entries):continue
        record=dict(case=next(c for c in source['cases'] if c['dataset']==name and c['seed']==seed),dataset=source['datasets'][name])
        if cid in spec.get('already_pooled_cases',[]):
            assert record['case']['protocol_version']==4 and record['case']['train_meta']['protocol']=='grouped80-20-nontest'
        elif spec.get('reclaim_training_fraction'):
            record=expanded_record(record,directory,spec['reclaim_training_fraction'])
        data=data_before_test(record);task=record['case']['task'];folds=make_folds(data['dev_y'],task,seed+20261028)
        features={'train':data['train_x'],'development':data['dev_x']};labels={'train':data['train_y']};models=[];traces=[]
        if cid in old_cases and not spec.get('fresh_warmup',False):
            warm_ref=old_cases[cid]['metas']['full/b00'];assert digest(warm_ref['artifact'])==warm_ref['sha256'];warm=load_protected(warm_ref['artifact'])
        else:
            inc,_=fit_increment(record,data,None,seed+81000,40,directory/'warmup');warm=canonical(as_protected(inc))
        warm_path=directory/'warmup.npz';warm.save(warm_path)
        ordinary=warm;ordinary_snapshots=[]
        for block in range(4):
            step=directory/'ordinary'/f'b{block}';step.mkdir(parents=True,exist_ok=True)
            inc,fit=fit_increment(record,data,ordinary,seed+81040+40*block,40,step)
            ordinary=canonical(append_protected(ordinary,inc));raw=ordinary.raw(data['dev_x']);ev=evaluate(raw,data['dev_y'],task,folds,cfg,step)
            initial_checkpoint=ordinary_snapshots[-1]['artifact'] if ordinary_snapshots else str(warm_path)
            item=serialize(ordinary,raw,ev,step,f'ordinary/{block}',dict(family='ordinary',block=block,fit=fit,initial_checkpoint=initial_checkpoint))
            models.append(item);ordinary_snapshots.append(item);event('unified_ordinary',case=cid,block=block,trees=ordinary.n_trees)
        reference_bytes=block_cost(ordinary)['cost_bytes'];reference=quality_row(ordinary_snapshots[-1]['evaluations'],task)
        view,_=decode_model(ordinary);dense=blank(replace(view.base(),bias=view.bias.copy()))
        # Full posthoc map, no memory or 5% quality filtering.
        posthoc_dir=directory/'posthoc';posthoc_dir.mkdir(exist_ok=True)
        for candidate in candidate_grid(None,dense,features,labels,None):
            p=posthoc_dir/candidate['uid'];p.mkdir(exist_ok=True)
            ev=evaluate(candidate['raw']['development'],data['dev_y'],task,folds,cfg,p)
            item=serialize(candidate['model'],candidate['raw']['development'],ev,p,'posthoc/'+candidate['uid'],
                dict(family='posthoc',rank=candidate['rank'],leaf_cap=candidate['leaf_cap'],residual_fraction=candidate['residual_fraction']))
            models.append(item)
        for ratio in spec['memory_ratios']:
            budget=int(np.floor(ratio*reference_bytes));checkpoint=warm;path_trace=[];initial_checkpoint=str(warm_path)
            for block in range(4):
                step=directory/f'online{ratio:g}'/f'b{block}';step.mkdir(parents=True,exist_ok=True)
                inc,fit=fit_increment(record,data,checkpoint,seed+81040+40*block,40,step)
                teacher=canonical(append_protected(checkpoint,inc));teacher_dir=step/'uncompressed_proposal';teacher_dir.mkdir(exist_ok=True)
                teacher_evaluations=evaluate(teacher.raw(data['dev_x']),data['dev_y'],task,folds,cfg,teacher_dir)
                teacher_quality=quality_row(teacher_evaluations,task)['quality']
                candidates=list(candidate_grid(checkpoint,inc,features,labels,budget));chosen=shortlist(candidates,spec['maximum_development_candidates'])
                if not chosen:
                    path_trace.append(dict(block=block,status='no_representation_in_this_grid_budget',fit=fit,candidates=len(candidates)));break
                evaluated=[]
                for candidate in chosen:
                    p=step/candidate['uid'];p.mkdir(exist_ok=True)
                    ev=evaluate(candidate['raw']['development'],data['dev_y'],task,folds,cfg,p)
                    item=serialize(candidate['model'],candidate['raw']['development'],ev,p,f'online{ratio:g}/{block}/'+candidate['uid'],
                        dict(family='online',budget=budget,memory_ratio_target=ratio,block=block,fit=fit,rank=candidate['rank'],
                            leaf_cap=candidate['leaf_cap'],residual_fraction=candidate['residual_fraction'],initial_checkpoint=initial_checkpoint))
                    evaluated.append((candidate,item));models.append(item)
                def key(pair):
                    q=quality_row(pair[1]['evaluations'],task)['quality']
                    return (q['loss'],-q['f1_macro'],pair[1]['cost_bytes']) if task=='multiclass' else (-q['f1_macro'],q['loss'],pair[1]['cost_bytes'])
                candidate,item=min(evaluated,key=key);checkpoint=candidate['model'];q=quality_row(item['evaluations'],task)['quality']
                compression_loss=1-q['f1_macro']/max(teacher_quality['f1_macro'],1e-30)
                path_trace.append(dict(block=block,status='accepted_without_5percent_gate',chosen=item['uid'],fit=fit,
                    generated_candidates=len(candidates),development_candidates=len(chosen),cost_bytes=item['cost_bytes'],quality=q,
                    uncompressed_proposal_evaluations=teacher_evaluations,macro_loss_vs_this_full_proposal=compression_loss))
                if compression_loss>.5:
                    path_trace[-1]['status']='compression_loss_over50percent_stop_this_path'
                    break
                initial_checkpoint=item['artifact']
                event('unified_online',case=cid,ratio=ratio,block=block,rank=item['rank'],leaf_cap=item['leaf_cap'],quality=q,bytes=item['cost_bytes'])
                del candidates,chosen,evaluated;gc.collect();memory_guard()
            traces.append(dict(memory_ratio=ratio,budget=budget,steps=path_trace))
        for item in models:
            for evaluation in item['evaluations']:
                evaluation['macro_loss_vs_full']=1-evaluation['quality']['f1_macro']/max(reference['quality']['f1_macro'],1e-30)
        case=dict(case=cid,task=task,record=record,models=models,traces=traces,reference_bytes=reference_bytes,
            reference=reference,sizes=data['sizes'],warmup=dict(artifact=str(warm_path),sha256=digest(warm_path)))
        case['audit']=audit_case(case,data)
        fit_checks=0
        for m in models:
            if m.get('initial_checkpoint') is None:continue
            before=load_protected(m['initial_checkpoint']);fit=m['fit']
            assert digest(fit['artifact'])==fit['sha256']
            with np.load(fit['initial_path'],allow_pickle=False) as saved:
                np.testing.assert_allclose(saved['train'],before.raw(data['train_x']).astype(np.float32),atol=0,rtol=0)
                np.testing.assert_allclose(saved['validation'],before.raw(data['dev_x']),atol=0,rtol=0)
            fit_checks+=1
        case['audit']['actual_checkpoint_initialization_checks']=fit_checks
        case['fronts']={obj:frontier([dict(uid=m['uid'],cost_bytes=m['cost_bytes'],quality=next(e['quality'] for e in m['evaluations'] if e['objective']==obj)) for m in models],obj) for obj in ('loss','f1_macro')}
        file=directory/'case.json';write(file,case);entries.append(dict(case=cid,path=str(file),sha256=digest(file),audit=case['audit'],models=len(models)))
        write(out/'progress.json',dict(cases=entries,paid_new_trees=paid));event('unified_training_case_complete',case=cid,models=len(models),paid_new_trees=paid,audit=case['audit'])
    result=dict(status='unified_training_complete',cases=entries,environment=environment,config=spec,paid_new_trees=paid,
        ledger=dict(path=str(out/'training_ledger.json'),sha256=digest(out/'training_ledger.json')),test_evaluated=False,
        elapsed_seconds=perf_counter()-started,cpu_peak_rss_bytes=memory_guard(),limitations=[
            'block40 compact feedback; within-block temporary full proposals remain',
            'leaf pruning uses static full-output surrogate; joint choice is bounded grid, not global optimization',
            'development reused adaptively; outside validation still needed',
            '50percent region tagged; no unseen extrapolation toward more aggressive settings'])
    write(out/'report.json',result);event('unified_training_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'),cases=len(entries),paid_new_trees=paid)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
