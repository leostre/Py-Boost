"""A2 closed-loop finite controller; full tree work explicitly ledgered."""
from pathlib import Path
from itertools import product
from time import perf_counter
import json,sys,gc,hashlib
import numpy as np
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,append_protected,as_protected
from abl_training_increment import fit_training_increment
from unified_training import evaluate,serialize,quality_row
from unified_training_core import canonical,rank_grid,shortlist
from unified_frontier_core import make_folds
from unified_recalibration import audit_case
from experiments.sparse_leaf_core import contrast
from adaptive_controller_core import Action,promotion,candidates,diagnose,local_actions


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);started=perf_counter();environment=check_native_environment(cfg);spec=cfg['adaptive_training']
    a1=locked(spec['a1']['path'],spec['a1']['sha256']);assert a1['status']=='adaptive_saved_complete'
    promoted=promotion([locked(r['path'],r['sha256']) for r in a1['cases']]);policy=promoted['chosen']
    write(out/'basis_policy_sealed.json',promoted);event('adaptive_basis_promoted',**promoted)
    source=locked(spec['source']['path'],spec['source']['sha256']);entries=[];ledger=[];paid=0;cache={}
    def fit(record,data,initial,seed,directory):
        nonlocal paid
        tr=initial.raw(data['train_x']).astype(np.float32);dv=initial.raw(data['dev_x'])
        key=hashlib.sha256(tr.tobytes()+dv.tobytes()+str((record['case']['dataset'],record['case']['seed'],seed)).encode()).hexdigest()
        if key in cache:
            result=cache[key];ledger.append(dict(case=record['case']['dataset'],trees=40,reused=True,fit=result));return native_load(result['artifact']),result
        if paid+40>spec['maximum_new_trees']:raise RuntimeError('A2 registered tree-work exhausted')
        directory.mkdir(parents=True,exist_ok=True);initial_path=directory/'initial.npz';np.savez(initial_path,train=tr,validation=dv)
        data_path=directory.parents[2]/'fit_data.npz'
        if not data_path.exists():np.savez(data_path,train_x=data['train_x'],train_y=data['train_y'],validation_x=data['dev_x'],validation_y=data['dev_y'])
        paid+=40;ledger.append(dict(case=record['case']['dataset'],trees=40,reused=False,status='reserved'))
        write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger))
        result=fit_training_increment(dict(data_path=str(data_path),initial_path=str(initial_path),config={**cfg,'max_depth':3},
            task=record['case']['task'],seed=seed,trees=40,lr=cfg['lr'],subsample=1.,colsample=1.,minimum_leaf=10,
            output=str(directory/'increment.npz'),minimum_free_mib=8192,own_pool_limit_bytes=16*1024**3))
        result['initial_path']=str(initial_path);cache[key]=result;ledger[-1].update(status='complete',fit=result)
        write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger));return native_load(result['artifact']),result
    for cid in spec['cases']:
        ref=next(r for r in source['cases'] if r['case']==cid);old=locked(ref['path'],ref['sha256']);record=old['record'];task=old['task']
        data=data_before_test(record);features=dict(train=data['train_x'],development=data['dev_x']);labels={'train':data['train_y']}
        folds=make_folds(data['dev_y'],task,record['case']['seed']+20261028);directory=out/cid;directory.mkdir(exist_ok=True)
        assert digest(old['warmup']['artifact'])==old['warmup']['sha256'];warm=load_protected(old['warmup']['artifact'])
        budget=int(spec['memory_ratio']*old['reference_bytes']);models=[dict(m,uid='reference/'+m['uid']) for m in old['models'] if m['family']=='ordinary']
        grid=rank_grid(len(warm.bias),task);default=min(grid,key=lambda r:abs(r-len(warm.bias)/2));traces=[]
        for method in spec['methods']:
            checkpoint=warm;previous_ref=old['warmup'];rank=default;cap=8;fraction=.02;steps=[]
            for block in range(4):
                step=directory/method/f'b{block}';step.mkdir(parents=True,exist_ok=True);t=perf_counter()
                inc,fit_record=fit(record,data,checkpoint,record['case']['seed']+81040+40*block,step/'fit')
                teacher=canonical(append_protected(checkpoint,inc));teacher_dir=step/'full_proposal';teacher_dir.mkdir(exist_ok=True)
                teacher_raw=teacher.raw(data['dev_x']);teacher_ev=evaluate(teacher_raw,data['dev_y'],task,folds,cfg,teacher_dir)
                teacher_item=serialize(teacher,teacher_raw,teacher_ev,teacher_dir,f'{method}/proposal{block}',dict(family='proposal',block=block,fit=fit_record))
                models.append(teacher_item)
                previous_basis=checkpoint.blocks[0].base.basis if len(checkpoint.blocks)==1 and checkpoint.blocks[0].base.representation=='low_rank' else None
                diagnosis=None
                if method=='static':actions=[Action(default,8,.02)]
                elif method=='grid':actions=[Action(r,l,s) for r,l,s in product(grid,(4,6,8),(0.,.02,.05))]
                else:
                    probes=list(candidates(checkpoint,inc,features,labels,[Action(rank,8,0.,'keep'),Action(rank,4,0.,'keep')],policy,previous_basis))
                    tr=teacher.raw(data['train_x']);diagnosis=diagnose(contrast(tr-probes[0]['raw']['train'],task),contrast(tr-probes[1]['raw']['train'],task))
                    actions=local_actions(rank,cap,fraction,grid,diagnosis);del probes
                pool=[c for c in candidates(checkpoint,inc,features,labels,actions,policy,previous_basis) if c['cost']['cost_bytes']<=budget]
                if not pool and method=='controller':
                    pool=[c for c in candidates(checkpoint,inc,features,labels,[Action(1,4,0.)],policy,previous_basis) if c['cost']['cost_bytes']<=budget]
                    diagnosis['emergency_minimum_representation']=True
                if not pool:
                    steps.append(dict(block=block,status='no_affordable_registered_action',fit=fit_record));break
                chosen=shortlist(pool,spec['maximum_development_candidates']) if method=='grid' else pool[:spec['maximum_development_candidates']]
                evaluated=[]
                for c in chosen:
                    p=step/c['uid'];p.mkdir(exist_ok=True);ev=evaluate(c['raw']['development'],data['dev_y'],task,folds,cfg,p)
                    m=serialize(c['model'],c['raw']['development'],ev,p,f'{method}/{block}/'+c['uid'],dict(family=method,block=block,rank=c['rank'],
                        leaf_cap=c['leaf_cap'],residual_fraction=c['residual_fraction'],basis_kind=c['basis_kind'],fit=fit_record,initial_checkpoint=previous_ref['artifact']))
                    models.append(m);evaluated.append((c,m))
                def key(pair):
                    q=quality_row(pair[1]['evaluations'],task)['quality'];return (q['loss'],pair[1]['cost_bytes']) if task=='multiclass' else (-q['f1_macro'],q['loss'],pair[1]['cost_bytes'])
                c,m=min(evaluated,key=key);checkpoint=c['model'];previous_ref=m;rank=c['rank'];cap=c['leaf_cap'];fraction=c['residual_fraction']
                q=quality_row(m['evaluations'],task)['quality'];teacher_q=quality_row(teacher_ev,task)['quality'];loss=1-q['f1_macro']/max(teacher_q['f1_macro'],1e-30)
                state=dict(block=block,status='accepted',chosen=m['uid'],diagnosis=diagnosis,generated=len(pool),development_queries=len(evaluated),
                    elapsed_seconds=perf_counter()-t,cost_bytes=m['cost_bytes'],quality=q,proposal=teacher_item,macro_loss_vs_proposal=loss)
                steps.append(state);event('adaptive_training_step',case=cid,method=method,**{k:v for k,v in state.items() if k!='proposal'})
                if loss>.5:state['status']='over50percent_macro_loss_stop';break
                del pool,chosen,evaluated;gc.collect();memory_guard()
            traces.append(dict(method=method,steps=steps))
        case=dict(case=cid,task=task,record=record,models=models,traces=traces,reference=old['reference'],reference_bytes=old['reference_bytes'],basis_policy=policy,budget=budget)
        case['audit']=audit_case(case,data);checks=0
        for m in models:
            if m['family'] not in spec['methods']:continue
            before=load_protected(m['initial_checkpoint'])
            with np.load(m['fit']['initial_path'],allow_pickle=False) as f:
                np.testing.assert_array_equal(f['train'],before.raw(data['train_x']).astype(np.float32));np.testing.assert_array_equal(f['validation'],before.raw(data['dev_x']))
            checks+=1
        case['audit']['actual_compact_initialization_checks']=checks
        file=directory/'case.json';write(file,case);entries.append(dict(case=cid,path=str(file),sha256=digest(file),audit=case['audit']))
        write(out/'progress.json',dict(cases=entries,paid_new_trees=paid));event('adaptive_training_case_complete',case=cid,audit=case['audit'],paid_new_trees=paid)
    result=dict(status='adaptive_training_complete',cases=entries,promotion=promoted,paid_new_trees=paid,environment=environment,
        ledger=dict(path=str(out/'training_ledger.json'),sha256=digest(out/'training_ledger.json')),test_evaluated=False,
        elapsed_seconds=perf_counter()-started,limitations=['D adaptively reused','structure actions prune or preserve new splits; no unbounded deeper-tree search',
        'full proposals still paid; ordinary reference reused without retraining','six D queries maximum per controller/grid step; no claim of equal total compression work'])
    write(out/'report.json',result);event('adaptive_training_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
