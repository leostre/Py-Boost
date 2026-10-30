"""A4 frozen methods on new official holdouts; this phase never scores test."""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import json,sys,gc
import numpy as np
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import as_protected,append_protected,load_protected,block_cost
from abl_training_increment import fit_training_increment
from direct_leaf_worker import fit_direct
from direct_training import basis_for,grouped_gate
from unified_training import evaluate,serialize,quality_row
from unified_training_core import canonical,rank_grid
from unified_frontier_core import make_folds
from unified_recalibration import audit_case
from unified_serving_core import quality_key
from abl_budget_revision_core import decode_model,blank
from adaptive_controller_core import Action,candidates,diagnose,local_actions
from experiments.parallel_leaf_core import prefix
from experiments.sparse_leaf_core import contrast


def budget_before_fit(outputs,task):
    """Preregistered analytic depth3/200 storage coordinate; no trained teacher."""
    d=outputs-int(task=='multiclass')
    return int(.7*(200*(7*16+8*4*d+16)+32*outputs+4096))


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);started=perf_counter();spec=cfg['adaptive_fresh'];environment=check_native_environment(cfg)
    a3=locked(spec['a3']['path'],spec['a3']['sha256']);assert a3['status']=='direct_training_complete'
    manifest=locked(spec['manifest']['path'],spec['manifest']['sha256']);checks=grouped_gate()
    write(out/'method_freeze.json',dict(methods=spec['direct_modes'],rank_fraction=.5,residual_fraction=.02,controller='A2 risk_residual <=6local actions',
        budget_formula='0.7*(200*(7*16+8*4*(C-isMC)+16)+32*C+4096)',a3=spec['a3'],manifest=spec['manifest'],
        selection='D3fold; task NLL or macroF1; no test-dependent choices',test_evaluated=False))
    paid=0;ledger=[];entries=[]
    def fit(record,data,initial,depth,trees,path,direct=None,basis=None,importance=None):
        nonlocal paid
        path.mkdir(parents=True,exist_ok=True);c=record['case']['outputs'];task=record['case']['task'];prior=np.zeros(c,np.float32)
        if initial is None and task=='multilabel':
            p=np.clip(data['train_y'].mean(0),1e-7,1-1e-7);prior=np.log(p/(1-p)).astype(np.float32)
        train=np.broadcast_to(prior,(len(data['train_y']),c)).copy() if initial is None else initial.raw(data['train_x']).astype(np.float32)
        dev=np.broadcast_to(prior,(len(data['dev_y']),c)).copy() if initial is None else initial.raw(data['dev_x'])
        initial_path=path/'initial.npz';np.savez(initial_path,train=train,validation=dev)
        data_path=path/'fit_data.npz';np.savez(data_path,train_x=data['train_x'],train_y=data['train_y'],validation_x=data['dev_x'],validation_y=data['dev_y'])
        if paid+trees>spec['maximum_new_trees']:raise RuntimeError('A4 registered tree limit exhausted')
        paid+=trees;attempt=dict(case=record['case']['dataset'],seed=record['case']['seed'],trees=trees,depth=depth,mode=direct or 'native',status='reserved');ledger.append(attempt)
        write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger))
        request=dict(data_path=str(data_path),initial_path=str(initial_path),config={**cfg,'max_depth':depth},task=task,
            seed=record['case']['seed']+101000+(0 if initial is None else initial.n_trees),trees=trees,lr=cfg['lr'],subsample=1.,colsample=1.,
            minimum_leaf=10,output=str(path/'increment.npz'),minimum_free_mib=8192,own_pool_limit_bytes=16*1024**3)
        if direct:
            request.update(basis=basis.tolist(),residual_fraction=.02 if direct=='escape_risk' else 0.,full_coordinates=False,class_importance=importance.tolist())
            result=fit_direct(request)
        else:result=fit_training_increment(request)
        inc=native_load(result['artifact'])
        if initial is None:inc=replace(inc,base=replace(inc.base,bias=prior))
        result['initial_path']=str(initial_path);attempt.update(status='complete',fit=result)
        write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger));return inc,result
    for cc in manifest['cases']:
        cid=f"{cc['dataset']}_{cc['seed']}";record=dict(case=cc,dataset=manifest['datasets'][cc['dataset']]);data=data_before_test(record)
        directory=out/cid;directory.mkdir(exist_ok=True);task=cc['task'];c=cc['outputs'];budget=budget_before_fit(c,task)
        folds=make_folds(data['dev_y'],task,cc['seed']+20261028);models=[];traces=[]
        def store(model,path,uid,family,**meta):
            path.mkdir(parents=True,exist_ok=True);model=canonical(model);raw=model.raw(data['dev_x']);ev=evaluate(raw,data['dev_y'],task,folds,cfg,path)
            m=serialize(model,raw,ev,path,uid,dict(family=family,**meta));models.append(m);return m
        fulls={};warm=None
        for depth in (3,5):
            inc,fit_record=fit(record,data,None,depth,200,directory/f'fit-d{depth}');full=canonical(as_protected(inc));fulls[depth]=full
            view,_=decode_model(full);dense=blank(replace(view.base(),bias=view.bias.copy()))
            for count in (10,20,40,80,120,200):
                model=canonical(as_protected(prefix(dense,count)));m=store(model,directory/f'ordinary-d{depth}-t{count}',f'ordinary/d{depth}/t{count}','ordinary',depth=depth,fit=fit_record)
                if depth==3 and count==40:warm=model;warm_ref=m
        # Risk-aware posthoc map; actual bytes, no quality or memory gate.
        dense=fulls[3].blocks[0];dense=replace(dense,base=replace(dense.base,bias=fulls[3].bias.copy()))
        features=dict(train=data['train_x'],development=data['dev_x']);labels={'train':data['train_y']};ranks=rank_grid(c,task)
        for candidate in candidates(None,dense,features,labels,[Action(r,8,s) for r in ranks for s in (0.,.02,.05)],'risk_residual'):
            store(candidate['model'],directory/'posthoc'/candidate['uid'],'posthoc/'+candidate['uid'],'posthoc',rank=candidate['rank'],residual_fraction=candidate['residual_fraction'])
        checkpoint=warm;rank=min(ranks,key=lambda r:abs(r-c/2));cap=8;residual=.02;steps=[]
        for block in range(4):
            step=directory/'controller'/f'b{block}';inc,fit_record=fit(record,data,checkpoint,3,40,step/'fit')
            probes=list(candidates(checkpoint,inc,features,labels,[Action(rank,8,0.,'keep'),Action(rank,4,0.,'keep')],'risk_residual'))
            teacher=append_protected(checkpoint,inc);raw=teacher.raw(data['train_x']);diagnostic=diagnose(contrast(raw-probes[0]['raw']['train'],task),contrast(raw-probes[1]['raw']['train'],task))
            previous=checkpoint.blocks[0].base.basis if len(checkpoint.blocks)==1 and checkpoint.blocks[0].base.representation=='low_rank' else None
            pool=[p for p in candidates(checkpoint,inc,features,labels,local_actions(rank,cap,residual,ranks,diagnostic),'risk_residual',previous) if p['cost']['cost_bytes']<=budget]
            if not pool:pool=[p for p in candidates(checkpoint,inc,features,labels,[Action(1,4,0.)],'risk_residual') if p['cost']['cost_bytes']<=budget]
            if not pool:steps.append(dict(block=block,status='no_affordable_action'));break
            pairs=[]
            for p in pool[:6]:
                m=store(p['model'],step/p['uid'],f'controller/{block}/'+p['uid'],'controller',block=block,rank=p['rank'],residual_fraction=p['residual_fraction'],fit=fit_record)
                pairs.append((p,m))
            chosen,m=min(pairs,key=lambda pair:quality_key(pair[1],task));checkpoint=chosen['model'];rank=chosen['rank'];cap=chosen['leaf_cap'];residual=chosen['residual_fraction']
            steps.append(dict(block=block,uid=m['uid'],cost_bytes=m['cost_bytes'],diagnostic=diagnostic,quality=quality_row(m['evaluations'],task)['quality']))
        traces.append(dict(method='controller',steps=steps))
        for direct_mode in spec['direct_modes']:
            checkpoint=warm;rank=max(1,int(.5*(c-int(task=='multiclass'))));b,imp,meta=basis_for(warm,data,task,rank)
            for block in range(4 if direct_mode=='refresh_risk' else 1):
                if direct_mode=='refresh_risk':b,imp,meta=basis_for(checkpoint,data,task,rank)
                step=directory/direct_mode/f'b{block}';inc,fit_record=fit(record,data,checkpoint,3,40 if direct_mode=='refresh_risk' else 160,step/'fit',direct_mode,b,imp)
                checkpoint=canonical(append_protected(checkpoint,inc));store(checkpoint,step,direct_mode+'/'+str(block),direct_mode,block=block,rank=rank,fit=fit_record)
        case=dict(case=cid,record=record,task=task,models=models,traces=traces,budget=budget,warmup=warm_ref,reference_bytes=block_cost(fulls[3])['cost_bytes'])
        case['audit']=audit_case(case,data);file=directory/'case.json';write(file,case);entries.append(dict(case=cid,path=str(file),sha256=digest(file),audit=case['audit']))
        write(out/'progress.json',dict(cases=entries,paid_new_trees=paid));event('adaptive_fresh_case',case=cid,paid_new_trees=paid,models=len(models),audit=case['audit'])
        del fulls,checkpoint,models;gc.collect();memory_guard()
    result=dict(status='adaptive_fresh_training_complete',cases=entries,paid_new_trees=paid,environment=environment,elapsed_seconds=perf_counter()-started,
        checks=checks,test_evaluated=False,method_freeze=dict(path=str(out/'method_freeze.json'),sha256=digest(out/'method_freeze.json')),
        ledger=dict(path=str(out/'training_ledger.json'),sha256=digest(out/'training_ledger.json')))
    write(out/'report.json',result);event('adaptive_fresh_training_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
