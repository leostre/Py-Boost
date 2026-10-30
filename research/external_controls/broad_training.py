"""F1: bounded fits and common D-only selection; never loads target[test]."""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import json,sys,gc
import numpy as np
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_training_increment import fit_training_increment
from direct_leaf_worker import fit_direct
from direct_training import grouped_gate
from abl_protected_core import as_protected,append_protected,block_cost
from abl_budget_revision_core import decode_model,blank
from adaptive_basis_core import risk_basis,risk_weights
from adaptive_controller_core import Action,candidates,diagnose,local_actions
from broad_core import ORDINARY_COUNTS,budget_reference,byte_limit,ranks_for,grouped_folds,assess,model_prefix,train_shortlist,qkey,select_cases
from experiments.parallel_leaf_core import prefix


def run(cfg,out):
    started=perf_counter();out.mkdir(parents=True,exist_ok=True);spec=cfg['broad'];environment=check_native_environment(cfg)
    manifest=locked(spec['manifest']['path'],spec['manifest']['sha256']);checks=grouped_gate()
    ledger=[];paid=int(spec.get('prior_paid_new_trees',0));entries=[]
    def fit(case,data,initial,depth,trees,path,method='full',basis=None,importance=None):
        nonlocal paid
        path.mkdir(parents=True,exist_ok=True);c=case['outputs'];prior=np.zeros(c,np.float32)
        if initial is None:
            p=np.clip(data['train_y'].mean(0),1e-7,1-1e-7);prior=np.log(p/(1-p)).astype(np.float32)
        tr=np.broadcast_to(prior,(len(data['train_y']),c)).copy() if initial is None else initial.raw(data['train_x']).astype(np.float32)
        dv=np.broadcast_to(prior,(len(data['dev_y']),c)).copy() if initial is None else initial.raw(data['dev_x'])
        initial_file=path/'initial.npz';np.savez(initial_file,train=tr,validation=dv)
        data_file=path/'data.npz';np.savez(data_file,train_x=data['train_x'],train_y=data['train_y'],validation_x=data['dev_x'],validation_y=data['dev_y'])
        for reuse in spec.get('reuse_fits',[]):
            if [case['dataset'],case['seed'],depth,trees,method] != reuse['key']:continue
            result=locked(reuse['path'],reuse['sha256'])
            old_initial=Path(result['artifact']).parent/'initial.npz';old_data=Path(result['artifact']).parent/'data.npz'
            assert digest(result['artifact'])==result['sha256'] and digest(old_initial)==result['initial_sha256'] and digest(old_data)==result['data_sha256']
            with np.load(old_initial,allow_pickle=False) as f:
                assert np.array_equal(f['train'],tr) and np.array_equal(f['validation'],dv)
            with np.load(old_data,allow_pickle=False) as f:
                for key,value in dict(train_x=data['train_x'],train_y=data['train_y'],validation_x=data['dev_x'],validation_y=data['dev_y']).items():
                    assert np.array_equal(f[key],value)
            inc=native_load(result['artifact'])
            if initial is None:inc=replace(inc,base=replace(inc.base,bias=prior))
            ledger.append(dict(case=case['dataset'],seed=case['seed'],depth=depth,trees=trees,method=method,status='verified_reuse',fit=result))
            event('broad_fit_verified_reuse',case=case['dataset'],method=method,trees=trees,paid=paid)
            return inc,result
        if paid+trees>spec['maximum_new_trees']:raise RuntimeError('registered broad tree budget exhausted')
        paid+=trees;attempt=dict(case=case['dataset'],seed=case['seed'],trees=trees,depth=depth,method=method,status='reserved');ledger.append(attempt)
        write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger,prior_paid=spec.get('prior_paid_new_trees',0)))
        request=dict(data_path=str(data_file),initial_path=str(initial_file),config={**cfg,'max_depth':depth},task='multilabel',
                     seed=case['seed']+101000+(0 if initial is None else initial.n_trees),trees=trees,lr=cfg['lr'],
                     subsample=1.,colsample=1.,minimum_leaf=10,output=str(path/'increment.npz'),minimum_free_mib=8192,own_pool_limit_bytes=12*1024**3)
        if basis is None:
            request['method']=method;result=fit_training_increment(request)
        else:
            request.update(basis=basis.tolist(),class_importance=importance.tolist(),residual_fraction=.02 if method=='direct_ABL' else 0.,
                           full_coordinates=False,cumulative_quota=True)
            result=fit_direct(request)
        inc=native_load(result['artifact'])
        if initial is None:inc=replace(inc,base=replace(inc.base,bias=prior))
        attempt.update(status='complete',fit=result);write(out/'training_ledger.json',dict(paid_new_trees=paid,attempts=ledger,prior_paid=spec.get('prior_paid_new_trees',0)))
        event('broad_fit',case=case['dataset'],method=method,depth=depth,trees=trees,seconds=result['seconds'],paid=paid)
        return inc,result

    if spec.get('preflight_only'):
        fits=[];rng=np.random.default_rng(2904)
        for c in (6,159,983):
            x=rng.normal(size=(160,9)).astype(np.float32);xv=rng.normal(size=(40,9)).astype(np.float32)
            w=rng.normal(size=(9,c));y=(x@w>0).astype(np.float32);yv=(xv@w>0).astype(np.float32)
            data=dict(train_x=x,train_y=y,dev_x=xv,dev_y=yv);case=dict(dataset=f'synthetic{c}',seed=93000+c,outputs=c)
            b=np.linalg.qr(rng.normal(size=(c,min(64,max(1,c//2)))))[0].astype(np.float32)
            for method in ('full','sketchboost_topk_half','direct_AB','direct_ABL'):
                inc,record=fit(case,data,None,3,2,out/f'c{c}'/method,method,None if method in ('full','sketchboost_topk_half') else b,np.ones(c))
                if method=='direct_ABL':
                    trace=record['trace'];atoms=sum(t['active_leaves']*c for t in trace)
                    assert sum(t['requested_residual_count'] for t in trace)==int(np.floor(.02*atoms))
                    assert sum(t['nnz'] for t in trace)<=int(np.floor(.02*atoms))
                fits.append(dict(outputs=c,method=method,fit=record));del inc;gc.collect()
        report=dict(status='broad_preflight_complete',fits=fits,checks=checks,paid_new_trees=paid,environment=environment,test_evaluated=False)
        write(out/'report.json',report);return

    prerequisite=locked(spec['preflight']['path'],spec['preflight']['sha256']);assert prerequisite['status']=='broad_preflight_complete'
    write(out/'method_freeze.json',dict(protocol=spec['protocol'],source_manifest=spec['manifest'],preflight=spec['preflight'],
         ordinary_counts=ORDINARY_COUNTS,posthoc='train shortlist24; risk geometry; no test',direct_rank='min(ceil(C/2),64)',controller='risk_residual/4x40',
         mixed_catalog='12 ordinary byte-quantiles +12 posthoc byte-quantiles; no D-quality-based construction',calibration='2fold feature-grouped OOF, common grid; T1; refit allD'))
    for cc in manifest['cases']:
        cid=f"{cc['dataset']}_{cc['seed']}";record=dict(case=cc,dataset=manifest['datasets'][cc['dataset']]);data=data_before_test(record)
        directory=out/cid;directory.mkdir(exist_ok=True);c=cc['outputs'];reference=budget_reference(c);budget=byte_limit(reference,.7)
        folds=grouped_folds(data['dev_x'],cc['seed']+20261029);models=[];counts={};traces=[]
        def store(model,path,uid,family,raw=None,**meta):
            counts[family]=counts.get(family,0)+1
            if counts[family]>24:raise RuntimeError('family D query limit exceeded')
            path.mkdir(parents=True,exist_ok=True);raw=model.raw(data['dev_x']) if raw is None else raw
            assessment,arrays=assess(raw,data['dev_y'],folds,cfg['decision_grid'],cfg['decision_shrinkage'])
            artifact=path/'model.npz';model.save(artifact);cache=path/'raw.npz';np.savez_compressed(cache,raw=raw)
            predictions=path/'oof.npz';np.savez_compressed(predictions,prediction=arrays[0],loss=arrays[1],brier=arrays[2])
            m=dict(uid=uid,family=family,artifact=str(artifact),sha256=digest(artifact),raw=dict(path=str(cache),sha256=digest(cache)),
                   oof=dict(path=str(predictions),sha256=digest(predictions)),assessment=assessment,**block_cost(model),**meta)
            models.append(m);write(path/'model.json',m);return m
        fulls={};warm=None
        for depth in (3,5):
            inc,fit_record=fit(cc,data,None,depth,200,directory/f'fit-d{depth}');full=as_protected(inc);fulls[depth]=full
            for n in ORDINARY_COUNTS:
                model=model_prefix(full,n);m=store(model,directory/f'ordinary-d{depth}-t{n}',f'ordinary/d{depth}/t{n}','ordinary',depth=depth,fit=fit_record)
                if depth==3 and n==40:warm=model;warm_ref=m
        inc,fit_record=fit(cc,data,None,3,200,directory/'fit-sketch','sketchboost_topk_half');sketch=as_protected(inc)
        for n in np.linspace(8,200,24).astype(int):
            store(model_prefix(sketch,int(n)),directory/f'sketch-t{n}',f'sketch/t{n}','sketch',fit=fit_record)
        features=dict(train=data['train_x'],development=data['dev_x']);labels={'train':data['train_y']};ranks=ranks_for(c);r0=min(64,max(1,int(np.ceil(c*.5))))
        view,_=decode_model(fulls[3]);dense=blank(replace(view.base(),bias=view.bias.copy()))
        actions=[Action(r,cap,s) for cap in (4,8) for r in ranks for s in (0.,.02,.05)]
        pool=list(candidates(None,dense,features,labels,actions,'risk_residual',max_basis_rank=min(c,64)))
        selected=train_shortlist(pool,24,mandatory=(Action(r0,8,0.).uid,Action(r0,8,.02).uid))
        for p in selected:
            store(p['model'],directory/'posthoc'/p['uid'],'posthoc/'+p['uid'],'posthoc',raw=p['raw']['development'],rank=p['rank'],
                  residual_fraction=p['residual_fraction'],leaf_cap=p['leaf_cap'],train_risk_mse=p['train_risk_mse'])
        del pool,selected;gc.collect()
        checkpoint=warm;rank=r0;cap=8;residual=.02;steps=[];controller_queries=1
        models.append(dict(warm_ref,uid='controller/warm',family='controller',reused_assessment=warm_ref['uid']))
        counts['controller']=1
        for block in range(4):
            step=directory/'controller'/f'b{block}';inc,fit_record=fit(cc,data,checkpoint,3,40,step/'fit')
            probes=list(candidates(checkpoint,inc,features,labels,[Action(rank,8,0.,'keep'),Action(rank,4,0.,'keep')],
                                  'risk_residual',max_basis_rank=min(c,64)))
            teacher=append_protected(checkpoint,inc);teacher_raw=teacher.raw(data['train_x'])
            diagnostic=diagnose(teacher_raw-probes[0]['raw']['train'],teacher_raw-probes[1]['raw']['train'])
            previous=checkpoint.blocks[0].base.basis if len(checkpoint.blocks)==1 and checkpoint.blocks[0].base.representation=='low_rank' else None
            pool=[p for p in candidates(checkpoint,inc,features,labels,local_actions(rank,cap,residual,ranks,diagnostic),'risk_residual',previous,min(c,64)) if p['cost']['cost_bytes']<=budget]
            if not pool:pool=[p for p in candidates(checkpoint,inc,features,labels,[Action(1,4,0.)],'risk_residual',max_basis_rank=min(c,64)) if p['cost']['cost_bytes']<=budget]
            available=min(6,24-controller_queries)
            if not pool or available<=0:steps.append(dict(block=block,status='no_affordable_action_or_query'));break
            pairs=[]
            for p in pool[:available]:
                m=store(p['model'],step/p['uid'],f'controller/{block}/'+p['uid'],'controller',raw=p['raw']['development'],rank=p['rank'],
                        residual_fraction=p['residual_fraction'],leaf_cap=p['leaf_cap'],block=block,fit=fit_record)
                pairs.append((p,m));controller_queries+=1
            chosen,m=min(pairs,key=lambda pair:qkey(pair[1]));checkpoint=chosen['model'];rank=chosen['rank'];cap=chosen['leaf_cap'];residual=chosen['residual_fraction']
            steps.append(dict(block=block,uid=m['uid'],diagnostic=diagnostic,queries=controller_queries,cost_bytes=m['cost_bytes']))
            del probes,pool,pairs,teacher,teacher_raw;gc.collect()
        traces.append(dict(method='controller',steps=steps,queries_including_warm=controller_queries))
        warm_raw=warm.raw(data['train_x']);rw,cw=risk_weights(data['train_y'],warm_raw,'multilabel');b,meta=risk_basis(warm_raw-warm.bias,'multilabel',rw,cw,r0)
        for method in ('direct_AB','direct_ABL'):
            inc,fit_record=fit(cc,data,warm,3,160,directory/method/'fit',method,b,cw)
            for n in np.linspace(1,160,24).astype(int):
                model=append_protected(warm,prefix(inc,int(n)))
                store(model,directory/method/f't{n+40}',method+f'/t{n+40}',method,rank=r0,residual_fraction=.02 if method=='direct_ABL' else 0.,fit=fit_record)
        case=dict(case=cid,record=record,task='multilabel',models=models,traces=traces,queries=counts,
                  analytic_reference_bytes=reference,controller_limit_bytes=budget,reference_bytes=block_cost(fulls[3])['cost_bytes'],warmup=warm_ref)
        file=directory/'case.json';write(file,case);entries.append(dict(case=cid,path=str(file),sha256=digest(file)))
        write(out/'progress.json',dict(cases=entries,paid_new_trees=paid));event('broad_case_complete',case=cid,models=len(models),paid=paid,queries=counts)
        del models,fulls,sketch,checkpoint;gc.collect();memory_guard()
    cases=[locked(r['path'],r['sha256']) for r in entries];seal=select_cases(cases)
    seal.update(source_manifest=spec['manifest'],cases=entries,method_freeze=dict(path=str(out/'method_freeze.json'),sha256=digest(out/'method_freeze.json')))
    write(out/'pretest_seal.json',seal)
    report=dict(status='broad_training_complete',cases=entries,paid_new_trees=paid,environment=environment,checks=checks,test_evaluated=False,
                pretest_seal=dict(path=str(out/'pretest_seal.json'),sha256=digest(out/'pretest_seal.json')),elapsed_seconds=perf_counter()-started)
    write(out/'report.json',report);event('broad_training_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'),paid=paid)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
