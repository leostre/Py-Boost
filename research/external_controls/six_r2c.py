"""R2: warm-start size, compact warm start, and causal blockwise residual gate."""
from pathlib import Path
from dataclasses import replace
import gc,json,sys,time
import numpy as np
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_worker import check_native_environment
from abl_protected_core import append_protected
from adaptive_basis_core import risk_weights,risk_basis
from adaptive_controller_core import candidates,Action
from broad_core import model_prefix,qkey
from direct_leaf_worker import fit_direct
from six_common import load_cases,load_model,store_model,test_case,gpu_raw,price
from abl_transfer_study import data_before_test

def weaken(inc,alpha):
    values=(inc.values*alpha).astype(np.float32);keep=values!=0
    return replace(inc,rows=inc.rows[keep].copy(),columns=inc.columns[keep].copy(),values=values[keep])

def run(cfg,out):
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=12*1024**3);out.mkdir(parents=True,exist_ok=True)
    env=check_native_environment(cfg);source,cases=load_cases(cfg,first=False);spec=cfg['six'];paid=0;ledger=[];refs=[];started=time.perf_counter()
    first=set();remaining=[]
    for case in cases:
        name=case['record']['case']['dataset']
        if name in first:remaining.append(case)
        first.add(name)
    cases=remaining
    locked(spec['R1']['path'],spec['R1']['sha256'])
    event('R2_protocol',cases=len(cases),new_tree_limit=3840,gate_alphas=[0.,.5,1.],prefixes=[80,120,160,200],ordinary_warm_reused=True,
          gate_control='both gate and alpha1 control use four40-tree blocks; no single160 vs blockwise confounding',test_opened=False)
    for case in cases:
        data=data_before_test(case['record']);directory=out/case['case'];directory.mkdir(exist_ok=True);c=case['record']['case']['outputs'];rank=min(64,int(np.ceil(c/2)))
        by={m['uid']:m for m in case['models']};full=load_model(by['ordinary/d3/t200']);warm40=model_prefix(full,40);warm20=model_prefix(full,20)
        raw40=gpu_raw(warm40,data['train_x']);rw,cw=risk_weights(data['train_y'],raw40,'multilabel');b40,bmeta=risk_basis(raw40-warm40.bias,'multilabel',rw,cw,rank);b40=b40[:,:rank]
        compressed=next(candidates(None,warm40,dict(train=data['train_x'],development=data['dev_x']),dict(train=data['train_y']),[Action(rank,8,0.)],'risk_residual',max_basis_rank=rank))['model']
        raw20=gpu_raw(warm20,data['train_x']);r20,c20=risk_weights(data['train_y'],raw20,'multilabel');b20,_=risk_basis(raw20-warm20.bias,'multilabel',r20,c20,rank);b20=b20[:,:rank]
        datafile=directory/'fit-data.npz';np.savez(datafile,train_x=data['train_x'],train_y=data['train_y'],validation_x=data['dev_x'],validation_y=data['dev_y'])
        models=[];traces=[]
        for family in ('direct_AB','direct_ABL'):
            baseline=load_model(by[family+'/t200'])
            for n in (80,120,160,200):
                m=store_model(model_prefix(baseline,n),'baseline-'+family+f'/t{n}','baseline-'+family,data,cfg,case,directory/('baseline-'+family)/str(n),rank=rank)
                models.append(m)
            del baseline
        modes=[('short20_AB',warm20,b20,c20,0.,False,False),('short20_ABL',warm20,b20,c20,.02,False,False),
               ('compressed40_AB',compressed,b40,cw,0.,False,False),('compressed40_ABL',compressed,b40,cw,.02,False,False),
               ('blockwise40_ABL',warm40,b40,cw,.02,True,False),('gated40_ABL',warm40,b40,cw,.02,True,True)]
        for mode,initial,basis,importance,residual,blockwise,gate in modes:
            if mode!='compressed40_ABL':continue
            checkpoint=initial;mode_dir=directory/mode;mode_dir.mkdir(exist_ok=True);steps=[]
            for block in range(4 if blockwise else 1):
                k=40 if blockwise else 200-initial.n_trees
                if paid+k>3840:raise RuntimeError('R2 tree budget exceeded')
                paid+=k;step=mode_dir/f'b{block}';step.mkdir(exist_ok=True)
                init=step/'initial.npz';train=gpu_raw(checkpoint,data['train_x']);validation=gpu_raw(checkpoint,data['dev_x']);np.savez(init,train=train.astype(np.float32),validation=validation)
                request=dict(data_path=str(datafile),initial_path=str(init),basis=basis.tolist(),class_importance=np.asarray(importance).tolist(),residual_fraction=residual,
                    full_coordinates=False,cumulative_quota=True,task='multilabel',seed=case['record']['case']['seed']+101000+checkpoint.n_trees,
                    trees=k,lr=cfg['lr'],subsample=1.,colsample=1.,minimum_leaf=10,config={**cfg,'max_depth':3},output=str(step/'increment.npz'),own_pool_limit_bytes=12*1024**3)
                attempt=dict(case=case['case'],mode=mode,block=block,trees=k,status='reserved');ledger.append(attempt);write(out/'training_ledger.json',dict(new_trees=paid,attempts=ledger))
                fit=fit_direct(request);inc=native_load(fit['artifact']);attempt.update(status='complete',fit=fit)
                write(out/'training_ledger.json',dict(new_trees=paid,attempts=ledger))
                if blockwise:
                    options=[]
                    for alpha in ((0.,.5,1.) if gate else (1.,)):
                        model=append_protected(checkpoint,weaken(inc,alpha));m=store_model(model,f'{mode}/b{block}/a{alpha:g}',mode,data,cfg,case,step/f'a{alpha:g}',rank=rank,alpha=alpha,fit=fit)
                        models.append(m);options.append((model,m))
                    checkpoint,m=min(options,key=lambda pair:qkey(pair[1]));steps.append(dict(trees=checkpoint.n_trees,chosen=m['uid'],alpha=m['alpha'],D_quality=m['assessment']['quality'],packed_bytes=m['packed_bytes']))
                    del options
                else:
                    for n in (80,120,160,200):
                        from experiments.parallel_leaf_core import prefix
                        model=append_protected(initial,prefix(inc,n-initial.n_trees));m=store_model(model,f'{mode}/t{n}',mode,data,cfg,case,step/str(n),rank=rank,fit=fit)
                        models.append(m);steps.append(dict(trees=n,chosen=m['uid'],D_quality=m['assessment']['quality'],packed_bytes=m['packed_bytes']))
                event('R2_fit',case=case['case'],mode=mode,block=block,new_trees=paid,fit_seconds=fit['seconds'],last=steps[-1])
                del inc,train,validation;gc.collect();cp.get_default_memory_pool().free_all_blocks()
            traces.append(dict(mode=mode,steps=steps,D_queries=12 if gate else 4))
        item=dict(case=case['case'],record=case['record'],models=models,traces=traces,rank=rank,
            warm_storage=dict(full40=price(warm40),full20=price(warm20),compressed40=price(compressed)),
            limitations=['warm prefixes reused from old full fits; measured time is continuation, not end-to-end fit','adaptive gate12D assessments versus4for its fixed blockwise control; no pooled best-of-all-modes claim'])
        p=directory/'development.json';write(p,item);refs.append(dict(case=case['case'],path=str(p),sha256=digest(p)));write(out/'progress.json',dict(cases=refs,new_trees=paid,test_opened=False))
        del data,full,warm40,warm20,compressed;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    seal=out/'pretest_seal.json';write(seal,dict(cases=refs,new_trees=paid,test_opened=False));event('R2_seal',sha256=digest(seal),new_trees=paid)
    results=[]
    for ref in refs:
        case=locked(ref['path'],ref['sha256']);rows=test_case(case,case['models'],out/case['case']/'test.json')
        results.append(dict(case=case['case'],models=rows,traces=case['traces'],warm_storage=case['warm_storage']))
        for mode in ('compressed40_ABL',):
            trace=next(t for t in case['traces'] if t['mode']==mode);last=next(m for m in rows if m['uid']==trace['steps'][-1]['chosen'])
            base=next(m for m in rows if m['uid']==('baseline-direct_ABL/t200' if 'ABL' in mode else 'baseline-direct_AB/t200'))
            event('R2_result',case=case['case'],mode=mode,test_delta_pp=100*(last['quality']['f1_macro']-base['quality']['f1_macro']),bytes_ratio=last['packed_bytes']/base['packed_bytes'],D_queries=trace['D_queries'])
    result=dict(status='R2_complete',cases=refs,results=results,new_trees=paid,pretest_seal=dict(path=str(seal),sha256=digest(seal)),environment=env,seconds=time.perf_counter()-started)
    write(out/'report.json',result);event('R2_complete',report=str(out/'report.json'),sha256=digest(out/'report.json'),new_trees=paid)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
