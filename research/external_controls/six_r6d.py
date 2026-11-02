"""Fixed research catalogues and D-only exact bank allocation."""
from pathlib import Path
import gc,json,sys
import numpy as np
from budget_quality_worker import write,locked,digest,event
from abl_quality_worker import check_native_environment
from six_common import load_cases,load_model,price
from abl_transfer_study import data_before_test
from six_quant_core import prepare_quant,QuantSnapshot
from six_r5 import quant_raw
from broad_core import assess,grouped_folds
from unified_serving_core import bank_frontier
from residual_gate_core import sufficient,quality


def run(cfg,out):
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=12*1024**3);out.mkdir(parents=True,exist_ok=True);env=check_native_environment(cfg)
    _,cases=load_cases(cfg,first=True);spec=cfg['six'];sources={}
    for key in ('R1_seal','R2','R3'):
        report=locked(spec[key]['path'],spec[key]['sha256']);sources[key]={r['case']:r for r in report['cases']}
    refs=[];catalogue=[];full_bytes=0
    event('R6D_protocol',tasks=12,catalogue_queries=dict(ordinary_int8=24,compact=24,mixed=24),new_trees=0,
          caveat='exploratory fixed portfolio chosen after previous research; equal current catalogue sizes do not erase upstream research or training cost')
    for case in cases:
        name=case['case'];data=data_before_test(case['record']);directory=out/name;directory.mkdir(exist_ok=True)
        first=locked(sources['R1_seal'][name]['path'],sources['R1_seal'][name]['sha256']);r2=locked(sources['R2'][name]['path'],sources['R2'][name]['sha256']);r3=locked(sources['R3'][name]['path'],sources['R3'][name]['sha256'])
        ordinary=sorted((m for m in first['models'] if m['family']=='ordinary'),key=lambda m:(m['packed_bytes'],m['uid']));assert len(ordinary)==24
        fixed=[(r2,f'compressed40_ABL/t{n}') for n in (80,120,160,200)]
        fixed += [(r3,f'refit/t{n}') for n in (50,100)]
        fixed += [(r3,f'refit_ABL/t{n}') for n in (50,100,150,200)]
        fixed += [(r3,f'coarse_refit_ABL/t{n}') for n in (50,100)]
        compact=[next(m for m in src['models'] if m['uid']==uid) for src,uid in fixed];models=[]
        ordinary_half=set(np.linspace(0,len(ordinary)-1,12).astype(int).tolist())
        for kind,pool in (('ordinary',ordinary),('compact',compact)):
            for index,original in enumerate(pool):
                model=load_model(original)
                for codec in (('int8',) if kind=='ordinary' else ('fp16','int8')):
                    uid=f'{kind}/{index}/{codec}';snap=prepare_quant(model,np.zeros(len(model.bias),np.float32),codec)
                    raw=quant_raw(snap,data['dev_x']);folds=grouped_folds(data['dev_x'],case['record']['case']['seed']+20261029)
                    assessment,_=assess(raw,data['dev_y'],folds,cfg['decision_grid'],cfg['decision_shrinkage'])
                    snap.arrays['shifts']=np.asarray(assessment['refit_control']['shifts'],np.float32)
                    p=directory/(uid.replace('/','-')+'.npz');snap.save(p)
                    belongs=['ordinary_int8'] if kind=='ordinary' else ['compact']
                    if (kind=='ordinary' and index in ordinary_half) or (kind=='compact' and codec=='int8'):belongs.append('mixed')
                    models.append(dict(uid=uid,artifact=str(p),sha256=digest(p),assessment=assessment,packed_bytes=snap.resident_bytes+8,
                        codec=codec,source_uid=original['uid'],source_sha256=original['sha256'],catalogues=belongs))
                del model;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        full=load_model(next(m for m in case['models'] if m['uid']=='ordinary/d3/t200'));size=price(full);full_bytes+=size
        p=directory/'development.json';item=dict(case=name,record=case['record'],models=models,full_bytes=size);write(p,item);refs.append(dict(case=name,path=str(p),sha256=digest(p)));catalogue.append(item)
        event('R6D_catalogue',case=name,models=len(models),catalogues={k:sum(k in m['catalogues'] for m in models) for k in ('ordinary_int8','compact','mixed')})
    ratios=(.125,.25,.5,1.);budgets=[int(full_bytes*r) for r in ratios];banks=[]
    for traffic in ('uniform','skew_priority'):
        frequency=np.ones(len(cases)) if traffic=='uniform' else 1/(np.arange(len(cases))+1.)**1.2
        importance=np.ones(len(cases));importance[:3]=1. if traffic=='uniform' else 2.
        task_weights=frequency*importance;task_weights/=task_weights.sum()
        for family in ('ordinary_int8','compact','mixed'):
            options=[[dict(uid=m['uid'],cost_bytes=m['packed_bytes'],utility=float(len(cases)*task_weights[i]*m['assessment']['quality']['f1_macro']))
                      for m in case['models'] if family in m['catalogues']] for i,case in enumerate(catalogue)]
            for ratio,bank in zip(ratios,bank_frontier(options,budgets)):
                bank.update(family=family,traffic=traffic,ratio=ratio,cases=[c['case'] for c in catalogue],task_weights=task_weights.tolist(),frequency=(frequency/frequency.sum()).tolist())
                banks.append(bank)
    seal=out/'pretest_seal.json';write(seal,dict(cases=refs,banks=banks,full_bank_bytes=full_bytes,test_opened=False));event('R6D_seal',sha256=digest(seal));results=[]
    for case in catalogue:
        cc,dd=case['record']['case'],case['record']['dataset'];assert digest(cc['split_file'])==cc['split_sha256'] and digest(dd['path'])==dd['sha256']
        with np.load(cc['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
        with np.load(dd['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
        rows=[]
        for m in case['models']:
            s=QuantSnapshot.load(m['artifact']);raw=quant_raw(s,x);q=quality(y,*sufficient(y,raw,'multilabel',m['assessment']['refit_control']),'multilabel',s.outputs)
            rows.append(dict(uid=m['uid'],quality=q,packed_bytes=m['packed_bytes']))
        p=out/case['case']/'test.json';write(p,dict(case=case['case'],models=rows));results.append(dict(case=case['case'],models=rows));event('R6D_test',case=case['case'],models=len(rows))
    by={c['case']:{m['uid']:m for m in c['models']} for c in results}
    for bank in banks:
        if bank['status']=='feasible':
            quality_per_task=[by[c][uid]['quality']['f1_macro'] for c,uid in zip(bank['cases'],bank['choices'])]
            bank['test_equal_task_macro_f1']=float(np.mean(quality_per_task));bank['test_weighted_macro_f1']=float(np.asarray(quality_per_task)@bank['task_weights'])
    write(out/'report.json',dict(status='R6D_complete',cases=refs,banks=banks,results=results,full_bank_bytes=full_bytes,new_trees=0,environment=env,pretest_seal=dict(path=str(seal),sha256=digest(seal))))
    event('R6D_complete',sha256=digest(out/'report.json'))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
