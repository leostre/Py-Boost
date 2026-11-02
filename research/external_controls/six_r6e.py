"""Stronger dense-refit ordinary bank; zero extra trees and equal24 options."""
from pathlib import Path
import json,sys,gc
import numpy as np
from budget_quality_worker import write,locked,digest,event
from abl_quality_worker import check_native_environment
from six_common import load_model
from abl_transfer_study import data_before_test
from six_quant_core import prepare_quant,QuantSnapshot
from six_r5 import quant_raw
from broad_core import grouped_folds,assess
from residual_gate_core import sufficient,quality
from unified_serving_core import bank_frontier


def run(cfg,out):
    import cupy as cp
    out.mkdir(parents=True,exist_ok=True);cp.get_default_memory_pool().set_limit(size=12*1024**3);env=check_native_environment(cfg)
    source=locked(cfg['six']['R6D']['path'],cfg['six']['R6D']['sha256']);r3=locked(cfg['six']['R3']['path'],cfg['six']['R3']['sha256'])
    r3by={r['case']:r for r in r3['cases']};cases=[];refs=[]
    for ref in source['cases']:
        case=locked(ref['path'],ref['sha256']);name=case['case'];data=data_before_test(case['record']);directory=out/name;directory.mkdir(exist_ok=True)
        rows=case['models'];ordinary=[m for m in rows if 'ordinary_int8' in m['catalogues']]
        ordinary.sort(key=lambda m:(m['packed_bytes'],m['uid']))
        for i in np.linspace(0,len(ordinary)-1,20).astype(int):ordinary[i]['catalogues'].append('ordinary_refit')
        for m in rows:
            if m['codec']=='int8' and m['source_uid'] in ('refit/t50','refit/t100'):m['catalogues'].append('ordinary_refit')
        ref3=r3by[name];saved=locked(ref3['path'],ref3['sha256']);new=[]
        for n in (150,200):
            original=next(m for m in saved['models'] if m['uid']==f'refit/t{n}');model=load_model(original);snap=prepare_quant(model,np.zeros(len(model.bias),np.float32),'int8')
            raw=quant_raw(snap,data['dev_x']);folds=grouped_folds(data['dev_x'],case['record']['case']['seed']+20261029)
            assessment,_=assess(raw,data['dev_y'],folds,cfg['decision_grid'],cfg['decision_shrinkage']);snap.arrays['shifts']=np.asarray(assessment['refit_control']['shifts'],np.float32)
            p=directory/f'refit{n}-int8.npz';snap.save(p);item=dict(uid=f'enhanced/refit{n}',artifact=str(p),sha256=digest(p),assessment=assessment,
                  packed_bytes=snap.resident_bytes+8,codec='int8',source_uid=original['uid'],source_sha256=original['sha256'],catalogues=['ordinary_refit'])
            rows.append(item);new.append(item['uid']);del model;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        assert sum('ordinary_refit' in m['catalogues'] for m in rows)==24
        case['new_uids']=new;p=directory/'development.json';write(p,case);refs.append(dict(case=name,path=str(p),sha256=digest(p)));cases.append(case)
    banks=[]
    for traffic in ('uniform','skew_priority'):
        weights=np.asarray(next(b for b in source['banks'] if b['traffic']==traffic)['task_weights']);frequency=next(b for b in source['banks'] if b['traffic']==traffic)['frequency']
        for family in ('ordinary_int8','ordinary_refit','compact','mixed'):
            options=[[dict(uid=m['uid'],cost_bytes=m['packed_bytes'],utility=float(len(cases)*weights[i]*m['assessment']['quality']['f1_macro'])) for m in c['models'] if family in m['catalogues']] for i,c in enumerate(cases)]
            for ratio,bank in zip((.125,.25,.5,1.),bank_frontier(options,[int(source['full_bank_bytes']*r) for r in (.125,.25,.5,1.)])):
                bank.update(family=family,traffic=traffic,ratio=ratio,cases=[c['case'] for c in cases],task_weights=weights.tolist(),frequency=frequency);banks.append(bank)
    seal=out/'pretest_seal.json';write(seal,dict(cases=refs,banks=banks,test_opened=False));event('R6E_seal',sha256=digest(seal));results=[]
    old={c['case']:c['models'] for c in source['results']}
    for case in cases:
        cc,dd=case['record']['case'],case['record']['dataset'];assert digest(cc['split_file'])==cc['split_sha256'] and digest(dd['path'])==dd['sha256']
        with np.load(cc['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
        with np.load(dd['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
        rows=list(old[case['case']])
        for m in case['models']:
            if m['uid'] not in case['new_uids']:continue
            s=QuantSnapshot.load(m['artifact']);raw=quant_raw(s,x);q=quality(y,*sufficient(y,raw,'multilabel',m['assessment']['refit_control']),'multilabel',s.outputs)
            rows.append(dict(uid=m['uid'],quality=q,packed_bytes=m['packed_bytes']))
        results.append(dict(case=case['case'],models=rows))
    by={c['case']:{m['uid']:m for m in c['models']} for c in results}
    for bank in banks:
        if bank['status']=='feasible':
            q=np.asarray([by[c][uid]['quality']['f1_macro'] for c,uid in zip(bank['cases'],bank['choices'])]);bank.update(test_equal_task_macro_f1=float(q.mean()),test_weighted_macro_f1=float(q@bank['task_weights']))
    write(out/'report.json',dict(status='R6E_complete',cases=refs,banks=banks,results=results,full_bank_bytes=source['full_bank_bytes'],new_trees=0,environment=env,
                reused_source=cfg['six']['R6D'],pretest_seal=dict(path=str(seal),sha256=digest(seal))))
    event('R6E_complete',sha256=digest(out/'report.json'))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
