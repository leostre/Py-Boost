"""R1: ordinary controls densely matched to actual compact GPU bytes."""
from pathlib import Path
import gc,json,sys,time
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment
from broad_core import model_prefix,qkey,select_cases
from six_common import load_cases,load_model,price,store_model,test_case
from abl_transfer_study import data_before_test

def run(cfg,out):
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=12*1024**3)
    out.mkdir(parents=True,exist_ok=True);environment=check_native_environment(cfg);spec=cfg['six'];started=time.perf_counter()
    source,cases=load_cases(cfg);oldseal=locked(source['pretest_seal']['path'],source['pretest_seal']['sha256']);records=[]
    event('R1_protocol',stage=spec['stage'],cases=len(cases),maximum_D_candidates=24,matching_fraction=.05,new_trees=0)
    for case in cases:
        data=data_before_test(case['record']);directory=out/case['case'];directory.mkdir(exist_ok=True)
        by={m['uid']:m for m in case['models']};targets=[]
        for row in oldseal['selected']:
            if row['case']==case['case'] and row['family'] in ('direct_AB','direct_ABL') and row['uid']:
                m=dict(by[row['uid']]);model=load_model(m);m['packed_bytes']=price(model)
                targets.append(dict(m=m,ratio=row['ratio']));del model
        fulls={depth:load_model(by[f'ordinary/d{depth}/t200']) for depth in (3,5)}
        pool=[]
        for depth,full in fulls.items():
            for count in range(1,201):
                model=model_prefix(full,count);pool.append(dict(uid=f'ordinary/d{depth}/t{count}',depth=depth,count=count,packed_bytes=price(model)));del model
        chosen={}
        for target in targets:
            for depth in (3,5):
                option=min((p for p in pool if p['depth']==depth),key=lambda p:(abs(np.log(p['packed_bytes']/target['m']['packed_bytes'])),p['count']))
                chosen[option['uid']]=option
        order=sorted(pool,key=lambda p:(p['packed_bytes'],p['uid']))
        for i in np.linspace(0,len(order)-1,24).astype(int):
            if len(chosen)>=24:break
            chosen[order[i]['uid']]=order[i]
        controls=[]
        for p in chosen.values():
            if p['uid'] in by:m=dict(by[p['uid']],packed_bytes=p['packed_bytes'],reused_D_assessment=True)
            else:
                model=model_prefix(fulls[p['depth']],p['count']);m=store_model(model,p['uid'],'ordinary',data,cfg,case,directory/p['uid'].replace('/','-'),depth=p['depth']);del model
            controls.append(m)
        pairs=[]
        for target in targets:
            a=target['m'];near=[m for m in controls if abs(m['packed_bytes']/a['packed_bytes']-1)<=.05]
            b=min(near,key=qkey) if near else None
            pairs.append(dict(case=case['case'],dataset=case['record']['case']['dataset'],family=a['family'],ratio=target['ratio'],A=a['uid'],B=b['uid'] if b else None,
                A_bytes=a['packed_bytes'],B_bytes=b['packed_bytes'] if b else None,relative_byte_gap=b['packed_bytes']/a['packed_bytes']-1 if b else None,
                D_delta_pp=100*(a['assessment']['quality']['f1_macro']-b['assessment']['quality']['f1_macro']) if b else None,status='matched' if b else 'no_5percent_match'))
        unique={m['uid']:m for m in controls+[t['m'] for t in targets]}
        item=dict(case=case['case'],record=case['record'],models=list(unique.values()),pairs=pairs,ordinary_queries=len(controls),original_family_queries=24)
        path=directory/'development.json';write(path,item);records.append(dict(path=str(path),sha256=digest(path),case=case['case']))
        write(out/'progress.json',dict(cases=records,test_opened=False));event('R1_development',case=case['case'],matched=sum(p['status']=='matched' for p in pairs),pairs=len(pairs),ordinary_queries=len(controls))
        del fulls,data;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    seal=out/'pretest_seal.json';write(seal,dict(cases=records,source=spec['training'],selection_uses_test=False));event('R1_seal',sha256=digest(seal))
    allpairs=[]
    for ref in records:
        case=locked(ref['path'],ref['sha256']);requested={p[k] for p in case['pairs'] for k in ('A','B') if p[k]}
        results=test_case(case,[m for m in case['models'] if m['uid'] in requested],out/case['case']/'test.json');by={m['uid']:m for m in results}
        for p in case['pairs']:
            if p['B']:p.update(test_delta_pp=100*(by[p['A']]['quality']['f1_macro']-by[p['B']]['quality']['f1_macro']),test_A=by[p['A']]['quality'],test_B=by[p['B']]['quality'])
            allpairs.append(p)
        event('R1_test',case=case['case'],pairs=len(case['pairs']))
    summary=[]
    for family in ('direct_AB','direct_ABL'):
        for ratio in (.25,.5,.7,1.):
            rows=[p for p in allpairs if p['family']==family and p['ratio']==ratio and p['B']];names=sorted({p['dataset'] for p in rows})
            means=[np.mean([p['test_delta_pp'] for p in rows if p['dataset']==n]) for n in names]
            row=dict(family=family,ratio=ratio,pairs=len(rows),datasets=len(names),mean_dataset_delta_pp=float(np.mean(means)) if means else None,positive_datasets=int(sum(v>0 for v in means)));summary.append(row);event('R1_result',**row)
    result=dict(status='R1_complete',cases=records,pairs=allpairs,summary=summary,pretest_seal=dict(path=str(seal),sha256=digest(seal)),new_trees=0,environment=environment,seconds=time.perf_counter()-started,
        limitations=['12 historically inspected test datasets; three partitions share one test','target byte anchors from previous D-selected compact models; ordinary catalogue24 is chosen without its D scores','5percent is byte-matching tolerance, not quality acceptance','no universal predictor validated from12 datasets'])
    write(out/'report.json',result);event('R1_complete',report=str(out/'report.json'),sha256=digest(out/'report.json'),new_trees=0)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
