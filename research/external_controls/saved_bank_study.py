"""Stage I: saved models only, nested calibration/selection, then old test replay."""
from pathlib import Path
from time import perf_counter
import json,sys,gc,hashlib
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,block_cost
from unified_serving_core import coalesce
from unified_frontier_core import make_folds,replay_calibration,variants,sufficient,metrics
from saved_bank_core import SELECTORS,policy_summary,score_options,allocate


def assessment(z,y,task,seed,count,cfg,out):
    replays=[]
    for repeat in range(3):
        replay=replay_calibration(z,y,task,make_folds(y,task,seed+repeat*101,count),cfg['decision_grid'],cfg['decision_shrinkage'])
        # Store actual fold controls and scores, not duplicated prediction arrays.
        replays.append({k:replay[k] for k in ('scores','states')})
    summaries=policy_summary(replays,task)
    fitted=variants(z,y,z[:1],y[:1],task,cfg['decision_grid'],cfg['decision_shrinkage'])
    for name in summaries:summaries[name]['control']=fitted[summaries[name]['policy']]
    write(out,dict(replays=replays,summaries=summaries,seed=seed,fold_count=count))
    return summaries


def primary_anchor(case):
    ordinary=[m for m in case['models'] if m['family']=='ordinary']
    # Frozen structural reference, not the winner on D or test.
    if any(m['uid']=='ordinary/d3/t200' for m in ordinary):return next(m for m in ordinary if m['uid']=='ordinary/d3/t200')
    return max(ordinary,key=lambda m:(m.get('trees',0),m['cost_bytes'],m['uid']))


def run(cfg,out):
    started=perf_counter();out.mkdir(parents=True,exist_ok=True);spec=cfg['saved_banks'];environment=check_native_environment(cfg)
    fresh=locked(spec['fresh']['path'],spec['fresh']['sha256']);legacy=locked(spec['legacy']['path'],spec['legacy']['sha256'])
    entries=list(fresh['cases'])+[r for r in legacy['case_representations'] if r['case'] in spec['legacy_cases']]
    cases={};datasets={};prepared=[];audits=[];prediction_cache={}
    allowed={'ordinary','posthoc','online','controller','fixed_risk','escape_risk'}
    for ref in entries:
        source=locked(ref['path'],ref['sha256']);cid=source['case'];data=data_before_test(source['record']);task=source['task'];directory=out/cid;directory.mkdir(exist_ok=True)
        source['models']=[m for m in source['models'] if m['family'] in allowed];anchor_uid=primary_anchor(source)['uid']
        datasets[cid]=data
        folds=make_folds(data['dev_y'],task,source['record']['case']['seed']+20261028);models=[];checkmax=0.;calibration_cache={}
        for index,m in enumerate(source['models']):
            assert digest(m['artifact'])==m['sha256'];assert digest(m['raw']['path'])==m['raw']['sha256']
            with np.load(m['raw']['path'],allow_pickle=False) as f:z=f['raw'].copy()
            model=coalesce(load_protected(m['artifact']));ids=np.linspace(0,len(z)-1,min(17,len(z))).astype(int)
            error=float(np.max(np.abs(model.raw(data['dev_x'][ids])-z[ids])));checkmax=max(checkmax,error);assert error<2e-5
            cost=block_cost(model)['cost_bytes'];p=directory/f'm{index:03d}';p.mkdir(exist_ok=True)
            signature=hashlib.sha256(z.tobytes()).hexdigest()
            if signature in calibration_cache:
                previous=calibration_cache[signature]
                models.append(dict(previous,uid=m['uid'],family=m['family'],artifact=m['artifact'],sha256=m['sha256'],cost_bytes=cost,raw=m['raw'],calibration_reused_from=previous['uid']))
                continue
            full=assessment(z,data['dev_y'],task,source['record']['case']['seed']+20261028,3,cfg,p/'full.json')
            outer=[]
            for k,f in enumerate(folds):
                a,b=np.asarray(f.fitting),np.asarray(f.validation)
                assessed=assessment(z[a],data['dev_y'][a],task,source['record']['case']['seed']+20261120+k*1000,2,cfg,p/f'outer{k}.json')
                arrays={};evaluation={}
                for selector in SELECTORS:
                    control=assessed[selector]['control'];pred,loss,brier=sufficient(data['dev_y'][b],z[b],task,control['temperature'],control['shifts'])
                    arrays.update({selector+'__prediction':pred,selector+'__loss':loss,selector+'__brier':brier})
                    evaluation[selector]=metrics(data['dev_y'][b],pred,loss,brier,task,z.shape[1])
                file=p/f'outer{k}-predictions.npz';np.savez_compressed(file,**arrays)
                outer.append(dict(fitting=list(f.fitting),validation=list(f.validation),summaries=assessed,evaluation=evaluation,
                    predictions=dict(path=str(file),sha256=digest(file)),assessment=dict(path=str(p/f'outer{k}.json'),sha256=digest(p/f'outer{k}.json'))))
            models.append(dict(uid=m['uid'],family=m['family'],artifact=m['artifact'],sha256=m['sha256'],cost_bytes=cost,
                summaries=full,outer=outer,raw=m['raw'],assessment=dict(path=str(p/'full.json'),sha256=digest(p/'full.json')),raw_signature=signature))
            calibration_cache[signature]=models[-1]
            if index%10==0:event('saved_bank_model',case=cid,index=index,total=len(source['models']))
            del model,z;gc.collect()
        case=dict(case=cid,task=task,record=source['record'],models=models,anchor_uid=anchor_uid,outer_folds=[dict(fitting=list(f.fitting),validation=list(f.validation)) for f in folds])
        file=directory/'case.json';write(file,case);prepared.append(dict(case=cid,path=str(file),sha256=digest(file)))
        cases[cid]=case;audits.append(dict(case=cid,models=len(models),unique_raw_functions=len(calibration_cache),maximum_raw_error=checkmax,status='passed'))
        write(out/'progress.json',dict(cases=prepared,audits=audits));event('saved_bank_case_complete',case=cid,models=len(models));memory_guard()
    banks=[]
    first=['emotions_6104','medical_6204','pendigits_6304','satimage_6404']
    for offset in (0,10,20):banks.append((f'previous_repeat{offset//10}',[x.rsplit('_',1)[0]+'_'+str(int(x.rsplit('_',1)[1])+offset) for x in first]))
    banks += [('small_multilabel',['scene_2104','yeast_1304','birds_2204','medical_6204']),
        ('wide',['mediamill_1104','bibtex_2304','helena_1204','mnist_3104'])]
    banks.append(('combined',list(dict.fromkeys(first+banks[-2][1]+banks[-1][1]))))
    allocations=[];nested=[]
    for name,ids in banks:
        budget_base=sum(next(m['cost_bytes'] for m in cases[cid]['models'] if m['uid']==cases[cid]['anchor_uid']) for cid in ids)
        budgets=[int(r*budget_base) for r in spec['ratios']]
        for cap in spec['catalogue_caps']:
            for selector in SELECTORS:
                full_options=[];outer_options=[[] for _ in range(3)]
                for cid in ids:
                    case=cases[cid];anchor=next(m for m in case['models'] if m['uid']==case['anchor_uid'])
                    full_options.append(score_options(case['models'],selector,anchor,case['task']))
                    for k in range(3):
                        mm=[dict(m,summaries=m['outer'][k]['summaries']) for m in case['models']];aa=next(m for m in mm if m['uid']==case['anchor_uid'])
                        outer_options[k].append(score_options(mm,selector,aa,case['task']))
                for strategy in ('ordinary_equal','ordinary','compact','mixed'):
                    full_rows=allocate(full_options,budgets,strategy,cap)
                    fold_rows=[allocate(options,budgets,strategy,cap) for options in outer_options]
                    for i,a in enumerate(full_rows):
                        item=dict(bank=name,cases=ids,selector=selector,strategy=strategy,cap=cap,allocation=a)
                        allocations.append(item)
                        folds=[fold_rows[k][i] for k in range(3)]
                        result=dict(bank=name,cases=ids,selector=selector,strategy=strategy,cap=cap,budget=a['budget'],fold_allocations=folds)
                        if all(f['status']=='feasible' for f in folds):
                            qualities=[]
                            for j,cid in enumerate(ids):
                                case=cases[cid];data=datasets[cid];pred=[];loss=[];brier=[];validation=[]
                                for k,allocation in enumerate(folds):
                                    m=next(m for m in case['models'] if m['uid']==allocation['choices'][j]);ref=m['outer'][k]['predictions'];assert digest(ref['path'])==ref['sha256']
                                    cache_key=(ref['path'],selector)
                                    if cache_key not in prediction_cache:
                                        with np.load(ref['path'],allow_pickle=False) as f:prediction_cache[cache_key]=tuple(f[selector+'__'+key].copy() for key in ('prediction','loss','brier'))
                                    pp,ll,bb=prediction_cache[cache_key]
                                    pred.append(pp);loss.append(ll);brier.append(bb);validation.extend(m['outer'][k]['validation'])
                                q=metrics(data['dev_y'][validation],np.concatenate(pred),np.concatenate(loss),np.concatenate(brier),case['task'],case['record']['case']['outputs'])
                                qualities.append(dict(case=cid,quality=q,chosen=[f['choices'][j] for f in folds]))
                            result.update(status='feasible',quality=qualities)
                        else:result['status']='infeasible'
                        nested.append(result)
        event('saved_bank_allocations',bank=name,records=len(allocations))
    # Freeze every all-D choice before any old test evaluation.
    seal=dict(cases=prepared,allocations=allocations,nested=nested,source_fresh=spec['fresh'],source_legacy=spec['legacy'],test_used_for_selection=False)
    write(out/'pretest_seal.json',seal);test_cache={};test_rows=[]
    for row in allocations:
        allocation=row['allocation']
        if allocation['status']!='feasible':continue
        evaluated=[]
        for cid,uid in zip(row['cases'],allocation['choices']):
            key=(cid,uid,row['selector']);case=cases[cid];m=next(m for m in case['models'] if m['uid']==uid)
            if key not in test_cache:
                record=case['record']
                with np.load(record['case']['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
                with np.load(record['dataset']['path'],allow_pickle=False) as f:x=f['features'][ids].astype(np.float32);y=f['target'][ids].copy()
                z=load_protected(m['artifact']).raw(x);control=m['summaries'][row['selector']]['control'];pred,loss,brier=sufficient(y,z,case['task'],control['temperature'],control['shifts'])
                quality=metrics(y,pred,loss,brier,case['task'],z.shape[1]);test_cache[key]=dict(case=cid,uid=uid,selector=row['selector'],quality=quality,cost_bytes=m['cost_bytes'])
            evaluated.append(test_cache[key])
        test_rows.append(dict(**row,quality=evaluated))
    result=dict(status='saved_banks_complete',cases=prepared,audits=audits,allocations=allocations,nested=nested,test_rows=test_rows,
        environment=environment,seal=dict(path=str(out/'pretest_seal.json'),sha256=digest(out/'pretest_seal.json')),
        elapsed_seconds=perf_counter()-started,new_trees=0,unique_test_evaluations=len(test_cache),
        limitations=['old tests exploratory; overlapping banks and trainD repeats not independent tasks','conservative score mean minus one empirical SD, not a confidence guarantee',
        'mean selector is equal-repeat control; standard uses first repeat','catalogue caps simulate query limits; all calibration computations are still paid',
        'no new inference timing or operational GPU capacity test in stages I/II'])
    write(out/'report.json',result);event('saved_banks_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'),banks=len(banks),allocations=len(allocations),nested=len(nested),new_trees=0)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
