"""U1: paired, ungated calibration/frontiers over immutable BR0 models."""
from pathlib import Path
from dataclasses import asdict
import gc,json,sys
from time import perf_counter
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment,memory_guard
from abl_budget_revision import source_case,role_data,pin
from abl_protected_core import load_protected,block_cost
from unified_frontier_core import (Fold,make_folds,replay_calibration,variants,
    choose_policy,sufficient,metrics,frontier,quality_change)


def score_policy(raw,y,task,policy):
    return metrics(y,*sufficient(y,raw,task,policy['temperature'],policy['shifts']),task,raw.shape[1])


def store_replay(replay,path):
    arrays={'validation':replay['validation']}
    for name,parts in replay['arrays'].items():
        for key,value in zip(('prediction','loss','brier'),parts):arrays[name+'__'+key]=value
    np.savez_compressed(path,**arrays)
    return dict(path=str(path),sha256=digest(path),scores=replay['scores'],states=replay['states'])


def evaluate_model(raw,y,task,sizes,seed,grid,shrinkages,out,repeats=3):
    a,b=np.cumsum(sizes)[:2];n=len(y);records=[]
    # Original roles, but ungated policy choice. Historical BR0 choices remain separate.
    fitted=variants(raw[:a],y[:a],raw[a:b],y[a:b],task,grid,shrinkages)
    control={name:score_policy(raw[a:b],y[a:b],task,p) for name,p in fitted.items()}
    legacy=replay_calibration(raw,y,task,(Fold(tuple(range(a)),tuple(range(b,n))),),grid,shrinkages,False)
    saved=store_replay(legacy,out/'legacy.npz')
    for objective in ('f1_macro','loss'):
        name=choose_policy(control,objective)
        records.append(dict(protocol='legacy',repeat=0,objective=objective,policy=name,
            quality=legacy['scores'][name],control=control,refit_control=fitted[name],evaluation=saved,
            evaluation_scope='old selection only; policy selected on old control'))
    for repeat in range(repeats):
        for protocol in ('pooled_holdout','crossfit'):
            if protocol=='pooled_holdout':
                inner=replay_calibration(raw[:b],y[:b],task,make_folds(y[:b],task,seed+1009*repeat),grid,shrinkages)
                fitted=variants(raw[:b],y[:b],raw[b:],y[b:],task,grid,shrinkages)
                replay=replay_calibration(raw,y,task,(Fold(tuple(range(b)),tuple(range(b,n))),),grid,shrinkages,False)
                selection=inner['scores'];inner_ref=store_replay(inner,out/f'{protocol}-{repeat}-inner.npz')
                scope='same old selection rows as legacy; internal OOF policy choice in cal+control'
            else:
                replay=replay_calibration(raw,y,task,make_folds(y,task,seed+1009*repeat),grid,shrinkages)
                selection=replay['scores'];inner_ref=None
                fitted=variants(raw,y,raw[:1],y[:1],task,grid,shrinkages)
                scope='all development OOF; not the same population as old selection'
            saved=store_replay(replay,out/f'{protocol}-{repeat}.npz')
            for objective in ('f1_macro','loss'):
                name=choose_policy(selection,objective)
                records.append(dict(protocol=protocol,repeat=repeat,objective=objective,policy=name,
                    quality=replay['scores'][name],control=selection,refit_control=fitted[name],
                    evaluation=saved,inner_evaluation=inner_ref,evaluation_scope=scope))
    return records


def audit_case(case,data):
    """Independent replay of saved arrays and fold controls, including all 156 models."""
    from scipy.special import softmax,expit
    from sklearn.metrics import f1_score
    y=data['dev_y'];x=data['dev_x'];checks=0;maximum=0.
    for model in case['models']:
        if digest(model['artifact'])!=model['sha256']:raise RuntimeError('model changed')
        zref=model['raw'];assert digest(zref['path'])==zref['sha256']
        with np.load(zref['path'],allow_pickle=False) as cache:
            z=cache['raw'].copy() if 'raw' in cache else np.concatenate([cache[k] for k in ('calibration','control','selection')])
        ids=np.linspace(0,len(y)-1,min(17,len(y))).astype(int)
        difference=float(np.max(np.abs(load_protected(model['artifact']).raw(x[ids])-z[ids])))
        maximum=max(maximum,difference);assert difference<2e-5
        assert block_cost(load_protected(model['artifact']))['cost_bytes']==model['cost_bytes']
        references={r['evaluation']['path']:r['evaluation'] for r in model['evaluations']}
        references.update({r['inner_evaluation']['path']:r['inner_evaluation'] for r in model['evaluations'] if r.get('inner_evaluation')})
        for path,ref in references.items():
            assert digest(path)==ref['sha256']
            with np.load(path,allow_pickle=False) as replay:
                validation=replay['validation'];np.testing.assert_array_equal(validation,np.concatenate([s['validation'] for s in ref['states']]))
                for name,q in ref['scores'].items():
                    predictions=[];loss=[];brier=[]
                    for s in ref['states']:
                        assert not set(s['fitting'])&set(s['validation'])
                        idx=np.asarray(s['validation']);p=s['policies'][name];r=z[idx];yy=y[idx]
                        if case['task']=='multiclass':
                            pp=np.clip(softmax(r/p['temperature'],axis=1),1e-7,1-1e-7)
                            predictions.append((r/p['temperature']-p['shifts']).argmax(1))
                            loss.extend(-np.log(pp[np.arange(len(idx)),yy]))
                            brier.extend(np.sum(pp*pp,axis=1)-2*pp[np.arange(len(idx)),yy]+1)
                        else:
                            pp=np.clip(expit(r),1e-7,1-1e-7);assert p['temperature']==1.
                            predictions.append(r-p['shifts']>=0)
                            loss.extend(np.mean(-yy*np.log(pp)-(1-yy)*np.log1p(-pp),axis=1));brier.extend(np.mean((pp-yy)**2,axis=1))
                    pred=np.concatenate(predictions);np.testing.assert_array_equal(pred,replay[name+'__prediction'])
                    np.testing.assert_allclose(loss,replay[name+'__loss'],atol=1e-12)
                    np.testing.assert_allclose(brier,replay[name+'__brier'],atol=1e-12)
                    args={'labels':np.arange(z.shape[1])} if case['task']=='multiclass' else {}
                    for key,value in [('loss',np.mean(loss)),('brier',np.mean(brier)),
                        ('f1_macro',f1_score(y[validation],pred,average='macro',zero_division=0,**args)),
                        ('f1_weighted',f1_score(y[validation],pred,average='weighted',zero_division=0,**args))]:
                        assert abs(q[key]-value)<1e-10
                    checks+=1
        for r in model['evaluations']:
            expected=choose_policy(r['control'],r['objective']);assert r['policy']==expected
    return dict(status='passed',models=len(case['models']),metric_policy_checks=checks,maximum_raw_replay_error=maximum,
        scope='independent metrics and predictions, all fold controls/disjointness, objective selection and real bytes; refitting solver covered by local tests')


def run(cfg,out):
    started=perf_counter();environment=check_native_environment(cfg);spec=cfg['unified']
    source=locked(spec['br0']['path'],spec['br0']['sha256']);cases=[];locks={}
    for ref in source['cases']:
        old=locked(ref['path'],ref['sha256']);cid=old['case'];directory=out/cid;directory.mkdir(parents=True,exist_ok=True)
        original=source_case(old['registry'],locks);data,features,labels=role_data(original);task=old['task']
        entries=[dict(uid='br0:'+r['uid'],family=r['family'],artifact=r['artifact'],sha256=r['sha256'],raw=r['cache'],historical=r) for r in old['rows']]
        for uid,m in original['metas'].items():
            if uid.startswith('full/'):
                entries.append(dict(uid='ordinary:'+uid,family='ordinary',artifact=m['artifact'],sha256=m['sha256']))
        entries.append(dict(uid='reference',family='reference',artifact=old['anchor']['artifact'],sha256=old['anchor']['sha256']))
        if task=='multilabel':entries.append(dict(uid='input160',family='input',artifact=old['input_snapshot']['artifact'],sha256=old['input_snapshot']['sha256']))
        models=[]
        for i,item in enumerate(entries):
            p=directory/f'model{i:03d}';p.mkdir(exist_ok=True);assert digest(item['artifact'])==item['sha256']
            model=load_protected(item['artifact']);price=block_cost(model)
            if 'raw' in item:
                assert digest(item['raw']['path'])==item['raw']['sha256']
                with np.load(item['raw']['path'],allow_pickle=False) as f:z=np.concatenate([f[k] for k in ('calibration','control','selection')])
            else:
                z=model.raw(data['dev_x']);rawpath=p/'raw.npz';np.savez_compressed(rawpath,raw=z)
                item['raw']=dict(path=str(rawpath),sha256=digest(rawpath))
            evaluations=evaluate_model(z,data['dev_y'],task,data['sizes'],int(cid.split('_')[-1])+20261028,
                cfg['decision_grid'],cfg['decision_shrinkage'],p,spec['calibration_repeats'])
            result=dict(**item,**price,evaluations=evaluations);write(p/'model.json',result);models.append(result)
            event('unified_recalibrated',case=cid,uid=item['uid'],models_done=i+1,cost_bytes=price['cost_bytes'])
            del model,z;gc.collect();memory_guard()
        anchor=next(m for m in models if m['uid']=='reference');fronts=[]
        for protocol,repeat in sorted({(r['protocol'],r['repeat']) for r in anchor['evaluations']}):
            for objective in ('f1_macro','loss'):
                points=[];reference=next(r for r in anchor['evaluations'] if (r['protocol'],r['repeat'],r['objective'])==(protocol,repeat,objective))
                for m in models:
                    r=next(r for r in m['evaluations'] if (r['protocol'],r['repeat'],r['objective'])==(protocol,repeat,objective))
                    points.append(dict(uid=m['uid'],family=m['family'],cost_bytes=m['cost_bytes'],quality=r['quality'],
                        relative_macro_loss=quality_change(r['quality']['f1_macro'],reference['quality']['f1_macro'])))
                fronts.append(dict(protocol=protocol,repeat=repeat,objective=objective,points=points,frontier=frontier(points,objective)))
        case=dict(case=cid,task=task,record=original['record'],sizes=data['sizes'],models=models,fronts=fronts,new_tree_fits=0,test_evaluated=False)
        audit=audit_case(case,data);case['audit']=audit;path=directory/'case.json';write(path,case)
        cases.append(dict(case=cid,path=str(path),sha256=digest(path),audit=audit,models=len(models)))
        write(out/'progress.json',dict(cases=cases));event('unified_case_complete',case=cid,audit=audit)
    result=dict(status='unified_recalibration_complete',cases=cases,config=spec,environment=environment,
        new_tree_fits=0,test_evaluated=False,gpu_execution=False,elapsed_seconds=perf_counter()-started,cpu_peak_rss_bytes=memory_guard(),
        limitations=['OOF model selection is exploratory, not independent validation','legacy and pooled_holdout share selection rows; full crossfit uses all D',
            'multilabel retains original T=1; threshold comparison only','historical 5% rule outputs retained separately; no new acceptance gate'])
    write(out/'report.json',result);event('unified_recalibration_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'),cases=len(cases),models=sum(c['models'] for c in cases))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
