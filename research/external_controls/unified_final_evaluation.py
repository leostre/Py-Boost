"""Seal D choices first, then replay old test exactly once; no training or repair."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.special import softmax,expit,logsumexp
from sklearn.metrics import f1_score,log_loss
from budget_quality_worker import locked,digest,write,event
from abl_protected_core import load_protected
from unified_training import quality_row
from unified_frontier_core import sufficient,metrics
from unified_meta_core import training_profile
from abl_transfer_study import data_before_test


def pick(models,task):
    def key(m):
        q=quality_row(m['evaluations'],task)['quality']
        return (q['loss'],-q['f1_macro'],m['cost_bytes']) if task=='multiclass' else (-q['f1_macro'],q['loss'],m['cost_bytes'])
    return min(models,key=key)


def choose_native(report,cohort):
    choices=[]
    for ref in report['cases']:
        case=locked(ref['path'],ref['sha256']);task=case['task'];models=case['models']
        if report['status']=='unified_history_complete':
            selected=[dict(m,selection='fixed_final_step') for m in models if m['block']==3]
        else:
            selected=[]
            for family in ('ordinary','posthoc','online'):
                for ratio in (.4,.7,1.):
                    pool=[m for m in models if m['family']==family and m['cost_bytes']<=ratio*case['reference_bytes']]
                    if pool:selected.append(dict(pick(pool,task),selection=f'memory_slice{ratio:g}'))
            selected.append(dict(next(m for m in models if m['family']=='ordinary' and m['block']==3),selection='fixed_full200'))
        for m in selected:
            e=quality_row(m['evaluations'],task)
            choices.append(dict(cohort=cohort,case=case['case'],uid=m['uid'],family=m['family'],selection=m['selection'],
                artifact=m['artifact'],sha256=m['sha256'],control=e['refit_control'],development=e['quality'],record=case['record'],
                cost_bytes=m['cost_bytes'],parameters=m['parameters'],trees=m['trees'],blocks=m['blocks'],nnz=m['nnz'],rank=m.get('rank'),leaf_cap=m.get('leaf_cap')))
    return choices


def choose_allocations(report,cohort):
    """Replay U4 fixed UID assignments; never optimize from later test values."""
    seal=locked(report['seal']['path'],report['seal']['sha256'])
    cases={r['case']:locked(r['path'],r['sha256']) for r in report['case_representations']}
    order=report['cases'];choices={};allocations=[]
    def add(cid,uid):
        key=cohort+'/'+cid+'/'+uid
        if key not in choices:
            case=cases[cid];m=next(m for m in case['models'] if m['uid']==uid)
            e=quality_row(m['evaluations'],case['task']);option=next(o for o in seal['options'][cid] if o['uid']==uid)
            choices[key]=dict(cohort=cohort,case=cid,uid=uid,family=m['family'],selection='presealed_bank_allocation',
                allocation_key=key,artifact=m['artifact'],sha256=m['sha256'],control=e['refit_control'],
                development=e['quality'],record=case['record'],cost_bytes=option['cost_bytes'],
                cost_scope='U4 sealed coalesced runtime envelope; excludes context/workspace')
        return key
    references={m['case']:add(m['case'],m['uid']) for m in seal['inventory'] if m['bank']=='full'}
    assert set(references)==set(order)
    for mode,rows in seal['bank_frontiers'].items():
        for row in rows:
            a=dict(row,cohort=cohort,mode=mode,references=references)
            if row['status']=='feasible':
                assert len(row['choices'])==len(order)
                a['member_keys']=[add(cid,uid) for cid,uid in zip(order,row['choices'])]
                assert sum(choices[key]['cost_bytes'] for key in a['member_keys'])==row['cost_bytes']<=row['budget']
            allocations.append(a)
    return list(choices.values()),allocations


def data_test(record):
    c,d=record['case'],record['dataset'];assert digest(c['split_file'])==c['split_sha256'] and digest(d['path'])==d['sha256']
    with np.load(c['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
    with np.load(d['path'],allow_pickle=False) as f:return f['features'][ids].copy(),f['target'][ids].copy(),ids


def audit(raw,y,task,control,reported):
    # Separate library-level probability and classification calculation.
    z=raw/control['temperature'];p=np.clip(softmax(z,axis=1) if task=='multiclass' else expit(z),1e-7,1-1e-7)
    pred=np.argmax(z-np.asarray(control['shifts']),axis=1) if task=='multiclass' else np.greater_equal(z,control['shifts'])
    c=raw.shape[1];labels={'labels':np.arange(c)} if task=='multiclass' else {}
    loss=float(np.mean(-np.clip(z[np.arange(len(y)),y.astype(int)]-logsumexp(z,axis=1),np.log(1e-7),np.log1p(-1e-7)))) if task=='multiclass' else float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
    brier=float(np.mean(np.sum((p-np.eye(c)[y.astype(int)])**2,axis=1))) if task=='multiclass' else float(np.mean((p-y)**2))
    independent=dict(loss=float(loss),brier=brier,f1_macro=float(f1_score(y,pred,average='macro',zero_division=0,**labels)),
        f1_weighted=float(f1_score(y,pred,average='weighted',zero_division=0,**labels)))
    for k in independent:assert abs(independent[k]-reported[k])<2e-6,(k,independent[k],reported[k])
    return dict(status='passed',independent=independent)


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);spec=cfg['unified_final'];choices=[];allocations=[]
    assert spec.get('inputs_sealed',True),'final inputs are not sealed; preparation node must not execute'
    for cohort,ref in spec['native_sources'].items():
        report=locked(ref['path'],ref['sha256']);assert all(c['audit']['status']=='passed' for c in report['cases'])
        choices.extend(choose_native(report,cohort))
    for cohort,ref in spec['serving_sources'].items():
        report=locked(ref['path'],ref['sha256']);seal=locked(report['seal']['path'],report['seal']['sha256'])
        records={r['case']:r['record'] for r in seal['source_cases']}
        for m in seal['inventory']:
            choices.append(dict(cohort=cohort,case=m['case'],uid=m['uid'],family=m['bank'],selection='serving_sealed',
                artifact=m['artifact'],sha256=m['sha256'],control=m['control'],development=m['quality'],record=records[m['case']],cost_bytes=m['cost_bytes']))
    for cohort,ref in spec.get('allocation_sources',{}).items():
        report=locked(ref['path'],ref['sha256']);fixed,plan=choose_allocations(report,cohort);choices.extend(fixed);allocations.extend(plan)
    ext=locked(spec['external']['path'],spec['external']['sha256']);assert ext['status']=='unified_external_serving_complete'
    choices.extend(dict(cohort='matched_external',uid=r['family'],selection='serving_sealed',control=r['control'],record=r['record'],case=r['case'],family=r['family'],
        raw_cache=dict(path=r['path'],sha256=r['sha256'])) for r in ext['test_predictions'])
    reused=0
    if spec.get('reused_final'):
        previous=locked(spec['reused_final']['path'],spec['reused_final']['sha256'])
        previous_seal=locked(previous['seal']['path'],previous['seal']['sha256'])
        def cache_key(c):return (c['case'],c.get('sha256'),json.dumps(c['control'],sort_keys=True),c['record']['case']['split_sha256'])
        cache={cache_key(c):r['raw'] for c,r in zip(previous_seal['choices'],previous['rows']) if 'artifact' in c}
        assert len(previous_seal['choices'])==len(previous['rows'])
        for c in choices:
            if cache_key(c) in cache:c['raw_cache']=cache[cache_key(c)];reused+=1
    assert len(choices)<=spec['maximum_models']
    # This file is written BEFORE any call to data_test / access to test labels.
    profiles={}
    for c in choices:
        key=c['case']+'/'+c['record']['case']['split_sha256']
        if key not in profiles:
            record=c['record'];profiles[key]=training_profile(data_before_test(record),record['case']['task'],record['case']['outputs'])
    write(out/'training_profiles.json',profiles)
    write(out/'sealed_allocations.json',allocations)
    seal=out/'pretest_seal.json';write(seal,dict(choices=choices,source=spec,rule='no repairs or choices after test',test_labels_accessed=False,profiles_sha256=digest(out/'training_profiles.json')))
    rows=[];identities={};cache={};targets={}
    for index,choice in enumerate(choices):
        cid=choice['case'];record=choice['record'];task=record['case']['task']
        key=(cid,record['case']['split_sha256'])
        if key not in cache:cache[key]=data_test(record)
        x,y,ids=cache[key];identity=(ids.tobytes(),y.tobytes())
        if cid in identities:assert identities[cid]==identity,'old test changed across data protocols'
        identities[cid]=identity
        if cid not in targets:
            path=out/'test_targets'/f'{cid}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,target=y,indices=ids)
            targets[cid]=dict(path=str(path),sha256=digest(path),task=task,outputs=record['case']['outputs'])
        if 'raw_cache' in choice:
            ref=choice['raw_cache'];assert digest(ref['path'])==ref['sha256']
            with np.load(ref['path'],allow_pickle=False) as f:raw=f['raw'].copy()
        else:
            assert digest(choice['artifact'])==choice['sha256'];raw=load_protected(choice['artifact']).raw(x)
        control=choice['control'];pred,loss,brier=sufficient(y,raw,task,control['temperature'],control['shifts'])
        q=metrics(y,pred,loss,brier,task,raw.shape[1]);check=audit(raw,y,task,control,q)
        file=out/f'test_raw_{index:04d}.npz';np.savez_compressed(file,raw=raw,pred=pred,loss=loss,brier=brier)
        row={k:v for k,v in choice.items() if k not in ('record','raw_cache')};row.update(test=q,audit=check,
            test_rows=len(y),outputs=raw.shape[1],task=task,test_ids_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest(),
            raw=dict(path=str(file),sha256=digest(file)))
        rows.append(row);write(out/'results.json',rows)
        event('unified_test_model',index=index,case=cid,cohort=choice['cohort'],family=choice['family'],quality=q)
    result=dict(status='unified_final_complete',seal=dict(path=str(seal),sha256=digest(seal)),rows=rows,targets=targets,
        training_profiles=dict(path=str(out/'training_profiles.json'),sha256=digest(out/'training_profiles.json')),
        audits_passed=all(r['audit']['status']=='passed' for r in rows),allocations=allocations,reused_raw_predictions=reused,
        case_tests_identical_across_protocols=True,new_tree_fits=0,test_evaluated=True,
        interpretation='Exploratory evaluation on previously viewed old tests; no claim of independent generalization and no reselection from these metrics')
    write(out/'report.json',result);event('unified_final_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'),models=len(rows))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
