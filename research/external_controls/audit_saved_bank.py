"""Independent numerical replay of stage I, without recalibration or fitting trees."""
from pathlib import Path
import numpy as np
from scipy.special import softmax,expit
from sklearn.metrics import f1_score
from budget_quality_worker import locked,digest,write,event
from abl_transfer_study import data_before_test


def independent(y,z,task,policy):
    zz=z/policy['temperature'];shifts=np.asarray(policy['shifts'])
    if task=='multiclass':
        p=np.clip(softmax(zz,axis=1),1e-7,1-1e-7);pred=(zz-shifts).argmax(1)
        loss=-np.log(p[np.arange(len(y)),y.astype(int)]);brier=(p*p).sum(1)-2*p[np.arange(len(y)),y.astype(int)]+1
    else:
        assert policy['temperature']==1.;p=np.clip(expit(z),1e-7,1-1e-7);pred=z>=shifts
        loss=(-y*np.log(p)-(1-y)*np.log1p(-p)).mean(1);brier=((p-y)**2).mean(1)
    return pred,loss,brier


def quality(y,pred,loss,brier,task,c):
    args={'labels':np.arange(c)} if task=='multiclass' else {}
    return dict(loss=float(np.mean(loss)),brier=float(np.mean(brier)),f1_macro=float(f1_score(y,pred,average='macro',zero_division=0,**args)),
        f1_weighted=float(f1_score(y,pred,average='weighted',zero_division=0,**args)))


def audit(report,out):
    cases={};checks=0;maximum=0.;seen=set();outer_checked=0
    for ref in report['cases']:
        case=locked(ref['path'],ref['sha256']);cid=case['case'];data=data_before_test(case['record']);task=case['task'];cases[cid]=case
        for m in case['models']:
            assert digest(m['raw']['path'])==m['raw']['sha256']
            with np.load(m['raw']['path'],allow_pickle=False) as f:raw=f['raw'].copy()
            for outer_index,assessment in [(-1,m['assessment'])]+[(k,o['assessment']) for k,o in enumerate(m['outer'])]:
                if assessment['path'] in seen:continue
                seen.add(assessment['path']);record=locked(assessment['path'],assessment['sha256'])
                global_ids=np.arange(len(data['dev_y'])) if outer_index<0 else np.asarray(m['outer'][outer_index]['fitting'])
                z=raw[global_ids];y=data['dev_y'][global_ids]
                if outer_index>=0:assert not set(global_ids)&set(m['outer'][outer_index]['validation'])
                for replay in record['replays']:
                    ids=[];collected={}
                    for state in replay['states']:
                        a,b=np.asarray(state['fitting']),np.asarray(state['validation'])
                        assert len(set(a)&set(b))==0 and set(a)|set(b)==set(range(len(y)))
                        ids.extend(b)
                        for name,control in state['policies'].items():collected.setdefault(name,[]).append(independent(y[b],z[b],task,control))
                    assert sorted(ids)==list(range(len(y)))
                    for name,parts in collected.items():
                        pred,loss,brier=[np.concatenate([p[k] for p in parts]) for k in range(3)]
                        actual=quality(y[ids],pred,loss,brier,task,z.shape[1]);error=max(abs(actual[k]-replay['scores'][name][k]) for k in actual)
                        assert error<1e-11;maximum=max(maximum,error);checks+=1
                if outer_index>=0:
                    o=m['outer'][outer_index];assert digest(o['predictions']['path'])==o['predictions']['sha256'];b=np.asarray(o['validation'])
                    with np.load(o['predictions']['path'],allow_pickle=False) as f:
                        for selector in ('standard','mean','conservative'):
                            pred,loss,brier=independent(data['dev_y'][b],raw[b],task,o['summaries'][selector]['control'])
                            np.testing.assert_array_equal(pred,f[selector+'__prediction']);np.testing.assert_allclose(loss,f[selector+'__loss'],atol=1e-12)
                            np.testing.assert_allclose(brier,f[selector+'__brier'],atol=1e-12);outer_checked+=1
        event('saved_bank_audit_case',case=cid,checks=checks,outer_predictions=outer_checked)
    allocations_checked=0
    for row in report['allocations']:
        a=row['allocation']
        if a['status']!='feasible':continue
        selected=[next(m for m in cases[cid]['models'] if m['uid']==uid) for cid,uid in zip(row['cases'],a['choices'])]
        assert len(selected)==len(row['cases']) and sum(m['cost_bytes'] for m in selected)==a['cost_bytes']<=a['budget']
        if row['strategy'].startswith('ordinary'):assert all(m['family']=='ordinary' for m in selected)
        if row['strategy']=='compact':assert all(m['family']!='ordinary' for m in selected)
        allocations_checked+=1
    result=dict(status='passed',calibration_score_checks=checks,outer_prediction_checks=outer_checked,allocation_checks=allocations_checked,
        maximum_metric_error=maximum,cases=len(cases),scope='independent scipy/sklearn replay of every unique assessment and outer prediction; fold disjointness and exact bytes')
    write(out,result);event('saved_bank_audit_complete',**result);return result
