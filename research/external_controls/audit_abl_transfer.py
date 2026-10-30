"""Independent multiclass/multilabel quality, frozen ranking and array checks."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score
from budget_quality_worker import locked,digest,native_load
from audit_calibration_pilot import independent_quality
from audit_abl_quality import control
from audit_abl_width import width_choice


def independent_task_quality(y,z,cp,task):
    if task=='multiclass':return independent_quality(y,z,cp)
    assert cp['temperature']==1.
    p=np.clip(expit(z.astype(np.float64)),1e-7,1-1e-7);pred=z-np.asarray(cp['shifts'])>=0
    return dict(loss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
        f1_macro=float(f1_score(y,pred,average='macro',zero_division=0)),
        f1_weighted=float(f1_score(y,pred,average='weighted',zero_division=0)))


def independent_bytes(model):
    p=model.prepare_inference()
    return sum(getattr(p.base,k).nbytes for k in ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias'))+sum(
        getattr(p,k).nbytes for k in ('indptr','columns','values','decision_shifts'))+4


def audit_development(entries,ledger):
    assert ledger['reserved_rounds']==4800 and len(ledger['attempts'])==6
    assert all(a['state']=='complete' and a['fit']['rounds']==800 and a['fit']['actual_minimum_leaf']==50 for a in ledger['attempts'])
    cases=[]
    for e in entries:
        case=locked(e['path'],e['sha256']);record=case['record'];split=record['case'];source=record['dataset']
        assert digest(split['split_file'])==split['split_sha256'] and digest(source['path'])==source['sha256']
        with np.load(split['split_file'],allow_pickle=False) as f:
            ids=f['selection'].copy();allroles=[f[k].copy() for k in ('train','calibration','control','selection','test')]
            fit=np.r_[allroles[1],allroles[2]]
        joined=np.concatenate(allroles);assert np.array_equal(np.sort(joined),np.arange(len(joined)))
        with np.load(source['path'],allow_pickle=False) as f:y=f['target'][ids];fit_y=f['target'][fit]
        for row in case['rows']:
            meta=case['metas'][row['uid']];cache=meta['development_cache']
            assert digest(cache['path'])==cache['sha256'] and digest(meta['source']['artifact'])==meta['source']['sha256']
            with np.load(cache['path'],allow_pickle=False) as f:z=f['raw'][-len(ids):].copy()
            for name,score in row['validation'].items():
                measured=independent_task_quality(y,z,control(row,name),case['task'])
                for k,v in measured.items():assert abs(v-score[k])<2e-10
            model=native_load(meta['source']['artifact']);assert independent_bytes(model)==meta['cost_bytes']
            if meta['family']=='compact':assert len(model.values)<=int(meta['residual_fraction']*meta['parent_parameters'])
        for c in (*case['choices'],*case['full_reference_choices']):
            assert width_choice(case['rows'],case['metas'],c)==c['chosen']
        cases.append(dict(case=case['case_id'],status='passed',candidates=len(case['metas']),choices=len(case['choices'])))
    assert len(cases)==15
    return dict(status='passed',cases=cases,new_tree_fits=0,gpu_execution=False,test_accessed=False)


def audit_final(report):
    results=[]
    for entry in report['evaluations']:
        result=locked(entry['path'],entry['sha256']);source=result['development'];case=locked(source['path'],source['sha256'])
        target=result['target'];assert digest(target['path'])==target['sha256']
        with np.load(target['path'],allow_pickle=False) as f:y=f['target'];indices=f['indices']
        with np.load(case['record']['case']['split_file'],allow_pickle=False) as f:np.testing.assert_array_equal(indices,f['test'])
        with np.load(case['record']['dataset']['path'],allow_pickle=False) as f:np.testing.assert_array_equal(y,f['target'][indices])
        raw={}
        for uid,item in result['predictions'].items():
            assert digest(item['path'])==item['sha256']
            with np.load(item['path'],allow_pickle=False) as f:raw[uid]=f['raw'].copy()
        for p,c in zip((*result['points'],*result['full_reference_points']),(*case['choices'],*case['full_reference_choices'])):
            for name in ('mode','objective','group','budget_bytes'):assert p[name]==c[name]
            if c['chosen'] is None:assert p['status']=='no_feasible_candidate';continue
            assert p['uid']==c['chosen']['uid'] and p['policy']==c['chosen']['policy']
            if p['status']!='evaluated':continue
            q=independent_task_quality(y,raw[p['uid']],p['control'],case['task'])
            for k,v in q.items():assert abs(v-p['test'][k])<2e-10
            assert p['cost_bytes']==case['metas'][p['uid']]['cost_bytes']
            assert p['cost_bytes']<=p['budget_bytes']
        results.append(dict(case=result['case'],status='passed',points=len(result['points'])))
    assert len(results)==15
    return dict(status='passed',cases=results,new_tree_fits=0,gpu_execution=False)
