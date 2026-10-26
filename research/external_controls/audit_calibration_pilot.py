"""Independent H-CAL1 replay; run inside the recorded child source snapshot.

No output calibration or tree fitting occurs here. The auditor verifies saved
rankings, test metrics, row boundaries and exact resident-array accounting.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from budget_quality_worker import digest, write, locked, event, native_load


def policy_meta(name):
    temperature=name.startswith('temperature')
    fraction=float(name.split('_shift_')[1]) if '_shift_' in name else 0.
    return temperature,fraction


def policy_rank(name,score,objective):
    temp,fraction=policy_meta(name)
    tail=(int(temp)+int(fraction>0),fraction,name)
    return ((score['loss'],*tail) if objective=='loss' else
            (-score['f1_macro'],score['loss'],*tail))


def independent_quality(target,raw,control):
    z=raw.astype(np.float64)/control['temperature']
    exp=np.exp(z-z.max(axis=1,keepdims=True));p=exp/exp.sum(axis=1,keepdims=True)
    p=np.clip(p,1e-6,1-1e-6);p=p/p.sum(axis=1,keepdims=True)
    c=z.shape[1];pred=(z-control['shifts']).argmax(axis=1)
    truth=np.eye(c)[target]
    return dict(loss=float(-np.log(np.clip(p[np.arange(len(target)),target],1e-7,1-1e-7)).mean()),
        brier=float(np.square(p-truth).sum(axis=1).mean()),
        f1_macro=float(f1_score(target,pred,labels=np.arange(c),average='macro',zero_division=0)),
        f1_weighted=float(f1_score(target,pred,labels=np.arange(c),average='weighted',zero_division=0)))


def check_plans(plans,y,train,test,development_indices):
    n=len(y);assert not set(development_indices)&(set(train)|set(test))
    for plan in plans:
        validations=[];used=set(plan['refit_temperature'])|set(plan['refit_shifts'])
        for fold in plan['folds']:
            t,s,v=(fold[k] for k in ('temperature','shifts','validation'))
            for group in (t,s,v):
                assert len(group)==len(set(group)) and min(group)>=0 and max(group)<n
            assert not set(v)&(set(t)|set(s))
            if plan['scheme']=='separate':assert not set(t)&set(s)
            else:assert t==s
            validations.extend(v);used.update(t);used.update(s);used.update(v)
        assert len(validations)==len(set(validations))
        if plan['scheme']=='crossfit':assert sorted(validations)==list(range(n))
        assert plan['unique_labels_used']==len(used)
    for repeat in range(10):
        for scheme in ('separate','pooled','crossfit'):
            group=sorted((p for p in plans if p['repeat']==repeat and p['scheme']==scheme),
                         key=lambda p:p['fraction'])
            assert [p['fraction'] for p in group]==[.25,.5,1.]
            for a,b in zip(group,group[1:]):
                for fa,fb in zip(a['folds'],b['folds']):
                    assert set(fa['temperature'])<=set(fb['temperature'])
                    assert set(fa['shifts'])<=set(fb['shifts'])
                    assert fa['validation']==fb['validation']


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text());src=cfg['calibration_pilot_completed']
    report=locked(src['path'],src['sha256']);root=Path(src['path']).parent
    seal=locked(root/'pretest_choices.json',report['pretest_seal_sha256'])
    assert seal['status']=='calibration_choices_sealed_before_test'
    assert report['new_tree_fits']==report['new_boosting_rounds']==0
    assert report['test_used_for_selection'] is False and report['gpu_execution'] is False
    for path,sha in seal['input_locks'].items():assert digest(path)==sha,path
    source=cfg['calibration_pilot']['source_development']
    original=locked(source['path'],source['sha256'])
    metrics={};targets={};test_predictions={};exported=[]
    for pred in report['prediction_audit']:
        assert digest(pred['path'])==pred['sha256']
        case=Path(pred['path']).parent.name
        with np.load(pred['path'],allow_pickle=False) as f:
            test_predictions[(case,pred['family'],pred['model'])]=f['raw'].copy()
            if case in targets:np.testing.assert_array_equal(targets[case],f['target'])
            targets[case]=f['target'].copy()
    def q(case,family,model,control):
        key=(case,family,model,control['temperature'],tuple(control['shifts']))
        if key not in metrics:
            metrics[key]=independent_quality(targets[case],test_predictions[(case,family,model)],control)
        return metrics[key]
    checked_policies=checked_budgets=checked_arrays=0
    cases={};index={}
    for item in seal['cases']:
        case=locked(item['path'],item['sha256']);name=case['case'];cases[name]=case
        planfile=locked(case['plan_path'],case['plan_sha256'])
        native=next(r for r in original['records'] if r['case_id']==name and r['family']=='pyboost')
        with np.load(native['case']['split_file'],allow_pickle=False) as f:
            train,test=f['train'].copy(),f['test'].copy()
        with np.load(native['dataset']['path'],allow_pickle=False) as f:
            y=f['target'][planfile['development_indices']].copy()
        check_plans(planfile['plans'],y,train,test,planfile['development_indices'])
        inputs=dict(target=y,indices=np.array(planfile['development_indices']))
        for model in case['models']:
            with np.load(model['development_file'],allow_pickle=False) as f:
                inputs[model['family']+'__'+model['identity']]=np.concatenate([f[r] for r in ('calibration','control','selection')])
            if model['family']=='pyboost':
                saved=native_load(model['artifact']).prepare_inference()
                nbytes=sum(getattr(saved.base,k).nbytes for k in
                           ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias'))
                nbytes+=sum(getattr(saved,k).nbytes for k in ('indptr','columns','values','decision_shifts'))
                assert nbytes==saved.resident_bytes
                rows=[r for r in case['rows'] if r['family']=='pyboost' and r['model_id']==model['identity']]
                assert all(r['cost_bytes']==nbytes+4 for r in rows)
                checked_arrays+=1
        inp=args.output/'inputs'/(name+'.npz');inp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(inp,**inputs)
        exported.append(dict(case=name,path=str(inp),sha256=digest(inp),models=len(case['models'])))
        assert Counter((r['family'],r['model_id']) for r in case['rows'])==Counter(
            {(m['family'],m['identity']):90 for m in case['models']})
        for r in case['rows']:
            index[(name,r['family'],r['model_id'],r['scheme'],r['repeat'],r['fraction'])]=r
            for objective,choice in r['choices'].items():
                scores=r['validation'];names=['raw','temperature'] if objective=='loss' else list(scores)
                assert choice['policy']==min(names,key=lambda name:policy_rank(name,scores[name],objective))
                assert choice['validation']==scores[choice['policy']]
                c=choice['control'];assert .1-1e-7<=c['temperature']<=10
                assert float(np.float32(c['temperature']))==c['temperature']
                assert np.array_equal(np.array(c['shifts']),np.array(c['shifts'],np.float32))
                checked_policies+=1
            assert r['validation']['raw']['f1_macro']==r['validation']['temperature']['f1_macro']
            for policy,score in r['validation'].items():
                if '_shift_' in policy:
                    base='temperature' if policy.startswith('temperature') else 'raw'
                    assert score['loss']==r['validation'][base]['loss']
                    assert score['brier']==r['validation'][base]['brier']
        for choice in case['budget_choices']:
            pool=[r for r in case['rows'] if r['family']=='pyboost' and r['scheme']==choice['scheme'] and
                  r['repeat']==choice['repeat'] and r['fraction']==choice['fraction'] and
                  r['cost_bytes']<=choice['budget_bytes']]
            objective=choice['objective']
            def key(r):
                score=r['choices'][objective]['validation']
                return ((score['loss'],r['cost_bytes'],r['model_id']) if objective=='loss' else
                        (-score['f1_macro'],score['loss'],r['cost_bytes'],r['model_id']))
            if not pool:assert choice['chosen'] is None
            else:
                selected=min(pool,key=key);assert choice['chosen']['model_id']==selected['model_id']
                assert choice['chosen']['policy']==selected['choices'][objective]['policy']
            checked_budgets+=1
    fixed=list(csv.DictReader((root/'fixed_models.csv').open(encoding='utf-8')))
    budgets=list(csv.DictReader((root/'budget_choices.csv').open(encoding='utf-8')))
    checked_metrics=0;max_error=0.
    def compare(value,expected):
        nonlocal checked_metrics,max_error
        error=abs(float(value)-expected);assert error<2e-11,(value,expected)
        max_error=max(max_error,error);checked_metrics+=1
    for row in fixed:
        r=index[(row['case'],row['family'],row['model_id'],row['scheme'],int(row['repeat']),float(row['fraction']))]
        c=len(r['fitted_temperature']['shifts'])
        raw=q(row['case'],row['family'],row['model_id'],dict(temperature=1.,shifts=[0.]*c))
        temp=q(row['case'],row['family'],row['model_id'],r['fitted_temperature'])
        compare(row['raw_loss'],raw['loss']);compare(row['raw_f1'],raw['f1_macro'])
        compare(row['temperature_loss'],temp['loss']);compare(row['temperature_brier'],temp['brier'])
        for objective in ('loss','f1_macro'):
            score=q(row['case'],row['family'],row['model_id'],r['choices'][objective]['control'])
            compare(row[objective+'_test_loss'],score['loss']);compare(row[objective+'_test_f1'],score['f1_macro'])
            compare(row[objective+'_test_brier'],score['brier'])
        if r.get('matched_refit'):
            matched=r['matched_refit']
            compare(row['matched_temperature_loss'],q(row['case'],row['family'],row['model_id'],matched['fitted_temperature'])['loss'])
            compare(row['matched_selected_loss'],q(row['case'],row['family'],row['model_id'],matched['choices']['loss'])['loss'])
            compare(row['matched_selected_f1'],q(row['case'],row['family'],row['model_id'],matched['choices']['f1_macro'])['f1_macro'])
    for row in budgets:
        if row['status']!='evaluated':continue
        r=index[(row['case'],'pyboost',row['model_id'],row['scheme'],int(row['repeat']),float(row['fraction']))]
        score=q(row['case'],'pyboost',row['model_id'],r['choices'][row['objective']]['control'])
        compare(row['test_loss'],score['loss']);compare(row['test_f1'],score['f1_macro'])
    for path,sha in seal['input_locks'].items():assert digest(path)==sha,path
    write(args.output/'audit_report.json',dict(status='passed',source_report_sha256=src['sha256'],
        pretest_seal_sha256=report['pretest_seal_sha256'],policy_choices_checked=checked_policies,
        budget_choices_checked=checked_budgets,pyboost_array_ledgers_checked=checked_arrays,
        scalar_metrics_checked=checked_metrics,max_metric_error=max_error,input_exports=exported,
        fixed_rows=len(fixed),budget_rows=len(budgets),new_tree_fits=0,new_output_fits=0,
        test_used_for_selection=False,gpu_execution=False))
    event('calibration_independent_audit_passed',policy_choices=checked_policies,budget_choices=checked_budgets,
          exact_array_ledgers=checked_arrays,scalar_metrics=checked_metrics,max_metric_error=max_error,
          fixed_rows=len(fixed),budget_rows=len(budgets),new_tree_fits=0,new_output_fits=0)


if __name__=='__main__':main()
