"""Independent saved-choice/metric/array audit; no fitting or GPU execution."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from budget_quality_worker import digest,write,locked,event,native_load
from audit_calibration_pilot import independent_quality,policy_rank,policy_meta


def policies(mode,objective):
    names=['raw','temperature','raw_shift_0.25','raw_shift_0.5','raw_shift_1',
        'temperature_shift_0.25','temperature_shift_0.5','temperature_shift_1']
    result=[]
    for name in names:
        t,f=policy_meta(name)
        if mode=='raw' and name!='raw':continue
        if mode=='temperature' and f:continue
        if mode=='shifts' and t:continue
        if objective=='loss' and f:continue
        result.append(name)
    return result


def in_group(meta,group):
    family=meta['family']
    return ((group=='native' and family=='pyboost') or
        (group=='ab' and family=='compact' and meta['nnz']==0) or
        (group=='abl' and family=='compact') or (group==family and family in ('catboost','xgboost','tabpfn8')))


def independent_choice(rows,metas,choice):
    items=[]
    for row in rows:
        meta=metas[row['uid']]
        if row['plan']!=choice['plan'] or not in_group(meta,choice['group']):continue
        if 'budget_bytes' in choice and (meta['cost_bytes'] is None or meta['cost_bytes']>choice['budget_bytes']):continue
        name=min(policies(choice['mode'],choice['objective']),
            key=lambda n:policy_rank(n,row['validation'][n],choice['objective']))
        score=row['validation'][name];size=meta['cost_bytes'] or 0
        key=(score['loss'],size,row['uid']) if choice['objective']=='loss' else (-score['f1_macro'],score['loss'],size,row['uid'])
        items.append((key,dict(uid=row['uid'],policy=name,validation=score,cost_bytes=meta['cost_bytes'])))
    return min(items,key=lambda x:x[0])[1] if items else None


def control(row,name):
    t,f=policy_meta(name);fit=row['refit']
    shifts=np.asarray(fit['temperature_shifts' if t else 'raw_shifts'],np.float32)
    return dict(temperature=fit['temperature'] if t else 1.,
        shifts=(shifts*f).astype(np.float32).tolist())


def plans_check(case):
    plans=locked(case['plans_path'],case['plans_sha256'])
    roles=case['record']['case']
    assert digest(roles['split_file'])==roles['split_sha256']
    with np.load(roles['split_file'],allow_pickle=False) as f:
        train=f['train'];test=f['test'];dev=np.concatenate([f[k] for k in ('calibration','control','selection')])
    np.testing.assert_array_equal(plans['train_ids'],train)
    np.testing.assert_array_equal(plans['test_ids'],test)
    np.testing.assert_array_equal(plans['dev_ids'],dev)
    assert not set(dev)&(set(train)|set(test))
    assert len(plans['plans'])==20
    n=len(dev)
    for repeat in range(10):
        group=[p for p in plans['plans'] if p['repeat']==repeat]
        assert sorted(p['scheme'] for p in group)==['crossfit','pooled']
        for plan in group:
            validation=[]
            assert plan['fraction']==1.
            for fold in plan['folds']:
                t,s,v=(fold[k] for k in ('temperature','shifts','validation'))
                assert t==s
                assert not set(t)&set(v)
                assert all(len(ids)==len(set(ids)) and min(ids)>=0 and max(ids)<n for ids in (t,s,v))
                if plan['scheme']=='crossfit':assert set(t)|set(v)==set(range(n))
                validation.extend(v)
            assert len(validation)==len(set(validation))
            if plan['scheme']=='crossfit':
                assert sorted(validation)==list(range(n))
                assert plan['refit_temperature']==plan['refit_shifts']==list(range(n))
            else:
                assert plan['refit_temperature']==plan['refit_shifts']==plan['folds'][0]['temperature']


def guard(a,b):
    return bool(a['loss']<=1.05*b['loss'] and a['f1_macro']>=.95*b['f1_macro'] and a['f1_weighted']>=.95*b['f1_weighted'])


def audit_case(case,result,root,*,select=independent_choice,check_plans=plans_check):
    check_plans(case);metas=case['metas'];rows={(r['uid'],r['plan']):r for r in case['rows']}
    for choice in (*case['budget_choices'],*case['external_choices']):
        assert select(case['rows'],metas,choice)==choice['chosen']
    manifest=json.loads((root/'test_predictions'/case['case_id']/'manifest.json').read_text())
    assert digest(manifest['target_file'])==manifest['target_sha256']
    with np.load(manifest['target_file'],allow_pickle=False) as f:target=f['target'];ids=f['indices']
    raw={}
    for uid,item in manifest['predictions'].items():
        assert digest(item['path'])==item['sha256']
        with np.load(item['path'],allow_pickle=False) as f:raw[uid]=f['raw'].copy()
    source=case['record']['dataset']
    assert digest(source['path'])==source['sha256']
    with np.load(source['path'],allow_pickle=False) as f:
        probe=f['features'][ids[:17]].astype(np.float32)
        np.testing.assert_array_equal(target,f['target'][ids])
    ledger_count=0
    for uid,meta in metas.items():
        assert digest(meta['source']['artifact'])==meta['source']['sha256']
        if meta['family'] not in ('compact','pyboost'):
            assert meta['cost_bytes'] is None;continue
        model=native_load(meta['source']['artifact']);prepared=model.prepare_inference()
        actual=[getattr(prepared.base,name) for name in ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias')]
        actual.extend(getattr(prepared,name) for name in ('indptr','columns','values','decision_shifts'))
        price=sum(a.nbytes for a in actual)+4
        assert meta['cost_bytes']==price==meta['ledger']['cost_bytes']
        assert meta['ledger']['both_retained_bytes']==price+model.payload_bytes
        assert meta['ledger']['parameters']==model.parameter_count+1
        assert not np.shares_memory(prepared.base.nodes,model.base.nodes)
        np.testing.assert_allclose(prepared.raw(probe),raw[uid][:17],atol=2e-4,rtol=2e-5)
        if meta['family']=='compact':
            assert meta['nnz']==len(model.values)
            assert all(meta['nnz']<=p['maximum_nnz'] for p in meta['requested_plans'])
            assert model.base.representation=='low_rank'
        ledger_count+=1
    cache={}
    def measured(uid,name,plan):
        c=control(rows[uid,plan],name);key=(uid,c['temperature'],tuple(c['shifts']))
        if key not in cache:cache[key]=independent_quality(target,raw[uid],c)
        return cache[key],c
    all_choices=(*case['budget_choices'],*case['external_choices'])
    all_results=(*result['budget_points'],*result['external_points'])
    assert len(all_choices)==len(all_results)
    for choice,item in zip(all_choices,all_results):
        selected=choice['chosen']
        if selected is None:
            assert item['status']=='no_feasible_candidate';continue
        uid,name=selected['uid'],selected['policy'];q,c=measured(uid,name,choice['plan'])
        assert item['uid']==uid and item['policy']==name and item['control']==c
        for metric,value in q.items():assert abs(item['test'][metric]-value)<2e-12
        parent=metas[uid].get('parent_uid',uid);parent_q,_=measured(parent,name,choice['plan'])
        assert item['own_parent_guard_5percent']==guard(q,parent_q)
    for item in result['raw_compression']:
        q=independent_quality(target,raw[item['uid']],dict(temperature=1.,shifts=[0.]*raw[item['uid']].shape[1]))
        for name,value in q.items():assert abs(item['test_'+name]-value)<2e-12
    for item in result['bootstrap_intervals']:
        assert item['repeats']==500 and item['class_domain']==list(range(case['record']['case']['outputs']))
        assert item['support']==np.bincount(target,minlength=len(item['class_domain'])).tolist()
        a=independent_quality(target,raw[item['left_uid']],item['left_control'])
        b=independent_quality(target,raw[item['right_uid']],item['right_control'])
        assert abs(item['delta_loss']-(a['loss']-b['loss']))<2e-12
        assert abs(item['delta_f1_pp']-100*(a['f1_macro']-b['f1_macro']))<2e-10
    return dict(case=case['case_id'],choices_replayed=len(all_choices),ledger_count=ledger_count,
        independently_measured_controls=len(cache),prediction_caches=len(raw),
        raw_diagnostics=len(result['raw_compression']),bootstrap_pairs=len(result['bootstrap_intervals']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    source=cfg['abl_study_completed'];report=locked(source['path'],source['sha256']);root=Path(source['path']).parent
    assert report['status']=='complete' and report['new_tree_fits']==0 and not report['test_used_for_selection']
    seal=locked(root/'pretest_choices.json',report['pretest_seal_sha256'])
    assert seal['status']=='abl_choices_sealed_before_test'
    checks=[]
    for entry in seal['cases']:
        case=locked(entry['path'],entry['sha256'])
        item=next(e for e in report['evaluations'] if e['case']==case['case_id'])
        result=locked(item['path'],item['sha256'])
        checks.append(audit_case(case,result,root));event('abl_independent_case_audited',**checks[-1])
    for path,expected in seal['input_locks'].items():assert digest(path)==expected
    value=dict(status='passed',source=source,cases=checks,new_tree_fits=0,gpu_execution=False,
        test_used_for_selection=False,immutable_input_files=len(seal['input_locks']),
        pretest_seal_sha256=report['pretest_seal_sha256'])
    write(args.output/'audit_report.json',value)
    event('abl_independent_audit_passed',path=str(args.output/'audit_report.json'),
        sha256=digest(args.output/'audit_report.json'),cases=len(checks))


if __name__=='__main__':main()
