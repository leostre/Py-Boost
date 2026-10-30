"""Independent W1 topology, stable-ranking and saved-prediction audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from budget_quality_worker import digest,write,locked,event,native_load
from audit_abl_quality import audit_case,independent_choice,policies,policy_meta,in_group


def width_choice(rows,metas,choice):
    if choice['group'] in ('catboost','xgboost','tabpfn8'):
        return independent_choice(rows,metas,choice)
    objective=choice['objective'];candidates=[]
    for row in rows:
        meta=metas[row['uid']]
        member=(meta['family'] in ('pyboost','compact') if choice['group']=='deployed' else in_group(meta,choice['group']))
        if row['plan']!=choice['plan'] or not member or meta['cost_bytes']>choice['budget_bytes']:continue
        names=policies(choice['mode'],objective)
        if objective=='f1_macro':
            maximum=max(row['validation'][name]['f1_macro'] for name in names)
            names=[name for name in names if row['validation'][name]['f1_macro']>=maximum-1e-12]
        minimum=min(row['validation'][name]['loss'] for name in names)
        names=[name for name in names if row['validation'][name]['loss']<=minimum+1e-6]
        def preference(name):
            temperature,fraction=policy_meta(name)
            return int(temperature)+int(fraction>0),fraction,name
        name=min(names,key=preference)
        candidates.append(dict(uid=row['uid'],policy=name,validation=row['validation'][name],cost_bytes=meta['cost_bytes']))
    if not candidates:return None
    if objective=='f1_macro':
        maximum=max(item['validation']['f1_macro'] for item in candidates)
        candidates=[item for item in candidates if item['validation']['f1_macro']>=maximum-1e-12]
    minimum=min(item['validation']['loss'] for item in candidates)
    candidates=[item for item in candidates if item['validation']['loss']<=minimum+1e-6]
    return min(candidates,key=lambda item:(item['cost_bytes'],item['uid']))


def width_plans_check(case):
    value=locked(case['plans_path'],case['plans_sha256'])
    assert len(value['plans'])==1;plan=value['plans'][0]
    assert plan['scheme']=='crossfit' and plan['repeat']==0 and plan['fraction']==1.
    record=case['record'];roles=record['case'];entry=record['dataset']
    assert digest(roles['split_file'])==roles['split_sha256'] and digest(entry['path'])==entry['sha256']
    with np.load(roles['split_file'],allow_pickle=False) as f:
        train=f['train'].copy();test=f['test'].copy();dev=np.concatenate([f[k] for k in ('calibration','control','selection')])
    np.testing.assert_array_equal(value['train_ids'],train)
    np.testing.assert_array_equal(value['test_ids'],test)
    np.testing.assert_array_equal(value['dev_ids'],dev)
    assert not set(train)&set(dev) and not set(test)&(set(train)|set(dev))
    with np.load(entry['path'],allow_pickle=False) as f:np.testing.assert_array_equal(value['target'],f['target'][dev])
    n=len(dev);validation=[]
    assert len(plan['folds'])==3 and plan['refit_temperature']==plan['refit_shifts']==list(range(n))
    for fold in plan['folds']:
        a,b,c=(fold[k] for k in ('temperature','shifts','validation'))
        assert a==b and not set(a)&set(c) and set(a)|set(c)==set(range(n))
        assert len(a)==len(set(a)) and len(c)==len(set(c));validation.extend(c)
    assert sorted(validation)==list(range(n))
    assert case['common_overhead_budget_bytes']==4 and case['numerical_tolerance']==dict(loss=1e-6,macro=1e-12)
    for choice in case['budget_choices']:
        assert choice['budget_bytes']==int(np.floor(case['reference_b0_bytes']*choice['budget_ratio']))+4
    assert len(case['budget_choices'])==224 and len(case['external_choices'])==24


def training_check(root,report,cases):
    ledger=json.loads((root/'training_ledger.json').read_text())
    assert ledger==report['training_ledger'] and ledger['fits']==6 and ledger['rounds']==4800
    assert len(ledger['records'])==6
    entries=[]
    for case in cases:
        record=case['record'];roles=record['case'];entry=record['dataset']
        with np.load(roles['split_file'],allow_pickle=False) as f:train=f['train'].copy()
        with np.load(entry['path'],allow_pickle=False) as f:x=f['features'][train].astype(np.float32)
        for depth in (3,5):
            path=root/'width_training'/case['case_id']/f'd{depth}'/'models'/'trajectory.json'
            trajectory=json.loads(path.read_text());fit=trajectory['fit']
            assert fit in ledger['records'] and fit['rounds']==800 and fit['actual_minimum_leaf']==50
            assert fit['own_pool_peak_bytes']<=16*1024**3 and fit['observed_minimum_leaf_support']>=50
            assert len(trajectory['models'])==6
            item=next(m for m in trajectory['models'] if m['rounds']==800)
            assert digest(item['artifact'])==item['sha256']
            model=native_load(item['artifact'])
            counts=np.bincount(model.base.leaf_indices(x).ravel(),minlength=len(model.base.coefficients))
            observed=int(counts[counts>0].min());assert observed==fit['observed_minimum_leaf_support']
            entries.append(dict(case=case['case_id'],depth=depth,minimum_leaf_support=observed,
                leaves=len(model.base.coefficients),internal_nodes=len(model.base.nodes),
                actual_prepared_bytes=model.prepare_inference().resident_bytes+4))
    return entries


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    source=cfg['abl_width_completed'];report=locked(source['path'],source['sha256']);root=Path(source['path']).parent
    assert report['status']=='complete' and report['new_model_fits']==6 and report['new_boosting_rounds']==4800
    assert report['new_requested_representations']==480 and report['reused_q1_compact_models']==480
    seal=locked(root/'pretest_choices.json',report['pretest_seal_sha256']);checks=[];cases=[]
    for entry in seal['cases']:
        case=locked(entry['path'],entry['sha256']);cases.append(case)
        result_entry=next(e for e in report['evaluations'] if e['case']==case['case_id'])
        result=locked(result_entry['path'],result_entry['sha256'])
        check=audit_case(case,result,root,select=width_choice,check_plans=width_plans_check)
        checks.append(check);event('abl_width_case_independently_audited',**check)
    topology=training_check(root,report,cases)
    for path,sha in seal['input_locks'].items():assert digest(path)==sha
    value=dict(status='passed',source=source,cases=checks,trained_topologies=topology,
        audited_new_model_fits=6,audited_new_boosting_rounds=4800,new_model_fits=0,gpu_execution=False)
    write(args.output/'audit_report.json',value)
    event('abl_width_audit_passed',path=str(args.output/'audit_report.json'),sha256=digest(args.output/'audit_report.json'))


if __name__=='__main__':main()
