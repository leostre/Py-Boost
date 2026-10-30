"""Bounded numerical-selection diagnostic over frozen Q1 predictions."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from budget_quality_worker import digest,write,locked,event
from budget_quality_core import quality
from abl_quality_core import fitted_control,MODES
from abl_stability_core import NumericalTolerance,select_stable


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'));output=args.output
    from audit_abl_serving import audit_serving
    serving_audit=audit_serving(cfg)
    write(output/'serving_audit.json',serving_audit)
    event('abl_serving_independently_audited',**serving_audit)
    source=cfg['abl_study_completed'];report=locked(source['path'],source['sha256']);root=Path(source['path']).parent
    seal=locked(root/'pretest_choices.json',report['pretest_seal_sha256']);cases=[];choices=[]
    variants=[('exact',NumericalTolerance()),('numeric',NumericalTolerance(1e-6,1e-12))]
    for entry in seal['cases']:
        case=locked(entry['path'],entry['sha256']);cases.append(case)
        base=next(c for c in case['budget_choices'] if c['scheme']=='crossfit' and c['repeat']==0)
        index=base['plan'];rows=[r for r in case['rows'] if r['plan']==index]
        for name,tolerance in variants:
            for overhead in (0,4):
                for mode in MODES:
                    for objective in ('loss','f1_macro'):
                        for group in ('native','ab','abl','deployed'):
                            for ratio in cfg['budget_ratios']:
                                budget=int(np.floor(case['reference_b0_bytes']*ratio))+overhead
                                chosen=select_stable(rows,case['metas'],group,mode,objective,budget,tolerance)
                                if name=='exact' and overhead==0 and group!='deployed':
                                    previous=next(c for c in case['budget_choices'] if c['plan']==index
                                        and c['mode']==mode and c['objective']==objective and c['group']==group and c['budget_ratio']==ratio)
                                    if chosen!=previous['chosen']:raise RuntimeError('zero-tolerance selector does not replay Q1')
                                choices.append(dict(case=case['case_id'],plan=index,variant=name,
                                    numerical_tolerance=asdict(tolerance),common_overhead_bytes=overhead,
                                    scheme='crossfit',repeat=0,mode=mode,objective=objective,group=group,
                                    budget_ratio=ratio,budget_bytes=budget,chosen=chosen))
    if len(choices)!=2688:raise RuntimeError('bounded diagnostic grid changed')
    choice_path=output/'pretest_choices.json'
    write(choice_path,dict(status='numerical_diagnostic_choices_sealed',choices=choices,source=source,
        model_fits=0,calibration_fits=0,test_is_reused_exploratory=True))
    event('abl_numerical_choices_sealed',choices=len(choices),sha256=digest(choice_path))
    results=[];cache={};counts=Counter()
    for case in cases:
        case_id=case['case_id'];metas=case['metas'];rows={(r['uid'],r['plan']):r for r in case['rows']}
        predictions=json.loads((root/'test_predictions'/case_id/'manifest.json').read_text())
        if digest(predictions['target_file'])!=predictions['target_sha256']:raise RuntimeError('frozen targets changed')
        with np.load(predictions['target_file'],allow_pickle=False) as f:y=f['target'].copy()
        raw={}
        for choice in (c for c in choices if c['case']==case_id):
            item={k:v for k,v in choice.items() if k!='chosen'};chosen=choice['chosen']
            if chosen is None:results.append(dict(item,status='no_feasible_candidate'));continue
            uid=chosen['uid'];policy=chosen['policy'];meta=metas[uid]
            if uid not in raw:
                prediction=predictions['predictions'][uid]
                if digest(prediction['path'])!=prediction['sha256']:raise RuntimeError('frozen logits changed')
                with np.load(prediction['path'],allow_pickle=False) as f:raw[uid]=f['raw'].copy()
            control=fitted_control(rows[uid,choice['plan']],policy)
            key=(case_id,uid,control.temperature,control.shifts)
            if key not in cache:cache[key]=quality(y,raw[uid],control.temperature,control.shifts)
            counts[choice['variant']+'+'+str(choice['common_overhead_bytes'])+':'+meta['family']]+=1
            results.append(dict(item,status='evaluated',uid=uid,policy=policy,validation=chosen['validation'],
                control=asdict(control),test=cache[key],cost_bytes=meta['cost_bytes'],rank=meta['rank'],nnz=meta['nnz'],
                selected_family=meta['family'],actual_compression=meta['family']=='compact' and
                    meta['cost_bytes']<metas[meta['parent_uid']]['cost_bytes']))
        del raw
    value=dict(status='complete',source=source,serving_audit=serving_audit,
        results=results,selection_family_counts=dict(counts),
        pretest_seal_sha256=digest(choice_path),new_tree_fits=0,new_calibration_fits=0,
        test_is_reused_exploratory=True,test_used_for_selection=False,
        purpose='numerical engineering diagnostic prompted by Q1; original Q1 and serving selections remain unchanged',
        new_uncertainty_intervals=False)
    write(output/'report.json',value)
    event('abl_numerical_diagnostic_complete',path=str(output/'report.json'),sha256=digest(output/'report.json'))


if __name__=='__main__':main()
