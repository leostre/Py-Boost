"""Offline evidence replay for the approved stage1; no fits or GPU work.

Run through the unchanged external OpenResearch launcher, not raw SSH Python.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score

from budget_quality_worker import digest, write, locked, event, native_load
from budget_quality_core import quality, probabilities


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    src=cfg['budget_quality_source'];report=locked(src['report'],src['report_sha256'])
    root=Path(src['report']).parent
    seal=locked(root/'pretest_choices.json',report['pretest_seal_sha256'])
    audit=json.loads((root/'whole_cohort_audit.json').read_text())
    assert audit['status']=='passed' and audit['pretest_seal_sha256']==report['pretest_seal_sha256']
    assert report['test_used_for_selection'] is False and report['quality_gate_5percent'] is False
    for path,sha in seal['input_locks'].items():
        assert digest(path)==sha, path
    models={(r['case_id'],r['family'],m['identity']):m for r in seal['records'] for m in r['models']}
    fits={}
    for (case,family,identity),m in models.items():
        if m['origin']=='new': fits[(case,family,m['depth'])]=m['fit']
    assert len(fits)==30 and sum(f['rounds'] for f in fits.values())==24000
    assert all(f['new_fits']==1 and f['rounds']==800 for f in fits.values())
    trajectory_ledger=[]
    for (case,family,depth),fit in fits.items():
        m=models[(case,family,f'd{depth}-t800')]
        trajectory_ledger.append({**fit,'case':case,'family':family,'depth':depth,
            'scalar_or_vector_trees':m['ledger']['trees'],'leaves':m['ledger']['leaves'],
            'leaf_parameters':m['ledger']['leaf_parameters']})
    # Ranking reference is deliberately explicit and independent of select().
    checked=0
    for choice in seal['choices']:
        if choice['objective']=='historical200': continue
        case,family,policy=choice['case_id'],choice['family'],choice['policy']
        pool=[row for (c,f,_),m in models.items() if c==case and f==family and m['origin']!='archival_control'
              for row in m['rows'] if row['policy']==policy]
        if family in ('catboost','xgboost'):
            pool=[r for r in pool if r['rounds']==int(choice['budget_label'][6:])]
        if choice['budget_bytes'] is not None: pool=[r for r in pool if r['cost_bytes']<=choice['budget_bytes']]
        def key(r):
            q=r['selection'];tail=(r['cost_bytes'],r['uid'])
            return (q['loss'],-q['f1_macro'],*tail) if choice['objective']=='loss' else (-q['f1_macro'],q['loss'],*tail)
        expected=sorted(pool,key=key)[0] if pool else None
        assert expected==choice['chosen']
        checked+=1
    predictions={};replayed={};details=[]
    for choice in report['results']:
        if choice['status']!='evaluated': continue
        row=choice['chosen'];case=choice['case_id'];family=choice['family'];modelid=row['model_id']
        key=(case,family,modelid)
        if key not in predictions:
            manifest=json.loads((root/'test'/case/family/'test.json').read_text())
            item=manifest['predictions'][modelid]
            assert digest(item['path'])==item['sha256']
            with np.load(item['path']) as f: predictions[key]={k:f[k].copy() for k in f.files}
        pred=predictions[key];z=pred['raw'];y=pred['target'];c=z.shape[1]
        actual=quality(y,z,row['temperature'],row['decision_shifts'])
        for metric,value in actual.items(): assert abs(value-choice['test'][metric])<1e-12
        labels=(z/row['temperature']-row['decision_shifts']).argmax(1)
        assert abs(f1_score(y,labels,labels=np.arange(c),average='macro',zero_division=0)-actual['f1_macro'])<1e-12
        assert abs(f1_score(y,labels,labels=np.arange(c),average='weighted',zero_division=0)-actual['f1_weighted'])<1e-12
        if row['uid'] not in replayed:
            directory=args.output/'selected_controls'/case/family/modelid/row['policy']/str(row['shift_fraction'])
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/'controls.npz'
            t=np.array([row['temperature']],np.float32);shifts=np.array(row['decision_shifts'],np.float32)
            np.savez(path,temperature=t,decision_shifts=shifts)
            assert t.nbytes+shifts.nbytes==row['calibration_bytes']
            m=models[key];ledger=m['ledger']
            if family=='pyboost':
                deployed=native_load(m['artifact']).prepare_inference()
                arrays=[getattr(deployed.base,n) for n in ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias')]
                arrays += [getattr(deployed,n) for n in ('indptr','columns','values','decision_shifts')]
                assert sum(a.nbytes for a in arrays)==ledger['deployed_array_bytes']
            expected=(ledger['deployed_array_bytes']+t.nbytes if family=='pyboost'
                      else ledger['native_file_bytes']+shifts.nbytes+t.nbytes)
            assert expected==row['cost_bytes']
            replayed[row['uid']]=dict(path=str(path),sha256=digest(path),array_bytes=t.nbytes+shifts.nbytes)
        details.append(dict(choice,ledger=models[key]['ledger'],fit=models[key]['fit']))
    # Original d5/200 prefixes must reproduce the frozen models, not a new task.
    prefix_checks=[]
    for record in seal['records']:
        if record['family']=='tabpfn8': continue
        new=next(m for m in record['models'] if m['identity']=='d5-t200')
        old=next(m for m in record['models'] if m['identity']=='archival-d5-t200')
        with np.load(new['development_file']) as a,np.load(old['development_file']) as b:
            errors={r:float(np.max(np.abs(a[r]-b[r]))) for r in ('calibration','control','selection')}
        # Some native libraries can vary GPU reductions. Report, don't conceal.
        prefix_checks.append(dict(case=record['case_id'],family=record['family'],errors=errors))
    write(args.output/'audit_report.json',dict(status='passed',source_run=src['run_id'],
        source_report_sha256=src['report_sha256'],selection_choices_replayed=checked,
        unique_test_models=len(predictions),policy_controls=replayed,trajectory_ledger=trajectory_ledger,
        original_prefix_replay=prefix_checks,results=details,new_fits=0,nn_weight_updates=0,
        test_used_for_selection=False,conditional_bootstrap_repeats=500))
    event('stage1_independent_evidence_verified',choices=checked,unique_test_models=len(predictions),
          scientific_trajectories=len(fits),boosting_rounds=sum(f['rounds'] for f in fits.values()),new_fits=0)
    # Compact logged evidence: independent objectives, not a test-best envelope.
    summary=[]
    for case in report['datasets']:
        for family in ('pyboost','catboost','xgboost','tabpfn8'):
            for policy in ('raw','temperature','raw_shifts','temperature_shifts'):
                for objective in ('loss','f1_macro'):
                    group=[r for r in details if r['case_id']==case and r['family']==family and
                           r['policy']==policy and r['objective']==objective]
                    if family=='pyboost':
                        row=next(r for r in group if r['budget_label']==4)
                    else:
                        # Same predeclared selection metric; all per-round
                        # winners were already sealed and tested by the parent.
                        def rank(r):
                            q=r['chosen'];s=q['selection']
                            return (s['loss'],-s['f1_macro'],q['cost_bytes'],q['uid']) if objective=='loss' else (-s['f1_macro'],s['loss'],q['cost_bytes'],q['uid'])
                        row=min(group,key=rank)
                    summary.append(row)
                    event('stage1_quality_summary',case=case,family=family,policy=policy,objective=objective,
                          model=row['chosen']['model_id'],temperature=row['chosen']['temperature'],
                          shift_fraction=row['chosen']['shift_fraction'],cost_bytes=row['chosen']['cost_bytes'],
                          test=row['test'],comparison=row['comparison'])
    decomposition=[]
    for row in summary:
        if row['objective']!='loss' or row['policy']!='raw': continue
        case,family=row['case_id'],row['family']
        entries=[('selection_chosen_raw',row)]
        if family!='tabpfn8':
            entries.append(('archival_d5_t200',next(r for r in details if r['case_id']==case and
                r['family']==family and r['objective']=='historical200' and r['policy']=='raw')))
        for label,entry in entries:
            chosen=entry['chosen'];model=models[(case,family,chosen['model_id'])]
            temp=next(r for r in model['rows'] if r['policy']=='temperature')
            pred=predictions[(case,family,chosen['model_id'])]
            a=quality(pred['target'],pred['raw']);b=quality(pred['target'],pred['raw'],temp['temperature'])
            decomposition.append(dict(case=case,family=family,role=label,model=chosen['model_id'],
                temperature=temp['temperature'],raw_test=a,temperature_test=b,ledger=model['ledger'],
                raw_selection=next(r['selection'] for r in model['rows'] if r['policy']=='raw'),
                temperature_selection=temp['selection']))
    write(args.output/'summary.json',dict(results=summary,fixed_model_calibration_effects=decomposition,
        interpretation='selection-chosen only; not best test models',
        tabpfn_cost_warning='cost_bytes is saved task-context file plus controls ONLY, excludes the once-shared frozen checkpoint; never interpreted as resident model bytes'))
    event('stage1_analysis_complete',audit_sha256=digest(args.output/'audit_report.json'),
          summary_sha256=digest(args.output/'summary.json'),new_fits=0)


if __name__=='__main__': main()
