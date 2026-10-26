"""Same-policy references, without fitting or selecting anything new."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from budget_quality_worker import digest, locked, write, event
from budget_quality_core import quality, paired_bootstrap


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text());src=cfg['budget_quality_source']
    original=locked(src['report'],src['report_sha256']);root=Path(src['report']).parent
    seal=locked(root/'pretest_choices.json',original['pretest_seal_sha256'])
    checked=locked(cfg['completed_audit']['path'],cfg['completed_audit']['sha256'])
    assert checked['status']=='passed' and checked['source_report_sha256']==src['report_sha256']
    result=copy.deepcopy(original);cache={};intervals={}
    references={(r['case_id'],r['policy'],r['objective']):r for r in original['results']
                if r['family']=='tabpfn8' and r['status']=='evaluated'}
    def prediction(case,family,model):
        key=(case,family,model)
        if key not in cache:
            manifest=json.loads((root/'test'/case/family/'test.json').read_text())
            item=manifest['predictions'][model];assert digest(item['path'])==item['sha256']
            with np.load(item['path']) as f: cache[key]={k:f[k].copy() for k in f.files}
        return cache[key]
    for row in result['results']:
        if row['status']!='evaluated': continue
        r=row['chosen'];objective=row['objective'] if row['objective']!='historical200' else 'loss'
        ref=references[(row['case_id'],row['policy'],objective)]['chosen']
        a=prediction(row['case_id'],row['family'],r['model_id'])
        b=prediction(row['case_id'],'tabpfn8',ref['model_id'])
        assert np.array_equal(a['indices'],b['indices']) and np.array_equal(a['target'],b['target'])
        key=(r['uid'],ref['uid'])
        if key not in intervals:
            intervals[key]=paired_bootstrap(a['target'],(a['raw'],r['temperature'],r['decision_shifts']),
                (b['raw'],ref['temperature'],ref['decision_shifts']),cfg['bootstrap_repeats'])
        row['unshifted_reference_comparison']=row['comparison']
        row['comparison']=intervals[key]
        row['reference_policy']=row['policy']
        row['reference_chosen']=ref
        row['reference_test']=quality(b['target'],b['raw'],ref['temperature'],ref['decision_shifts'])
        if row['family']=='tabpfn8':
            assert row['comparison']['delta_loss_ci95']==[0.,0.]
            assert row['comparison']['delta_f1_pp_ci95']==[0.,0.]
    result.update(reference='Same-policy frozen TabPFN8; own control-fitted shifts and selection fraction; unshifted references retained as secondary',
                  new_fits_in_postprocessing=0,source_report_sha256=src['report_sha256'],
                  source_audit_sha256=cfg['completed_audit']['sha256'],stage1_complete=True)
    write(args.output/'matched_policy_report.json',result)
    for case in original['datasets']:
        for policy in ('raw','temperature','raw_shifts','temperature_shifts'):
            row=next(r for r in result['results'] if r['case_id']==case and r['family']=='pyboost' and
                     r['policy']==policy and r['objective']=='f1_macro' and r['budget_label']==4)
            event('matched_policy_macro_summary',case=case,policy=policy,chosen=row['chosen']['model_id'],
                  test=row['test'],reference_test=row['reference_test'],comparison=row['comparison'])
    event('stage1_same_policy_reference_complete',rows=len(result['results']),new_fits=0,
          report_sha256=digest(args.output/'matched_policy_report.json'))


if __name__=='__main__': main()
