"""Reporting-only evidence export: never fits, calibrates or predicts a model."""
from pathlib import Path
import json
import math

from experiments.a100_runner import emit,sha256
from experiments.guard_budget_study import locked_json


def metric_ledger(rows):
    keys=[(r['dataset'],r['seed'],r['method']) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('duplicate frozen metric row')
    for r in rows:
        available=r['loss'] is not None
        quality=(available and r['loss']<=1.05*r['reference_loss'] and
                 r['macro_f1']>=.95*r['reference_macro_f1'] and r['weighted_f1']>=.95*r['reference_weighted_f1'])
        if r['test_quality_pass']!=quality or r['joint_success']!=(quality and r['state']=='accepted'):
            raise ValueError('frozen metric flags do not match the unchanged criterion')
        if available and (r['memory_fraction']>.75 or r['coefficient_fraction']>.75):
            raise ValueError('frozen candidate exceeds the hard budget')
    totals=[]
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method]
        totals.append(dict(method=method,cases=len(group),successes=sum(r['joint_success'] for r in group),
                           new_trees=sum(r['new_trees'] for r in group)))
    return totals


def guard_evidence(config,report,output):
    source,path=locked_json(config['frozen_result_report'],config['frozen_result_report_sha256'])
    if source['status']!='complete' or not source['test_evaluated'] or source['tree_fits']!=0:
        raise RuntimeError('not an answered frozen guard result')
    seal_path=path.parent/'sealed_test_inputs.json'
    if sha256(seal_path)!=source['sealed_test_inputs_sha256']:
        raise RuntimeError('frozen result input seal changed')
    sealed=json.loads(seal_path.read_text())
    for name,expected in sealed.items():
        if sha256(name)!=expected:
            raise RuntimeError('frozen evidence file changed: '+name)
    rows=source['records'];totals=metric_ledger(rows)
    if len(rows)!=162 or len(source['audit_cases'])!=18:
        raise RuntimeError('frozen evidence coverage differs')
    for row in rows:
        emit('guard_frozen_metric',**row)
    for summary in source['summaries']:
        emit('guard_frozen_dataset_summary',**summary)
    changed=[r for r in source['paired_guard'] if r['rescued'] or r['lost']]
    for row in changed:
        emit('guard_frozen_paired_change',**row)
    report.update(status='evidence_complete',source_report=str(path),source_report_sha256=sha256(path),
        source_scientific_commit='895570f',source_test_evaluated=True,tree_fits=0,new_test_predictions=0,
        records=rows,summaries=source['summaries'],method_totals=totals,paired_changes=changed,
        sealed_input_paths=len(sealed),sealed_test_inputs_sha256=source['sealed_test_inputs_sha256'])
    emit('guard_frozen_result_ledger',source_report_sha256=sha256(path),rows=len(rows),methods=totals,
         paired_changes=changed,new_fits=0,new_predictions=0)
