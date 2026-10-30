"""Paired fixed-work endpoint comparisons; never choose a method from test."""
from pathlib import Path
import argparse,csv,json,hashlib


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');a=parser.parse_args();root=Path(a.directory);source=root/'data'
    report=json.loads((source/'report.json').read_text());rows=[];deltas=[]
    for entry in report['cases']:
        case=json.loads((source/entry['case']/'case.json').read_text());task=case['task'];objective='loss' if task=='multiclass' else 'f1_macro'
        q=lambda m:next(e['quality'] for e in m['evaluations'] if e['objective']==objective)
        full={m['block']:m for m in case['models'] if m['family']=='one_full'}
        endpoints={}
        for m in case['models']:
            quality=q(m);baseline=q(full[m['block']]);row=dict(case=case['case'],dataset=case['case'].rsplit('_',1)[0],task=task,method=m['family'],
                block=m['block'],trees=m['trees'],cost_bytes=m['cost_bytes'],memory_ratio=m['cost_bytes']/full[m['block']]['cost_bytes'],
                nll=quality['loss'],f1_macro=quality['f1_macro'],relative_nll=quality['loss']/baseline['loss'],
                delta_macro_pp=100*(quality['f1_macro']-baseline['f1_macro']),rank=m['rank'],nnz=m['nnz'],branches=m['branches'],weight=m['branch_weight'])
            rows.append(row)
            if m['block']==3:endpoints[m['family']]=row
        for treatment,control in [('one_history','one_current'),('one_mean','one_current'),('one_history','one_functional'),('one_current','one_functional'),
            ('two_history_sum','two_current_sum'),('two_history_average','two_current_average'),('two_current_sum','one_current'),('two_history_sum','one_history'),('two_history_average','one_history')]:
            t,b=endpoints[treatment],endpoints[control]
            deltas.append(dict(case=case['case'],dataset=t['dataset'],task=task,treatment=treatment,control=control,
                delta_nll=t['nll']-b['nll'],delta_f1_macro_pp=100*(t['f1_macro']-b['f1_macro']),delta_bytes=t['cost_bytes']-b['cost_bytes']))
    def save(name,data):
        with (root/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save('points.csv',rows);save('matched_differences.csv',deltas)
    result=dict(status=report['status'],source_report_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),models=len(rows),cases=len(report['cases']),
        paid_new_trees=report['paid_new_trees'],all_audits_passed=all(r['audit']['status']=='passed' for r in report['cases']),
        actual_checkpoint_checks=sum(r['audit']['actual_checkpoint_initializations'] for r in report['cases']),
        overlapping_new_branch_epochs=sum(r['audit']['overlapping_new_branch_epochs'] for r in report['cases']),test_evaluated=report['test_evaluated'])
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
    for d in deltas:
        if d['treatment']=='one_history' and d['control']=='one_current':print(d)


if __name__=='__main__':main()
