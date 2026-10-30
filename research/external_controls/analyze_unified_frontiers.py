"""Rebuild inspectable CSV frontiers from exported SHA-verified JSON metadata."""
from pathlib import Path
import argparse,csv,json,hashlib


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');a=parser.parse_args();root=Path(a.directory);data=root/'data'
    report=json.loads((data/'report.json').read_text());points=[];slices=[];cases=[]
    for entry in report['cases']:
        case=json.loads((data/entry['case']/'case.json').read_text());task=case['task'];obj='loss' if task=='multiclass' else 'f1_macro'
        local=[]
        for m in case['models']:
            e=next(r for r in m['evaluations'] if r['objective']==obj);q=e['quality'];r=case['reference']['quality']
            row=dict(case=case['case'],dataset=case['case'].rsplit('_',1)[0],task=task,uid=m['uid'],family=m['family'],
                objective=obj,quality=q[obj],nll=q['loss'],f1_macro=q['f1_macro'],cost_bytes=m['cost_bytes'],memory_ratio=m['cost_bytes']/case['reference_bytes'],
                relative_nll=q['loss']/r['loss'],relative_macro_f1=q['f1_macro']/max(r['f1_macro'],1e-30),
                trees=m['trees'],blocks=m['blocks'],rank=m.get('rank'),leaf_cap=m.get('leaf_cap'),nnz=m['nnz'],
                coefficient_values=m['coefficient_values'],policy=e['policy'],
                temperature=e['refit_control']['temperature'])
            local.append(row)
        points.extend(local)
        for ratio in (.2,.3,.4,.5,.7,1.,1.3):
            for family in ('ordinary','posthoc','online'):
                pool=[r for r in local if r['family']==family and r['memory_ratio']<=ratio]
                if not pool:slices.append(dict(case=case['case'],ratio=ratio,family=family,status='infeasible'));continue
                best=min(pool,key=lambda r:(r['nll'] if obj=='loss' else -r['f1_macro'],r['cost_bytes'],r['uid']))
                slices.append(dict(case=case['case'],ratio=ratio,family=family,status='measured',**{k:v for k,v in best.items() if k not in ('case','family')}))
        cases.append(dict(case=case['case'],models=len(local),audit=case['audit'],reference=case['reference'],reference_bytes=case['reference_bytes'],data_reuse=case['record'].get('data_reuse')))
    def save(name,rows):
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (root/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    save('points.csv',points);save('budget_slices.csv',slices)
    result=dict(status=report['status'],source_report_sha256=hashlib.sha256((data/'report.json').read_bytes()).hexdigest(),cases=cases,
        datasets=len({r['dataset'] for r in points}),models=len(points),paid_new_trees=report['paid_new_trees'],
        all_audits_passed=all(c['audit']['status']=='passed' for c in cases),test_evaluated=report['test_evaluated'])
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','datasets','models','paid_new_trees','all_audits_passed','test_evaluated')}))
    for r in slices:
        if r['ratio']==.7 and r['status']=='measured':print(r['case'],r['family'],r['uid'],round(r['relative_nll'],4),round(r['relative_macro_f1'],4),r['cost_bytes'])


if __name__=='__main__':main()
