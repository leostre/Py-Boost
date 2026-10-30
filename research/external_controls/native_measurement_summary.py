from pathlib import Path
import json,numpy as np
from budget_quality_worker import locked,write,event

def summarize(spec,out):
    report=locked(spec['report']['path'],spec['report']['sha256']);assert report['status']=='native_comparison_complete'
    event('sealed_native_protocol',**report['source'],request_scope=report['request_scope'],gpu_scope=report['gpu_scope'],precision=report['precision'])
    groups={}
    for row in report['rows']:groups.setdefault((row['dataset'],row['batch']),[]).append(row)
    results=[]
    for batch in report['source']['batches']:
        for role in report['source']['roles']:
            for backend in ('legacy','packed'):
                rr=[];gr=[]
                for (name,b),sessions in groups.items():
                    if b!=batch:continue
                    key=backend+':'+role;ref='native:full200'
                    rr.append(float(np.median([np.quantile(s['samples'][ref]['request_ms'],.95)/np.quantile(s['samples'][key]['request_ms'],.95) for s in sessions])))
                    gr.append(float(np.median([np.median(s['samples'][ref]['gpu_ms'])/np.median(s['samples'][key]['gpu_ms']) for s in sessions])))
                row=dict(batch=batch,role=role,backend=backend,datasets=len(rr),request_p95_speedup_median=float(np.median(rr)),gpu_p50_speedup_median=float(np.median(gr)),request_speedup_min=min(rr),request_speedup_max=max(rr))
                results.append(row);event('native_speedup',**row)
    error=max(c['probability_max_abs_error'] for c in report['checks'] if c['backend']=='native')
    result=dict(status='native_summary_complete',source=spec['report'],summary=results,checks=len(report['checks']),native_probability_max_abs_error=error,label_disagreements=sum(c['label_disagreements'] for c in report['checks']),new_trees=0,new_gpu_timings=0)
    write(out/'report.json',result);event('native_summary_complete',**result)
