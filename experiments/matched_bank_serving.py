"""Verified K5 worker adapted to five genuine matched tasks and frozen banks."""
import json
from pathlib import Path
import subprocess
import sys
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,sha256,write_report,emit
from experiments.model_bank_study import locked_json
from experiments.uniform_threshold_core import ThresholdModel
from experiments.leaf_budget_runtime import load_expanded,load_roles
from experiments.history_basis_study import raw
from experiments.output_calibration_core import calibrated_quality
from experiments.shared_leaf_basis import quality_guard

FAMILIES=('full_reference','ordinary_selected','compact_or_full','joint_selected')


def matched_bank_serving(config,report,output):
    source=Path(config['bank_source_manifest'])
    bank=locked_json(source,config['bank_source_manifest_sha256'])
    if len(bank['records'])!=20 or {r['family'] for r in bank['records']}!=set(FAMILIES):
        raise RuntimeError('matched five-task bank incomplete')
    for p,h in bank['input_locks'].items():
        if sha256(p)!=h: raise RuntimeError('matched bank source changed')
    write_report(output/'traffic_seal.json',dict(bank_sha256=sha256(source),config=config,
        test_labels_read=False,new_fits=0,request_scope='five real tasks, no independent artificial copies'))
    workers=[]
    for backend in ('cpu','gpu'):
        for family in FAMILIES:
            path=output/f'{backend}_{family}.json'
            subprocess.run([sys.executable,'-u','-m','experiments.model_bank_capacity',str(source),
                str(ROOT/'experiments/a100_config.json'),family,backend,'5',str(path)],cwd=ROOT,check=True,timeout=3600)
            workers.append(json.loads(path.read_text(encoding='utf-8')))
    quality=[]
    with threadpool_limits(limits=1):
        for row in bank['records']:
            if sha256(row['path'])!=row['sha256']: raise RuntimeError('matched bank model changed')
            x,y=load_expanded(row['dataset'],dict(datasets={row['dataset']:row['data']}))
            roles=load_roles(row,len(x))
            model=ThresholdModel.load(row['path'])
            q=calibrated_quality(y[roles['test']],raw(model,x[roles['test']]),'multiclass',model.decision_shifts)
            quality.append(dict(task_id=row['task_id'],family=row['family'],quality=q,
                development_pass=row['development_pass'],sha256=row['sha256']))
    for q in quality:
        reference=next(r['quality'] for r in quality if r['task_id']==q['task_id'] and r['family']=='full_reference')
        q.update(reference=reference,test_quality_pass=quality_guard(q['quality'],reference))
    qmap={(q['task_id'],q['family']):q for q in quality}
    constraints=[]
    for worker in workers:
        for metric in worker['metrics']:
            for ram,gpu,name in ((1024,512,'original_conditional_quota'),(8192,16384,'separate_engineering_quota')):
                memory=metric['rss_peak_bytes']<=ram*1024**2 and (worker['backend']=='cpu' or metric['gpu_process_bytes']<=gpu*1024**2)
                tasks=[]
                for row in worker['records']:
                    q=qmap[row['task_id'],worker['family']]
                    latency=metric['per_task'][row['task_id']]['p95_ms']<=100
                    tasks.append(dict(task_id=row['task_id'],development_pass=q['development_pass'],
                        test_quality_pass=q['test_quality_pass'],latency_pass=latency,
                        pass_all=bool(memory and latency and q['development_pass'] and q['test_quality_pass'])))
                constraints.append(dict(family=worker['family'],backend=worker['backend'],batch=metric['batch'],
                    traffic=metric['traffic'],quota=name,ram_mib=ram,gpu_mib=gpu,memory_pass=memory,
                    slo_ms=100,tasks_passing=sum(s['pass_all'] for s in tasks),tasks=tasks))
    if any(sha256(p)!=h for p,h in bank['input_locks'].items()): raise RuntimeError('source changed during serving')
    report.update(status='complete',workers=workers,quality=quality,constraints=constraints,new_fits=0,new_trees=0,
        scientific_fits=0,test_is_reused_exploratory=True,raw_request_timings_retained=True,
        traffic_seal_sha256=sha256(output/'traffic_seal.json'),
        timing_limit='shared A100;128requests per trace, sparse task-tail samples; not SLA certification')
    emit('matched_bank_serving_complete',workers=len(workers),requests=sum(m['requests'] for w in workers for m in w['metrics']),new_fits=0)
