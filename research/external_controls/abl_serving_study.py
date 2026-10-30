"""Sequential GPU subprocesses governed by the inherited orx command."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from budget_quality_worker import digest,locked,write,event
from abl_quality_core import fitted_control
from abl_quality_worker import check_native_environment
from abl_serving_core import serving_plans,model_key
from experiments.inference_benchmark_core import timing_summary,paired_ratios


def summarize(timings,memories):
    times=[];pairs=[];memory=[];profiles=[]
    for item in timings:
        job=item['job'];group=defaultdict(list)
        for sample in item['samples']:
            group[sample['role'],sample['batch'],sample['mode']].append(sample['milliseconds'])
        for profile in item.get('profiles',[]):
            profiles.append(dict(case=job['case'],objective=job['objective'],
                budget_ratio=job['budget_ratio'],**profile))
        for (role,batch,mode),samples in group.items():
            meta=dict(case=job['case'],objective=job['objective'],budget_ratio=job['budget_ratio'],
                role=role,batch=batch,mode=mode,uid=job['models'][role]['uid'])
            times.append(dict(meta,**timing_summary(samples,batch)))
            if role!='native' and ('native',batch,mode) in group:
                pairs.append(dict(meta,paired_median_compact_over_native=paired_ratios(group['native',batch,mode],samples),
                    repetitions=96,blocks=3,scope='paired request order; no independent-device/seed interval'))
    for item in memories:
        for record in item['records']:
            memory.append(dict(case=item['job']['case'],uid=item['job']['model']['uid'],
                batch=item['job']['batch'],mode=record['mode'],state=record['state'],
                model_bytes=item['model_bytes'],model_pool_used_bytes=item['model_pool_used_bytes'],
                pool_peak_used_bytes=record['pool_peak_used_bytes'],
                pool_peak_reserved_bytes=record['pool_peak_reserved_bytes'],
                rss_peak_bytes=record['rss']['VmHWM']))
    return dict(timings=times,paired_ratios=pairs,memory=memory,profiles=profiles)


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'));output=args.output
    native_environment=check_native_environment(cfg)
    report=locked(cfg['abl_study_completed']['path'],cfg['abl_study_completed']['sha256'])
    audit=locked(cfg['abl_audit_completed']['path'],cfg['abl_audit_completed']['sha256'])
    if report['status']!='complete' or audit['status']!='passed' or audit['source']!=cfg['abl_study_completed']:
        raise RuntimeError('independent quality audit required before serving')
    root=Path(cfg['abl_study_completed']['path']).parent
    seal=locked(root/'pretest_choices.json',report['pretest_seal_sha256'])
    plans=[]
    for item in seal['cases']:
        case=locked(item['path'],item['sha256']);plans.extend(serving_plans(case,fitted_control))
    if len(plans)!=18:raise RuntimeError('three cases, two fixed objectives, three budgets expected')
    write(output/'serving_plan.json',dict(plans=plans,source=cfg['abl_study_completed'],
        groups_variant='groups',batches=[1,32,256,1024],warmup=2,requests=32,blocks=3,
        test_targets_read=False,new_tree_fits=0))
    event('abl_serving_plan_sealed',plans=len(plans),sha256=digest(output/'serving_plan.json'),new_tree_fits=0)
    jobs=output/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    exe=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    worker=Path(__file__).resolve().parent/'abl_serving_worker.py'
    timings=[];memories=[];unique={}
    rng=np.random.default_rng(20261026)
    # Fixed random plan order; model roles are paired and randomized per request.
    for index in rng.permutation(len(plans)).tolist():
        plan=plans[index];job=dict(plan,mode='timing',plan_index=index)
        job_path=jobs/f'timing-{index:02d}.json';out=jobs/f'timing-{index:02d}-result.json'
        write(job_path,job)
        subprocess.run([exe,str(worker),str(job_path),str(out)],check=True,timeout=1800)
        timings.append(json.loads(out.read_text()))
        event('abl_serving_timing_saved',case=plan['case'],objective=plan['objective'],
            budget_ratio=plan['budget_ratio'],path=str(out),sha256=digest(out))
        for model in plan['models'].values():
            if model is not None:unique.setdefault((plan['case'],model_key(model)),(plan,model))
    for index,((case,_),(plan,model)) in enumerate(unique.items()):
        for batch in (1,32,256,1024):
            job=dict(case=case,record=plan['record'],model=model,batch=batch,mode='memory')
            job_path=jobs/f'memory-{index:02d}-{batch}.json';out=jobs/f'memory-{index:02d}-{batch}-result.json'
            write(job_path,job)
            subprocess.run([exe,str(worker),str(job_path),str(out)],check=True,timeout=300)
            memories.append(json.loads(out.read_text()))
            event('abl_serving_memory_saved',case=case,uid=model['uid'],batch=batch,path=str(out),sha256=digest(out))
    for path,expected in seal['input_locks'].items():
        if digest(path)!=expected:raise RuntimeError('source changed during serving study')
    result=dict(status='complete',native_environment=native_environment,source=cfg['abl_study_completed'],
        timing_jobs=len(timings),memory_jobs=len(memories),unique_model_controls=len(unique),
        summaries=summarize(timings,memories),new_tree_fits=0,test_targets_read=False,
        memory_scope='prepared model arrays and owned CuPy pool peaks; CPU RSS and whole-device samples are separate',
        calibration_included_in_serving_latency=True,core='unchanged groups fused tree-gather-decoder kernel')
    write(output/'report.json',result)
    event('abl_serving_complete',path=str(output/'report.json'),sha256=digest(output/'report.json'),
        timing_jobs=len(timings),memory_jobs=len(memories))


if __name__=='__main__':main()
