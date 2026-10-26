"""Seal development-only allocations, then measure and score frozen banks."""
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.a100_runner import ROOT,emit,sha256,write_report
from experiments.model_bank_study import locked_json
from experiments.bank_allocation_core import choose_allocation,key


def bank_allocation_study(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_bank_allocation_core.py'],
                   cwd=ROOT,check=True,timeout=300)
    bank=locked_json(config['bank_source_manifest'],config['bank_source_manifest_sha256'])
    choices,bank_records,unique={},[],{}
    decisions=[]
    for group_id in range(3):
        groups=[]
        for name in config['datasets']:
            seed=config['bank_seeds_by_dataset'][name][group_id]
            groups.append([r for r in bank['records'] if r['dataset']==name and r['seed']==seed])
        full_bytes=sum(next(r['csr_bytes'] for r in group if r['family']=='full') for group in groups)
        for fraction in config['allocation_fractions']:
            budget=int(full_bytes*fraction)
            for policy in ('equal','independent','joint'):
                selected=choose_allocation(groups,budget,policy)
                identifier=f'g{group_id}-{fraction}-{policy}'
                signature=tuple(r['sha256'] if r is not None else None for r in selected)
                job=unique.setdefault(signature,f'bank{len(unique)}')
                decision=dict(identifier=identifier,group_id=group_id,fraction=fraction,policy=policy,
                    budget_model_bytes=budget,reference_model_bytes=full_bytes,
                    used_model_bytes=sum(r['csr_bytes'] for r in selected if r is not None),
                    development_served=sum(r is not None for r in selected),objective=list(key(selected)),
                    selected=[dict(task_id=group[0]['task_id'],sha256=r['sha256'],family=r['family'],
                                   candidate_id=r['candidate_id']) if r is not None else
                              dict(task_id=group[0]['task_id'],sha256=None,family=None,candidate_id=None)
                              for group,r in zip(groups,selected)],job=job)
                decisions.append(decision)
                if job not in choices:
                    choices[job]=[r for r in selected if r is not None]
                    bank_records.extend([dict(r,candidate_family=r['family'],family=job)
                                         for r in selected if r is not None])
    # All 27 decisions are immutable before reading any test quality.
    write_report(output/'allocation_seal.json',dict(decisions=decisions,
        source_sha256=config['bank_source_manifest_sha256'],test_labels_read=False,scientific_fits=0))
    mixed=output/'mixed_manifest.json'
    write_report(mixed,dict(records=bank_records,scientific_fits=0,test_labels_read=False))
    emit('allocation_decisions_sealed',decisions=len(decisions),unique_banks=len(choices),
         sha256=sha256(output/'allocation_seal.json'),test_labels_read=False)
    workers=[]
    # The actual mixed bank is measured, not inferred from pure-family process memory.
    for job,rows in choices.items():
        if not rows: continue
        for backend in ('cpu','gpu'):
            path=output/f'{job}_{backend}.json'
            subprocess.run([sys.executable,'-u','-m','experiments.model_bank_capacity',str(mixed),
                str(ROOT/'experiments/a100_config.json'),job,backend,str(len(rows)),str(path)],
                cwd=ROOT,check=True,timeout=1800)
            workers.append(dict(json.loads(path.read_text()),job=job,artifact=path.name))
    # Previously viewed test is a frozen exploratory audit, never an allocation input.
    quality=locked_json(config['allocation_quality_source'],config['allocation_quality_sha256'])
    qmap={}
    for row in quality:
        lookup=(row['task_id'],row['sha256'])
        old=qmap.setdefault(lookup,row)
        if old['quality']!=row['quality']:
            raise RuntimeError('same saved model has conflicting test quality')
    results=[]
    for decision in decisions:
        test=[qmap[(r['task_id'],r['sha256'])] if r['sha256'] is not None else None
              for r in decision['selected']]
        scored=dict(decision,test_served=sum(q is not None and q['test_quality_pass'] for q in test),
                    all_task_quality_pass=all(q is not None and q['test_quality_pass'] for q in test),
                    test_details=test)
        scored['execution_results']=[]
        for backend in ('cpu','gpu'):
            worker=next((w for w in workers if w['job']==decision['job'] and w['backend']==backend),None)
            if worker is None:
                scored['execution_results'].append(dict(backend=backend,qualified_served=0,all_tasks_pass=False))
                continue
            ram_peak=max(m['rss_peak_bytes'] for m in worker['metrics'])
            gpu_peak=max(m['gpu_process_bytes'] for m in worker['metrics']) if backend=='gpu' else 0
            memory_pass=ram_peak<=config['allocation_ram_mib']*1024**2 and gpu_peak<=config['allocation_gpu_mib']*1024**2
            qualified=0
            for row,q in zip(decision['selected'],test):
                if q is None: continue
                timing=max(m['per_task'][row['task_id']]['p95_ms'] for m in worker['metrics'])
                qualified+=bool(memory_pass and q['test_quality_pass'] and timing<=config['allocation_slo_ms'])
            scored['execution_results'].append(dict(backend=backend,ram_peak_bytes=ram_peak,
                gpu_peak_bytes=gpu_peak,memory_pass=memory_pass,qualified_served=qualified,
                all_tasks_pass=qualified==len(decision['selected'])))
        results.append(scored)
        emit('allocation_quality',**{k:v for k,v in scored.items() if k not in ('test_details','selected')})
    report.update(status='complete',results=results,workers=workers,scientific_fits=0,tree_fits=0,
        threshold_fits=0,test_quality_joined_after_seal=True,test_is_reused_exploratory=True,
        allocation_seal_sha256=sha256(output/'allocation_seal.json'))
    write_report(output/'report.json',report)
    emit('allocation_complete',decisions=len(decisions),unique_banks=len(choices),workers=len(workers),
         scientific_fits=0,summary=[dict(policy=p,fraction=f,development_served=sum(r['development_served']
             for r in results if r['policy']==p and r['fraction']==f),
             test_served=sum(r['test_served'] for r in results if r['policy']==p and r['fraction']==f))
             for f in config['allocation_fractions'] for p in ('equal','independent','joint')])
