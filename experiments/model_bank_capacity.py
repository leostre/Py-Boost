"""Frozen-bank end-to-end closed-loop traffic; real host and GPU costs."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.a100_runner import ROOT, emit, sha256, write_report
from experiments.model_bank_core import (task_order, traffic_schedule, executor_name,
    owned_bytes, gpu_items, cpu_items, latency_summary)
from experiments.model_bank_study import locked_json, rss_bytes, RssPeak, gpu_processes


def capacity_worker(manifest_path, config_path, family, backend, k_text, output_path):
    from experiments.uniform_threshold_core import ThresholdModel
    config = json.loads(Path(config_path).read_text())
    bank = json.loads(Path(manifest_path).read_text())
    records = task_order([r for r in bank['records'] if r['family'] == family])[:int(k_text)]
    cp, own_pid = None, None
    if backend == 'gpu':
        from experiments.inference_gpu_guard import gpu_probe
        ready, reason, _ = gpu_probe('shared', 2048)
        if not ready: raise RuntimeError(reason)
        before = gpu_processes()
        import cupy as cp
        cp.zeros(1).sum().item()
        added = set(gpu_processes())-set(before)
        if len(added) != 1: raise RuntimeError('ambiguous own GPU host PID')
        own_pid = added.pop()
    models, inputs, feature_cache, residents = [], {}, {}, []
    before_rss = rss_bytes()
    for row in records:
        if sha256(row['path']) != row['sha256']: raise RuntimeError('frozen model changed')
        source = ThresholdModel.load(row['path'])
        if cp is None:
            runtime = source.prepare_compiled_inference()
            items = cpu_items(runtime.snapshot)
        else:
            from experiments.traversal_leaf_gpu import TraversalLeafGpu
            from experiments.unified_leaf_gpu import UnifiedLeafGpu
            runtime = (TraversalLeafGpu(source,'groups') if executor_name(len(source.base.bias))=='groups'
                       else UnifiedLeafGpu(source))
            items = gpu_items(runtime)
        residents.extend(items)
        models.append(runtime)
        name = row['dataset']
        if name not in feature_cache:
            if sha256(row['data']['path']) != row['data']['sha256']: raise RuntimeError('data changed')
            with np.load(row['data']['path'], allow_pickle=False) as f:
                feature_cache[name] = np.ascontiguousarray(f['features'][:1024],dtype=np.float32)
            for batch in config['bank_batches']:
                inputs[name,batch] = feature_cache[name][np.arange(batch)%len(feature_cache[name])].copy()
        probe = feature_cache[name][:5].copy()
        np.testing.assert_allclose(runtime.predict(probe),source.predict(probe),atol=2e-12,rtol=0)
        del source
    exact = owned_bytes(residents)
    gc.collect()
    if cp is not None:
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    resident_rss = rss_bytes()
    local = threading.local()
    pool_peak = cp.get_default_memory_pool().used_bytes() if cp is not None else 0
    pool_lock = threading.Lock()
    if cp is not None:
        allocator = cp.cuda.get_allocator()
        def tracked(size):
            nonlocal pool_peak
            pointer = allocator(size)
            with pool_lock:
                pool_peak = max(pool_peak,cp.get_default_memory_pool().used_bytes())
            return pointer
        cp.cuda.set_allocator(tracked)
    def request(index, batch, submitted):
        runtime = models[index]
        x = inputs[records[index]['dataset'],batch]
        if cp is None:
            result = runtime.predict(x)
        else:
            with cp.cuda.Device(0):
                if not hasattr(local,'stream'):
                    local.stream = cp.cuda.Stream(non_blocking=True)
                with local.stream:
                    # Real host->device and device->host transfers are included.
                    result = runtime.predict(x)
        if not np.isfinite(result).all(): raise RuntimeError('nonfinite serving result')
        return index, (time.perf_counter_ns()-submitted)/1e6
    metrics = []
    with threadpool_limits(limits=1):
        for batch in config['bank_batches']:
            for jobs in config['bank_concurrency']:
                with ThreadPoolExecutor(max_workers=jobs) as executor:
                    # Warm every model, including its actual transfer path.
                    for i in range(len(models)):
                        executor.submit(request,i,batch,time.perf_counter_ns()).result()
                    for traffic in ('uniform','skew'):
                        schedule = traffic_schedule(len(models),config['bank_requests'],traffic,20261021)
                        samples, pending, next_request = [], {}, 0
                        with RssPeak() as peak:
                            started = time.perf_counter_ns()
                            while next_request < len(schedule) or pending:
                                while len(pending) < jobs and next_request < len(schedule):
                                    index = schedule[next_request]
                                    pending[executor.submit(request,index,batch,time.perf_counter_ns())] = index
                                    next_request += 1
                                done, _ = wait(pending,return_when=FIRST_COMPLETED)
                                for future in done:
                                    samples.append(future.result())
                                    del pending[future]
                            seconds = (time.perf_counter_ns()-started)/1e9
                        summary = latency_summary([s[1] for s in samples])
                        per_task = {r['task_id']: latency_summary([s[1] for s in samples if s[0]==i])
                                    for i,r in enumerate(records)}
                        metric = dict(family=family,backend=backend,k=len(models),batch=batch,jobs=jobs,
                            traffic=traffic,**summary,wall_seconds=seconds,
                            wall_qps=len(samples)/seconds,objects_per_second=len(samples)*batch/seconds,
                            per_task=per_task,rss_peak_bytes=peak.peak,
                            gpu_process_bytes=gpu_processes().get(own_pid) if cp is not None else None,
                            pool_used_peak_bytes=pool_peak if cp is not None else None,
                            pool_reserved_bytes=cp.get_default_memory_pool().total_bytes() if cp is not None else None)
                        metrics.append(metric)
                        emit('bank_capacity_measurement',**{k:v for k,v in metric.items() if k!='per_task'})
    result = dict(family=family,backend=backend,k=len(models),metrics=metrics,
        records=[dict(task_id=r['task_id'],dataset=r['dataset'],seed=r['seed'],family=family,
                      sha256=r['sha256'],development_pass=r['development_pass']) for r in records],
        resident_bytes=exact,resident_rss_bytes=resident_rss,before_rss_bytes=before_rss,
        lifetime_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_host_pid=own_pid,scientific_fits=0,test_labels_loaded=False,
        input_bytes=sum(x.nbytes for x in feature_cache.values())+sum(x.nbytes for x in inputs.values()),
        shared_gpu_timing=cp is not None,latency_mode='closed-loop bounded J, transfer-inclusive, ready resident models')
    write_report(Path(output_path),result)
    emit('bank_capacity_worker_complete',family=family,backend=backend,k=len(models),
         resident_bytes=exact,measurements=len(metrics),scientific_fits=0)


def frozen_quality(bank, config):
    """After the cohort and traffic are frozen: score old exploratory test only."""
    from experiments.leaf_budget_runtime import expanded_manifest,load_expanded,load_roles
    from experiments.uniform_threshold_core import ThresholdModel
    from experiments.history_basis_study import raw
    from experiments.output_calibration_core import calibrated_quality
    from experiments.shared_leaf_basis import quality_guard
    expanded = expanded_manifest(config)
    qualities = []
    with threadpool_limits(limits=1):
        for name in config['datasets']:
            x,y = load_expanded(name,expanded)
            rows = [r for r in bank['records'] if r['dataset']==name]
            for seed in config['bank_seeds_by_dataset'][name]:
                case = next(c for c in expanded['cases'] if c['dataset']==name and c['seed']==seed)
                roles = load_roles(case,len(x))
                current = [r for r in rows if r['seed']==seed]
                actual = {}
                for row in current:
                    if sha256(row['path']) != row['sha256']: raise RuntimeError('test artifact changed')
                    model = ThresholdModel.load(row['path'])
                    actual[row['family']] = calibrated_quality(y[roles['test']],raw(model,x[roles['test']]),
                                                               row['task'],model.decision_shifts)
                for row in current:
                    quality = actual[row['family']]
                    qualities.append(dict(task_id=row['task_id'],family=row['family'],sha256=row['sha256'],
                        development_pass=row['development_pass'],quality=quality,reference=actual['full'],
                        test_quality_pass=quality_guard(quality,actual['full'])))
    return qualities


def model_bank_capacity(config,report,output):
    source = Path(config['bank_source_manifest'])
    bank = locked_json(source,config['bank_source_manifest_sha256'])
    # Freeze the exact source and workload before any test targets are read.
    write_report(output/'traffic_seal.json',dict(bank_sha256=sha256(source),config=config,
                 test_labels_read=False,scientific_fits=0))
    workers = []
    for backend in ('cpu','gpu'):
        for family in ('full','rank_residual','joint_static','native_prefix'):
            for k in config['bank_k']:
                p = output/f'capacity_{backend}_{family}_{k}.json'
                subprocess.run([sys.executable,'-u','-m','experiments.model_bank_capacity',str(source),
                    str(ROOT/'experiments/a100_config.json'),family,backend,str(k),str(p)],
                    cwd=ROOT,check=True,timeout=3600)
                workers.append(json.loads(p.read_text()))
    quality = frozen_quality(bank,config)
    write_report(output/'frozen_quality.json',quality)
    qmap = {(q['task_id'],q['family']):q for q in quality}
    constraints = []
    for w in workers:
        for m in w['metrics']:
            for ram in config['bank_ram_budgets_mib']:
                for gpu in (config['bank_gpu_budgets_mib'] if w['backend']=='gpu' else [0]):
                    memory_pass = m['rss_peak_bytes'] <= ram*1024**2 and (
                        w['backend']!='gpu' or m['gpu_process_bytes'] <= gpu*1024**2)
                    for slo in config['bank_slo_ms']:
                        statuses = []
                        for record in w['records']:
                            q = qmap[record['task_id'],w['family']]
                            latency_pass = m['per_task'][record['task_id']]['p95_ms'] <= slo
                            statuses.append(dict(task_id=record['task_id'],development_pass=q['development_pass'],
                                test_quality_pass=q['test_quality_pass'],latency_pass=latency_pass,
                                pass_all=bool(memory_pass and q['development_pass'] and q['test_quality_pass'] and latency_pass)))
                        constraints.append(dict(backend=w['backend'],family=w['family'],k=w['k'],
                            batch=m['batch'],jobs=m['jobs'],traffic=m['traffic'],ram_budget_mib=ram,
                            gpu_budget_mib=gpu,slo_ms=slo,memory_pass=memory_pass,
                            tasks_passing=sum(s['pass_all'] for s in statuses),
                            all_tasks_pass=all(s['pass_all'] for s in statuses),tasks=statuses))
    report.update(status='complete',workers=workers,quality=quality,constraints=constraints,
        tree_fits=0,threshold_fits=0,test_evaluated=True,test_is_reused_exploratory=True,
        scientific_fits=0,traffic_seal_sha256=sha256(output/'traffic_seal.json'))
    write_report(output/'report.json',report)
    emit('bank_capacity_complete',workers=len(workers),measurements=sum(len(w['metrics']) for w in workers),
         requests=sum(m['requests'] for w in workers for m in w['metrics']),scientific_fits=0,
         test_quality_pass={f:sum(q['test_quality_pass'] for q in quality if q['family']==f)
                            for f in ('full','rank_residual','joint_static','native_prefix')},
         all_current_hardware_fit=all(max(m['rss_peak_bytes'] for m in w['metrics'])<1024**3
            and (w['backend']!='gpu' or max(m['gpu_process_bytes'] for m in w['metrics'])<1024**3) for w in workers))


if __name__=='__main__':
    capacity_worker(*sys.argv[1:])
