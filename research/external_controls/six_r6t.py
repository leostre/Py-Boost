"""Separate five-task classification cohort; no multilabel/TabPFN mixing."""
from pathlib import Path
import gc,json,sys,time,subprocess
import numpy as np
from budget_quality_worker import locked,write,digest,event


def boost(cfg,out):
    import cupy as cp
    from abl_quality_worker import check_native_environment
    from saved_serving_core import load_saved_serving as load_protected
    from packed_serving_core import prepare_snapshot
    from packed_serving_gpu import PackedGpu
    from six_quant_core import prepare_quant
    from six_quant_gpu import QuantPackedGpu
    from six_r5 import quant_raw
    from residual_gate_core import sufficient,quality
    from tfm_serving_core import request_plan
    env=check_native_environment(cfg);cp.get_default_memory_pool().set_limit(size=12*1024**3)
    seal=locked(cfg['six']['five_task_seal']['path'],cfg['six']['five_task_seal']['sha256']);tasks=cfg['quality_cases'];rows=[]
    records={c['case']:c['record'] for c in seal['source_cases']};inputs={};test={};models={}
    native=locked(cfg['bank_sources']['native']['path'],cfg['bank_sources']['native']['sha256'])
    for cid in tasks:
        if cid not in records:
            old=next(r for r in native['records'] if r['case_id']==cid and r['family']=='full')
            records[cid]=dict(case=old['case'],dataset=old['dataset'])
    for cid in tasks:
        record=records[cid];cc,dd=record['case'],record['dataset'];assert digest(cc['split_file'])==cc['split_sha256'] and digest(dd['path'])==dd['sha256']
        with np.load(cc['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
        with np.load(dd['path'],allow_pickle=False) as f:
            prefix=f['features'][:1024].astype(np.float32);inputs[cid]=prefix.copy();test[cid]=(f['features'][ids].astype(np.float32),f['target'][ids].copy())
        for family in ('full','small','posthoc'):
            ref=next((e for e in seal['inventory'] if e['case']==cid and e['bank']==family),None)
            if ref is None:
                # Binary rank C-1 is already one: no low-rank saving claim.
                old=next(r for r in native['records'] if r['case_id']==cid and r['family']=='full')
                path=old['selected_artifact'];sha=old['selected_sha256']
                if family=='small':path=old['artifacts']['d3-t40']['artifact'];sha=old['artifacts']['d3-t40']['sha256']
                model=load_protected(path)
                ref=dict(artifact=path,sha256=sha,control=dict(temperature=1.,shifts=model.decision_shifts.tolist()),binary_control=True,posthoc_is_identity=family=='posthoc')
            assert digest(ref['artifact'])==ref['sha256'];models[cid,family]=(load_protected(ref['artifact']),ref)
    for family in ('full','small','posthoc'):
        for codec in ('fp32','int8'):
            devices=[];qualities=[]
            for cid in tasks:
                model,ref=models[cid,family];control=ref['control'];s=prepare_snapshot(model,control['shifts']) if codec=='fp32' else prepare_quant(model,control['shifts'],'int8')
                g=next(g for g in (4,2,1) if 8*(s.outputs+g*s.max_width+g*s.outputs*int(s.has_residual)+32*g)<=49152)
                d=PackedGpu(model,control['temperature'],control['shifts'],g) if codec=='fp32' else QuantPackedGpu(s,control['temperature'],g);devices.append(d)
                # Fixed previous controls, no new calibration or model selection.
                x,y=test[cid];raw=np.empty((len(x),len(model.bias)))
                for start in range(0,len(x),1024):
                    dx=cp.asarray(np.ascontiguousarray(x[start:start+1024]));buf=d.buffers(len(dx));d.launch_into(dx,*buf,raw=True);raw[start:start+len(dx)]=buf[0].get();del dx,buf
                q=quality(y,*sufficient(y,raw,'multiclass',control),'multiclass',len(model.bias));qualities.append(dict(case=cid,quality=q))
            metrics=[]
            for batch in (1,32):
                for traffic in ('uniform','skew'):
                    schedule=request_plan(len(tasks),32,traffic)
                    for i,d in enumerate(devices):d.request(np.ascontiguousarray(inputs[tasks[i]][np.arange(batch)%len(inputs[tasks[i]])]))
                    cp.cuda.get_current_stream().synchronize();samples=[];per_task=[]
                    for iteration,index in enumerate(schedule):
                        x=inputs[tasks[index]];x=np.ascontiguousarray(x[np.arange(batch)%len(x)])
                        begin=time.perf_counter_ns();devices[index].request(x);elapsed=(time.perf_counter_ns()-begin)/1e6
                        samples.append(elapsed);per_task.append(tasks[index]);write(out/'boost-progress.json',dict(family=family,codec=codec,batch=batch,traffic=traffic,request_ms=samples,tasks=per_task))
                    metrics.append(dict(batch=batch,traffic=traffic,request_ms=samples,task_schedule=per_task,p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95))))
            gc.collect();cp.get_default_memory_pool().free_all_blocks();baseline=cp.get_default_memory_pool().used_bytes();peak=[baseline];allocator=cp.cuda.get_allocator()
            def tracked(size):
                ptr=allocator(size);peak[0]=max(peak[0],cp.get_default_memory_pool().used_bytes());return ptr
            with cp.cuda.using_allocator(tracked):
                for i,d in enumerate(devices):d.request(np.ascontiguousarray(inputs[tasks[i]][np.arange(32)%len(inputs[tasks[i]])]))
            rows.append(dict(family=family,codec=codec,quality=qualities,metrics=metrics,model_arrays=sum(d.resident_device_array_bytes for d in devices),own_pool_peak=peak[0]))
            write(out/'boost-report.json',dict(status='running',rows=rows));event('R6T_boost',family=family,codec=codec)
            del devices,d,tracked;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(out/'boost-report.json',dict(status='R6T_boost_complete',rows=rows,new_trees=0,environment=env,
          scope='previous five matched classification tasks; fixed old controls; actual packed full/small/posthoc versus INT8; GPU pool excludes context'))


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,__file__,str(Path(sys.argv[1])),str(out),'--boost'],check=True,timeout=3600)
    # Parent has no CUDA context; the boost child exited before Torch starts.
    external=Path(cfg['environment_root'])/'venv/bin/python';child=Path(__file__).parent/'tfm_bank_worker.py'
    tfm_cfg=dict(cfg,bank_batches=[1,32],bank_requests=32);p=out/'tfm-config.json';write(p,tfm_cfg)
    output=out/'tabpfn-cached-auto.json'
    subprocess.run([str(external),str(child),str(p),str(output),'--mode','fit_with_cache','--family','tabpfn_auto'],check=True,timeout=7200)
    report=dict(status='R6T_complete',boost=dict(path=str(out/'boost-report.json'),sha256=digest(out/'boost-report.json')),
                tabpfn=dict(path=str(output),sha256=digest(output)),new_trees=0,nn_weight_updates=0,
                caveats=['five tasks only, not the12multilabel tasks','old quality controls retained; no new5percent research gate','serial J1 transfer-inclusive traces,32samples not SLA guarantee','sequential native and external frozen environments; no speed attribution to environment changes'])
    write(out/'report.json',report);event('R6T_complete',sha256=digest(out/'report.json'))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2])
    with threadpool_limits(limits=2):
        if '--boost' in sys.argv:boost(cfg,out)
        else:run(cfg,out)
