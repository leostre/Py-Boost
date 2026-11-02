"""Exact-vector nvForest control paired with packed in one pinned environment."""
from pathlib import Path
from dataclasses import asdict
import gc,json,sys,time,subprocess
import numpy as np
from nvforest_probe import cases,export_full,sha


def run(cfg):
    import cupy as cp,nvforest,rmm.statistics as rs
    rs.enable_statistics();pool=cp.get_default_memory_pool();pool.set_limit(size=12*1024**3)
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=True);refs=[];rng=np.random.default_rng(202610307)
    # cases() installs the pinned pure model source paths before packed imports.
    for entry,model,x in cases(cfg):
        from nvf_packed_gpu import PackedGpu
        from nvf_packed_core import prepare_snapshot
        c=len(model.bias);control=entry['model']['assessment']['refit_control'];shifts=np.asarray(control['shifts'],np.float32);temperature=control['temperature']
        snap=prepare_snapshot(model,shifts);groups=next(g for g in (4,2,1) if 8*(c+g*snap.max_width+32*g)<=49152)
        packed=PackedGpu(model,temperature,shifts,groups);original=export_full(model);forests={};resident={};tuning=[]
        stream=cp.cuda.get_current_stream();ds=cp.asarray(shifts)
        from nvf_fused_controls import OutputControls
        finish=OutputControls(ds,temperature)
        for precision in ('single','double'):
            for layout in ('depth_first','breadth_first','layered'):
                for chunk in (1,32):
                    key=f'{precision}/{layout}/{chunk}';before=rs.get_statistics().current_bytes
                    handle=nvforest.Handle(stream=stream.ptr)
                    forest=nvforest.load_from_treelite_model(original,device='gpu',precision=precision,layout=layout,default_chunk_size=chunk,handle=handle)
                    probe=forest.predict_proba(cp.asarray(x[:1]));stream.synchronize();del probe;gc.collect()
                    resident[key]=rs.get_statistics().current_bytes-before;forests[key]=(forest,handle)
        measured=[]
        for batch in (1,32,256,1024,4096):
            features=np.ascontiguousarray(x[np.arange(batch)%len(x)]);xx=cp.asarray(features);buf=packed.buffers(batch)
            def gpu_call(name):
                if name=='packed':packed.launch_into(xx,*buf);return buf[0],buf[1]
                return finish(forests[name][0].predict_proba(xx))
            packed.launch_into(xx,*buf,raw=True);canonical=buf[0].get()
            np.testing.assert_allclose(canonical[:13],model.raw(features[:13]),atol=1e-8,rtol=1e-10)
            valid=[];checks={}
            for key in forests:
                raw=cp.asnumpy(forests[key][0].predict_proba(xx)).reshape(canonical.shape)
                finite=bool(np.isfinite(raw).all());error=float(np.max(np.abs(raw-canonical))) if finite else None
                passed=finite and error<5e-5
                checks[key]=dict(passed=passed,raw_error=error,finite=finite)
                if passed:valid.append(key)
            del raw,canonical
            times={key:[] for key in valid}
            for key in valid:
                for _ in range(3):gpu_call(key)
            stream.synchronize()
            for _ in range(16):
                for key in rng.permutation(valid):
                    a,b=cp.cuda.Event(),cp.cuda.Event();a.record();gpu_call(key);b.record();b.synchronize();times[key].append(float(cp.cuda.get_elapsed_time(a,b)))
            choices={precision:(min((k for k in valid if k.startswith(precision+'/')),key=lambda k:(np.median(times[k]),k)) if any(k.startswith(precision+'/') for k in valid) else None) for precision in ('single','double')}
            tuning.append(dict(batch=batch,choices=choices,samples=times,compatibility=checks));names=['packed']+[k for k in choices.values() if k is not None]
            (out/(entry['case']+'-tuning.json')).write_text(json.dumps(dict(tuning=tuning,final_used_for_selection=False),indent=2))
            for name in names:
                for _ in range(4):gpu_call(name)
            stream.synchronize()
            for session in range(3):
                samples={n:dict(gpu_ms=[],request_ms=[]) for n in names}
                for iteration in range(32):
                    for name in rng.permutation(names):
                        a,b=cp.cuda.Event(),cp.cuda.Event();a.record();gpu_call(name);b.record();b.synchronize();samples[name]['gpu_ms'].append(float(cp.cuda.get_elapsed_time(a,b)))
                numerical_failures=[];label_differences=[]
                for iteration in range(32):
                    actuals={}
                    features=np.ascontiguousarray(x[(np.arange(batch)+iteration*17+session*53+409)%len(x)])
                    for name in rng.permutation(names):
                        start=time.perf_counter_ns()
                        if name=='packed':actuals[name]=packed.request(features)
                        else:
                            px=cp.asarray(features);p,l=finish(forests[name][0].predict_proba(px));actuals[name]=(p.get(),l.get());del px,p,l
                        samples[name]['request_ms'].append((time.perf_counter_ns()-start)/1e6)
                    for name in names[1:]:
                        error=float(np.max(np.abs(actuals[name][0]-actuals['packed'][0])))
                        if not np.isfinite(error) or error>2e-5:numerical_failures.append(dict(iteration=iteration,name=name,probability_error=error))
                        different=actuals[name][1]!=actuals['packed'][1]
                        if np.any(different):
                            boundary=1/(1+np.exp(-shifts))
                            ambiguous=np.abs(actuals['packed'][0]-boundary)<1.5e-5
                            bad=int(np.sum(different & ~ambiguous));label_differences.append(dict(iteration=iteration,name=name,total=int(different.sum()),outside_tolerance=bad))
                            if bad:numerical_failures.append(dict(iteration=iteration,name=name,label_differences_outside_tolerance=bad))
                    del actuals
                measured.append(dict(batch=batch,session=session,samples=samples,numerical_failures=numerical_failures,label_differences=label_differences,valid=not numerical_failures))
                (out/(entry['case']+'-progress.json')).write_text(json.dumps(dict(measured=measured),indent=2))
            del xx,buf,gpu_call
        memory=[]
        for batch in (1,32,256,1024,4096):
            row=next(t for t in tuning if t['batch']==batch);features=np.ascontiguousarray(x[np.arange(batch)%len(x)])
            for name in ['packed']+[k for k in row['choices'].values() if k is not None]:
                gc.collect();pool.free_all_blocks();baseline=pool.used_bytes();peak=[baseline];allocator=cp.cuda.get_allocator();rs.push_statistics()
                def tracked(size):
                    ptr=allocator(size);peak[0]=max(peak[0],pool.used_bytes());return ptr
                with cp.cuda.using_allocator(tracked):
                    if name=='packed':packed.request(features)
                    else:
                        px=cp.asarray(features);p,l=finish(forests[name][0].predict_proba(px));p.get();l.get();del px,p,l
                stream.synchronize();stats=rs.pop_statistics();parameters=packed.resident_device_array_bytes if name=='packed' else resident[name]+ds.nbytes
                memory.append(dict(batch=batch,name=name,model_allocated_bytes=int(parameters),request_cupy_peak=peak[0]-baseline,
                    request_rmm_peak=int(stats.peak_bytes),model_plus_request=int(parameters+peak[0]-baseline+stats.peak_bytes),rmm=asdict(stats)))
        result=dict(case=entry['case'],model_sha256=entry['model']['sha256'],tuning=tuning,measured=measured,memory=memory,
                    forest_resident_rmm=resident,new_trees=0,environment_sha256=cfg['environment_sha256'],
                    note='fused CUDA output controls around unmodified nvForest engine; same vector-leaf model, same environment, same probability+label request; single uses FP32 accumulation, double canonical precision; RMM+CuPy tracked, excludes driver/context')
        path=out/(entry['case']+'-report.json');path.write_text(json.dumps(result,indent=2));refs.append(dict(case=entry['case'],path=str(path),sha256=sha(path)))
        (out/'progress.json').write_text(json.dumps(dict(rows=refs),indent=2));print(json.dumps(dict(event='N2F_model',case=entry['case'],choices=[t['choices'] for t in tuning])),flush=True)
        del packed,forests,forest,handle,original,model,ds,finish;gc.collect();pool.free_all_blocks()
    report=dict(status='N2F_complete',rows=refs,new_trees=0,environment_sha256=cfg['environment_sha256'])
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(event='N2F_complete',sha256=sha(out/'report.json'))),flush=True)

if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
