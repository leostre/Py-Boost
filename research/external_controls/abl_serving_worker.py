"""One orx-owned native-runtime serving measurement, without tree fitting."""
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

from budget_quality_worker import digest,write
from abl_serving_core import model_key,response_cpu
from experiments.uniform_threshold_core import ThresholdModel
from experiments.inference_gpu_guard import gpu_probe,sample_gpu


def rss():
    info=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    return {name:int(info[name].split()[0])*1024 for name in ('VmRSS','VmHWM')}


def features(job):
    record=job['record'];entry=record['dataset'];case=record['case']
    if digest(entry['path'])!=entry['sha256'] or digest(case['split_file'])!=case['split_sha256']:
        raise RuntimeError('immutable serving data changed')
    with np.load(case['split_file'],allow_pickle=False) as f:ids=f['test'].copy()
    with np.load(entry['path'],allow_pickle=False) as f:return np.ascontiguousarray(f['features'][ids],dtype=np.float32)


class ServingModel:
    def __init__(self,entry,probe):
        import cupy as cp
        from experiments.traversal_leaf_gpu import TraversalLeafGpu
        if digest(entry['source']['artifact'])!=entry['source']['sha256']:raise RuntimeError('serving model changed')
        model=ThresholdModel.load(entry['source']['artifact']);self.gpu=TraversalLeafGpu(model,'groups')
        self.gpu._decision_shifts=cp.asarray(entry['control']['shifts'],dtype=cp.float32)
        self.temperature=cp.array(entry['control']['temperature'],dtype=cp.float32)
        self.model_bytes=self.gpu.resident_bytes+self.temperature.nbytes
        if self.model_bytes!=entry['cost_bytes']:raise RuntimeError('GPU model ledger differs from selection price')
        self.resources=self.gpu.kernel_resources
        raw=model.prepare_inference().raw(probe)
        expected=response_cpu(raw,float(self.temperature.get()),np.asarray(entry['control']['shifts']))
        actual=self.response(cp.asarray(probe));cp.cuda.get_current_stream().synchronize()
        np.testing.assert_allclose(actual[0].get(),expected[0],atol=2e-5,rtol=2e-5)
        np.testing.assert_array_equal(actual[1].get(),expected[1])
        self.parity_error=float(np.max(np.abs(actual[0].get()-expected[0])))
        self.temperature_value=entry['control']['temperature'];del actual,expected,model
        gc.collect()

    def response(self,x):
        return self.finish(self.gpu.raw_device(x))

    def finish(self,z):
        import cupy as cp
        cp.divide(z,self.temperature,out=z)
        labels=(z-self.gpu._decision_shifts).argmax(axis=1)
        cp.subtract(z,z.max(axis=1,keepdims=True),out=z)
        cp.exp(z,out=z);z/=z.sum(axis=1,keepdims=True)
        cp.clip(z,1e-6,1-1e-6,out=z);z/=z.sum(axis=1,keepdims=True)
        return z,labels

    def host(self,x):
        import cupy as cp
        result=self.response(cp.asarray(x));return tuple(a.get() for a in result)


def timing(job,output):
    import cupy as cp
    x=features(job);models={};unique={};setup={}
    for role,entry in job['models'].items():
        if entry is None:continue
        key=model_key(entry);started=time.perf_counter()
        if key not in unique:unique[key]=ServingModel(entry,x[:1024])
        models[role]=unique[key]
        setup[role]=dict(seconds=time.perf_counter()-started,model_bytes=models[role].model_bytes,
            probability_parity_error=models[role].parity_error,resources=models[role].resources)
    result=dict(job=job,setup=setup,samples=[],profiles=[],status='in_progress',new_tree_fits=0)
    rng=np.random.default_rng(20261026+job['plan_index'])
    for batch in (1,32,256,1024):
        host=x[:min(batch,len(x))].copy();device=cp.asarray(host)
        for mode in ('device','host'):
            for role,model in models.items():
                for _ in range(2):
                    out=model.response(device) if mode=='device' else model.host(host)
                    cp.cuda.get_current_stream().synchronize();del out
            for block in range(3):
                for request in range(32):
                    order=rng.permutation(list(models)).tolist()
                    for role in order:
                        cp.cuda.get_current_stream().synchronize();start=time.perf_counter_ns()
                        out=models[role].response(device) if mode=='device' else models[role].host(host)
                        cp.cuda.get_current_stream().synchronize();elapsed=(time.perf_counter_ns()-start)/1e6
                        result['samples'].append(dict(role=role,batch=len(host),mode=mode,block=block,
                            request=request,order=order,milliseconds=elapsed));del out
                    # Durable per request, outside the measured interval.
                    write(output,result)
        for block in range(3):
            for role in rng.permutation(list(models)).tolist():
                events=[cp.cuda.Event() for _ in range(3)]
                cp.cuda.get_current_stream().synchronize();events[0].record()
                z=models[role].gpu.raw_device(device);events[1].record()
                out=models[role].finish(z);events[2].record();events[2].synchronize()
                result['profiles'].append(dict(role=role,batch=len(host),block=block,
                    fused_traverse_gather_decode_event_ms=float(cp.cuda.get_elapsed_time(events[0],events[1])),
                    calibration_probabilities_decisions_event_ms=float(cp.cuda.get_elapsed_time(events[1],events[2])),
                    scope='instrumented stream intervals, including dispatch waits; internal fused stages not separately identifiable'))
                del out,z
        write(output,result)
        del device;cp.get_default_memory_pool().free_all_blocks()
        ready,reason,_=gpu_probe('shared',256)
        if not ready:raise RuntimeError(reason)
    result.update(status='complete',rss=rss());write(output,result)


def memory(job,output):
    import cupy as cp
    x=features(job)[:job['batch']].copy();pool=cp.cuda.MemoryPool();peak=dict(used=0,reserved=0)
    def allocate(size):
        pointer=pool.malloc(size)
        peak['used']=max(peak['used'],pool.used_bytes());peak['reserved']=max(peak['reserved'],pool.total_bytes())
        if peak['used']>16*1024**3:raise MemoryError('owned inference pool exceeded 16 GiB')
        return pointer
    before_rss=rss();before_gpu=sample_gpu()
    with cp.cuda.using_allocator(allocate):
        model=ServingModel(job['model'],x[:17]);cp.cuda.get_current_stream().synchronize()
        preparation_peak=dict(peak)
        pool.free_all_blocks();model_pool_used=pool.used_bytes();records=[]
        for mode in ('host','device'):
            device=cp.asarray(x) if mode=='device' else None
            for state in ('first_request_after_validated_preparation','warm'):
                peak['used']=pool.used_bytes();peak['reserved']=pool.total_bytes()
                out=model.response(device) if mode=='device' else model.host(x)
                cp.cuda.get_current_stream().synchronize()
                records.append(dict(mode=mode,state=state,pool_peak_used_bytes=peak['used'],
                    pool_peak_reserved_bytes=peak['reserved'],rss=rss(),gpu_sample=sample_gpu().__dict__))
                del out
            del device;pool.free_all_blocks()
        value=dict(status='complete',job=job,model_bytes=model.model_bytes,
            model_pool_used_bytes=model_pool_used,preparation_including_parity_peak=preparation_peak,
            records=records,rss_before=before_rss,
            gpu_before=before_gpu.__dict__,parity_error=model.parity_error,
            memory_scope='owned pool peak plus sampled whole-device/RSS observations; not isolated GPU-process peak',
            new_tree_fits=0)
        write(output,value)


def main():
    job=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);ready,reason,samples=gpu_probe('shared',8192)
    if not ready:raise RuntimeError(reason)
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=16*1024**3)
    if job['mode']=='timing':timing(job,output)
    elif job['mode']=='memory':memory(job,output)
    else:raise ValueError('unknown measurement mode')


if __name__=='__main__':main()
