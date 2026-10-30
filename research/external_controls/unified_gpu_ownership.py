"""Bounded own-allocation fingerprint when NVML and container PID namespaces differ."""
import gc,time


def probe_candidate(events,tolerance=8*1024**2):
    candidates=None
    for e in events:
        possible={pid for pid in set(e['before'])&set(e['during'])&set(e['after'])
            if abs(e['during'][pid]-e['before'][pid]-e['bytes'])<=tolerance
            and abs(e['after'][pid]-e['before'][pid])<=tolerance}
        candidates=possible if candidates is None else candidates&possible
    return next(iter(candidates)) if candidates is not None and len(candidates)==1 and len(events)==2 else None


def controlled_owner(snapshot,namespace_pids,allocate,cleanup,synchronize):
    baseline=snapshot();direct=set(baseline)&set(namespace_pids)
    if len(direct)==1:return dict(pid=next(iter(direct)),verified=True,namespace_verified=True,method='namespace_pid_match',probes=[])
    events=[]
    for size in (64*1024**2,96*1024**2):
        marker=None
        try:
            cleanup();synchronize();before=snapshot();marker=allocate(size);synchronize()
            during=snapshot()
            for _ in range(5):
                if any(abs(during.get(pid,0)-value-size)<=8*1024**2 for pid,value in before.items()):break
                time.sleep(.1);during=snapshot()
            marked={pid for pid,value in before.items() if abs(during.get(pid,0)-value-size)<=8*1024**2}
            marker=None;gc.collect();cleanup();synchronize();after=snapshot()
            for _ in range(5):
                if any(abs(after.get(pid,0)-before[pid])<=8*1024**2 for pid in marked):break
                time.sleep(.1);after=snapshot()
            events.append(dict(bytes=size,before=before,during=during,after=after))
        except Exception as exc:
            marker=None;gc.collect();cleanup()
            return dict(pid=None,verified=False,namespace_verified=False,method='optional_probe_failed',reason=type(exc).__name__,probes=events)
    pid=probe_candidate(events)
    return dict(pid=pid,verified=pid is not None,namespace_verified=False,
        method='two_controlled_own_allocation_fingerprints' if pid is not None else 'ambiguous_controlled_fingerprint',
        probes=events,maximum_temporary_bytes=96*1024**2,
        interpretation='experimental process attribution; not an OS namespace match; no foreign allocation or process is changed')


def cupy_owner(snapshot,namespace_pids):
    import cupy as cp
    def allocate(size):
        memory=cp.cuda.memory.Memory(size);cp.cuda.runtime.memset(memory.ptr,0,size);return memory
    return controlled_owner(snapshot,namespace_pids,allocate,lambda:None,lambda:cp.cuda.get_current_stream().synchronize())


def torch_owner(snapshot,namespace_pids):
    import torch
    def allocate(size):return torch.empty(size,dtype=torch.uint8,device='cuda').zero_()
    return controlled_owner(snapshot,namespace_pids,allocate,torch.cuda.empty_cache,torch.cuda.synchronize)
