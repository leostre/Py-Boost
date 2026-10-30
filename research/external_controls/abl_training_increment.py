"""Pinned continuation adapter with explicit10/50 minimum-leaf setting.
Original native_source remains unchanged; orx-owned callers only.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits


def fit_training_increment(request):
    from experiments.a100_runner import build_model,write_report,sha256
    from experiments.continuation_leaf_boosting import ContinuationLeafBoosting
    from experiments.history_basis_study import raw
    from experiments.shared_leaf_basis import probabilities
    from experiments.inference_gpu_guard import gpu_probe
    import cupy as cp
    ready,reason,probe = gpu_probe('shared',request.get('minimum_free_mib',2048))
    if not ready:
        raise RuntimeError(reason)
    started_ns = time.time_ns()
    with np.load(request['data_path'],allow_pickle=False) as f:
        x,y,xv,yv = [f[k].copy() for k in ('train_x','train_y','validation_x','validation_y')]
    with np.load(request['initial_path'],allow_pickle=False) as f:
        initial,validation = f['train'].copy(),f['validation'].copy()
    with threadpool_limits(limits=1):
        config = request['config']
        native = build_model(request.get('method','full'),request['task'],initial.shape[1],config,request['seed'],
            config['max_depth'],config['lambda_l2'],trees=request['trees'])
        arguments = dict(native.params)
        arguments.update(lr=request['lr'],min_data_in_leaf=request['minimum_leaf'],subsample=request.get('subsample',.8),
                         colsample=request.get('colsample',.8))
        model = ContinuationLeafBoosting(**arguments,initial_train=initial,
            initial_validation=[validation],task=request['task'],
            pool_limit_bytes=request.get('own_pool_limit_bytes'))
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        fit_started_ns = time.time_ns()
        model.fit(x,y,eval_sets=[dict(X=xv,y=yv)])
        cp.cuda.get_current_stream().synchronize()
        fit_finished_ns = time.time_ns()
        seconds = time.perf_counter()-started
        increment = model.export_increment()
        visits=increment.base.leaf_indices(x)
        support=np.bincount(visits.ravel(),minlength=len(increment.base.coefficients))
        observed_minimum_leaf=int(support[support>0].min())
        if observed_minimum_leaf<request['minimum_leaf']:
            raise RuntimeError('continuation increment violates requested leaf support')
        del visits,support
        if increment.base.n_trees != request['trees']:
            raise RuntimeError('continuation tree count changed')
        train_error = float(np.max(np.abs(probabilities(initial.astype(np.float32)+raw(increment,x),request['task'])-
            probabilities(model.train_final_raw,request['task']))))
        validation_error = float(np.max(np.abs(probabilities(validation+raw(increment,xv),request['task'])-
            probabilities(model.validation_final_raw[0],request['task']))))
        if max(train_error,validation_error) > 2e-5:
            raise RuntimeError('continuation export mismatch')
        path = Path(request['output'])
        increment.save(path)
        record = dict(seed=request['seed'],trees=request['trees'],lr=request['lr'],pid=os.getpid(),
            minimum_leaf=request['minimum_leaf'],observed_minimum_leaf_support=observed_minimum_leaf,
            seconds=seconds,training_pool_peak_bytes=model.training_pool_peak,
            train_replay_error=train_error,validation_replay_error=validation_error,
            data_sha256=sha256(request['data_path']),initial_sha256=sha256(request['initial_path']),
            artifact=str(path),sha256=sha256(path),parameters=increment.parameter_count,
            payload_bytes=increment.payload_bytes,started_ns=started_ns,finished_ns=time.time_ns(),
            fit_started_ns=fit_started_ns,fit_finished_ns=fit_finished_ns,gpu_probe=probe)
        write_report(path.with_suffix('.json'),record)
    del model,native
    cp.get_default_memory_pool().free_all_blocks()
    return record
