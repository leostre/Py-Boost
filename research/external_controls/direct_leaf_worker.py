"""A3 isolated direct fit and GPU preflight; environments/engine remain pinned."""
from pathlib import Path
from time import perf_counter
import json,sys,gc,os
import numpy as np
from budget_quality_worker import digest,locked,write,event


def fit_direct(request):
    from experiments.a100_runner import build_model
    from experiments.inference_gpu_guard import gpu_probe
    from experiments.shared_leaf_basis import probabilities
    from direct_leaf_gpu import DirectLeafBoosting
    import cupy as cp
    ready,reason,probe=gpu_probe('shared',8192)
    if not ready:raise RuntimeError(reason)
    with np.load(request['data_path'],allow_pickle=False) as f:x,y,xv,yv=[f[k].copy() for k in ('train_x','train_y','validation_x','validation_y')]
    with np.load(request['initial_path'],allow_pickle=False) as f:initial,validation=f['train'].copy(),f['validation'].copy()
    cfg=request['config'];native=build_model('full',request['task'],initial.shape[1],cfg,request['seed'],cfg['max_depth'],cfg['lambda_l2'],trees=request['trees'])
    # The pinned EvalHistory always uses modulo(verbose); zero is unsupported.
    args=dict(native.params);args.update(verbose=200,use_hess=True,gd_steps=1)
    model=DirectLeafBoosting(**args,initial_train=initial,initial_validation=[validation],basis=np.asarray(request['basis'],np.float32),
        task=request['task'],residual_fraction=request['residual_fraction'],class_importance=request.get('class_importance'),
        full_coordinates=request['full_coordinates'],chunk_rows=request.get('chunk_rows',1024),
        pool_limit_bytes=request.get('own_pool_limit_bytes',16*1024**3),cumulative_quota=request.get('cumulative_quota',False))
    cp.cuda.get_current_stream().synchronize();started=perf_counter();model.fit(x,y,eval_sets=[dict(X=xv,y=yv)]);cp.cuda.get_current_stream().synchronize();seconds=perf_counter()-started
    increment=model.export_increment();p=request['task']
    train_error=float(np.max(np.abs(probabilities(initial+increment.prepare_inference().raw(x),p)-probabilities(model.train_final_raw,p))))
    validation_error=float(np.max(np.abs(probabilities(validation+increment.prepare_inference().raw(xv),p)-probabilities(model.validation_final_raw[0],p))))
    assert max(train_error,validation_error)<2e-5,(train_error,validation_error)
    assert increment.base.n_trees==request['trees']
    artifact=Path(request['output']);artifact.parent.mkdir(parents=True,exist_ok=True);increment.save(artifact)
    record=dict(artifact=str(artifact),sha256=digest(artifact),trees=increment.base.n_trees,rank=len(request['basis'][0]),
        outputs=initial.shape[1],full_coordinates=request['full_coordinates'],residual_fraction=request['residual_fraction'],
        seconds=seconds,training_pool_peak_bytes=model.training_pool_peak,trace=model.direct_trace,seed=request['seed'],
        train_replay_error=train_error,validation_replay_error=validation_error,initial_path=request['initial_path'],
        data_sha256=digest(request['data_path']),initial_sha256=digest(request['initial_path']),
        full_dataset_derivatives_retained=False,sample_rank_rank_curvature_materialized=False,
        structure_curvature='diagonal of projected Hessian for split scoring',leaf_curvature='full coupled projected Hessian plus regularization',gpu_probe=probe,pid=os.getpid())
    write(artifact.with_suffix('.json'),record);del model,native;gc.collect();cp.get_default_memory_pool().free_all_blocks();return record


def preflight(cfg,out):
    from abl_quality_worker import check_native_environment
    from direct_leaf_core import leaf_statistics_reference
    from experiments.gradient_basis_core import contrast_matrix
    from direct_leaf_gpu import leaf_statistics_gpu,projected_arrays
    import cupy as cp
    out.mkdir(parents=True,exist_ok=True);spec=cfg['direct_preflight'];assert spec['inputs_ready']
    prerequisite=locked(spec['a2']['path'],spec['a2']['sha256']);assert prerequisite['status']=='adaptive_training_complete'
    environment=check_native_environment(cfg);checks=[];fits=[];paid=0
    rng=np.random.default_rng(29)
    for task in ('multiclass','multilabel'):
        raw=rng.normal(size=(15,4));y=rng.integers(0,4,15) if task=='multiclass' else rng.integers(0,2,(15,4));leaves=np.arange(15)%3
        k=contrast_matrix(4,task);basis=(k@np.linalg.qr(rng.normal(size=(k.shape[1],2)))[0]).astype(np.float64)
        actual=leaf_statistics_gpu(cp.asarray(raw),cp.asarray(y),cp.asarray(leaves),cp.asarray(basis),task,4)
        expected=leaf_statistics_reference(raw,y,leaves,basis,task);errors={key:float(np.max(np.abs(value.get()-expected[key]))) for key,value in actual.items()}
        assert max(errors.values())<1e-10;checks.append(dict(task=task,errors=errors))
        if task=='multilabel':
            optimized=leaf_statistics_gpu(cp.asarray(raw),cp.asarray(y),cp.asarray(leaves),cp.eye(4),task,4,True)
            identity_ref=leaf_statistics_reference(raw,y,leaves,np.eye(4),task)
            assert optimized['hessian'] is None and optimized['cross'] is None
            np.testing.assert_allclose(optimized['full_gradient'].get(),identity_ref['full_gradient'],atol=1e-10)
            np.testing.assert_allclose(optimized['diagonal'].get(),identity_ref['diagonal'],atol=1e-10)
        x=rng.normal(size=(600,7)).astype(np.float32);xx=rng.normal(size=(120,7)).astype(np.float32)
        yy=((x[:,0]>0).astype(int)+2*(x[:,1]>0)) if task=='multiclass' else (x[:,:4]>0).astype(int)
        yv=((xx[:,0]>0).astype(int)+2*(xx[:,1]>0)) if task=='multiclass' else (xx[:,:4]>0).astype(int)
        data=out/f'{task}-data.npz';np.savez(data,train_x=x,train_y=yy,validation_x=xx,validation_y=yv)
        initial=out/f'{task}-initial.npz';np.savez(initial,train=np.zeros((len(x),4),np.float32),validation=np.zeros((len(xx),4)))
        for name,b,residual,full in [('full',np.eye(4),0.,True),('latent',basis,0.,False),('escape',basis,.1,False)]:
            paid+=3;assert paid<=spec['maximum_new_trees']
            request=dict(data_path=str(data),initial_path=str(initial),basis=b.tolist(),residual_fraction=residual,full_coordinates=full,
                task=task,seed=3329,trees=3,config={**cfg,'max_depth':3},output=str(out/task/(name+'.npz')),chunk_rows=128)
            record=fit_direct(request);fits.append(dict(task=task,mode=name,fit=record))
            write(out/'progress.json',dict(checks=checks,fits=fits,paid_new_trees=paid));event('direct_preflight_fit',task=task,mode=name,record=record)
    result=dict(status='direct_preflight_complete',environment=environment,checks=checks,fits=fits,paid_new_trees=paid,test_evaluated=False,
        scope='synthetic mathematical/GPU/export correctness only; no scientific training-performance conclusion')
    write(out/'report.json',result);event('direct_preflight_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):preflight(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
