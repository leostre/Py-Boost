"""Deliberately synthetic ensembles for independent packed-kernel checks."""
from dataclasses import replace
import numpy as np
import budget_quality_worker
from abl_budget_revision_core import blank,pack
from experiments.shared_leaf_basis import LeafEnsemble
from experiments.uniform_threshold_core import ThresholdModel


def fixture(c,residual=True,task='multilabel'):
    rng=np.random.default_rng(70+c);blocks=[]
    for index,width in enumerate((c,max(1,min(16,c//2)),max(1,min(8,c//3)))):
        trees=3+index;nodes=np.array([[(-1 if j%2 else 1),-.5+j*.17,-1,-2] for j in range(trees)],np.float32)
        b=np.linalg.qr(rng.normal(size=(c,width)))[0].astype(np.float32) if index else np.empty((0,0),np.float32)
        base=LeafEnsemble(nodes,np.arange(trees+1,dtype=np.int64),2*np.arange(trees+1,dtype=np.int64),
            rng.normal(0,.2,(2*trees,width)).astype(np.float32),b,np.zeros(c,np.float32),1,task,'low_rank' if index else 'dense')
        block=blank(base)
        if residual and index:
            rr=np.repeat(np.arange(2*trees),min(c,3)).astype(np.int32);cc=np.tile(np.arange(min(c,3)),2*trees).astype(np.int32)
            block=replace(block,rows=rr,columns=cc,values=rng.normal(0,.1,len(rr)).astype(np.float32))
        blocks.append(block)
    model=pack(blocks,rng.normal(0,.1,c).astype(np.float32))
    x=np.r_[np.array([np.nan,-1.,0.,1.],np.float32),rng.normal(size=15).astype(np.float32)][:,None]
    shifts=rng.normal(0,.1,c).astype(np.float32)
    return model,x,shifts


def gpu_preflight():
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=1024**3)
    from scipy.special import expit,softmax
    from packed_serving_gpu import PackedGpu
    results=[];skipped=[]
    for c,task in [(6,'multilabel'),(159,'multilabel'),(374,'multilabel'),(983,'multilabel'),(6,'multiclass')]:
        for residual in (False,True):
            model,x,shifts=fixture(c,residual,task);reference=model.raw(x);t=1. if task=='multilabel' else 1.3
            prob=np.clip(expit(reference/t) if task=='multilabel' else softmax(reference/t,axis=1),1e-7,1-1e-7)
            labels=reference/t>=shifts if task=='multilabel' else (reference/t-shifts).argmax(1)
            for groups in (1,2,4):
                for layout in ('aos','soa'):
                    try:device=PackedGpu(model,t,shifts,groups,layout)
                    except ValueError as exc:
                        skipped.append(dict(outputs=c,residual=residual,groups=groups,layout=layout,reason=str(exc)));continue
                    dx=cp.asarray(x);out,ml,mc=device.buffers(len(x));device.launch_into(dx,out,ml,mc,raw=True)
                    error=float(np.max(np.abs(out.get()-reference)));assert error<1e-10,(c,groups,layout,error)
                    actual,actual_labels=device.request(x)
                    np.testing.assert_allclose(actual,prob,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(actual_labels,labels)
                    empty,empty_labels=device.request(x[:0]);assert empty.shape==(0,c) and len(empty_labels)==0
                    results.append(dict(outputs=c,task=task,residual=residual,groups=groups,layout=layout,
                         raw_error=error,shared_bytes=device.plan.shared_bytes,resident_bytes=device.resident_device_array_bytes,
                         kernel_attributes=device.kernel.attributes,labels_equal=True,empty_batch_passed=True))
                    del device,dx,out,ml,mc;cp.get_default_memory_pool().free_all_blocks()
    return dict(status='packed_gpu_preflight_passed',checks=results,skipped=skipped)
