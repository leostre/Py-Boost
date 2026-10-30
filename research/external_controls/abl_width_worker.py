"""One bounded min-leaf50 trajectory in the immutable native runtime."""
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from budget_quality_worker import digest,write,event
from abl_quality_worker import before_test_data
from abl_quality_core import array_ledger


def main():
    import cupy as cp
    from py_boost import GradientBoosting
    from experiments.a100_runner import build_model
    from experiments.shared_leaf_basis import export_teacher,reference_class_model
    from experiments.closed_budget_study import empty_model
    from experiments.parallel_leaf_core import prefix
    from experiments.inference_gpu_guard import gpu_probe
    request=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
    if request['depth'] not in (3,5) or request['minimum_leaf']!=50 or request['rounds']!=800:
        raise RuntimeError('outside fixed width experiment')
    ready,reason,observations=gpu_probe('shared',8192)
    if not ready:raise RuntimeError(reason)
    locks={};data=before_test_data(request['record'],locks);case=request['record']['case']
    ncal=data['sizes'][0]
    class MeasuredBoosting(GradientBoosting):
        def _fit(self,builder,info):
            self.pool=info['mempool'];self.peak=self.pool.used_bytes();allocator=cp.cuda.get_allocator()
            def tracked(size):
                pointer=allocator(size);self.peak=max(self.peak,self.pool.used_bytes())
                if self.peak>16*1024**3:raise MemoryError('width training pool exceeded16GiB')
                return pointer
            with cp.cuda.using_allocator(tracked):return super()._fit(builder,info)
    cfg=request['training_config']
    template=build_model('full','multiclass',case['outputs'],cfg,case['seed'],request['depth'],1.,trees=800)
    params=dict(template.params);params['min_data_in_leaf']=50
    native=MeasuredBoosting(**params);del template
    start=perf_counter()
    native.fit(data['train_x'],data['train_y'],eval_sets=[dict(X=data['dev_x'][:ncal],y=data['dev_y'][:ncal])])
    cp.cuda.get_current_stream().synchronize()
    dense=export_teacher(native,data['train_x'].shape[1],'multiclass')
    dense.save(output/'trained_dense_teacher.npz')
    training_visits=dense.leaf_indices(data['train_x'])
    support=np.bincount(training_visits.ravel(),minlength=len(dense.coefficients))
    minimum_support=int(support[support>0].min())
    if minimum_support<50:raise RuntimeError('exported tree violates minimum leaf support50')
    del training_visits,support
    expected=native.predict(data['dev_x'][:64]);expected=expected.get() if hasattr(expected,'get') else np.asarray(expected)
    np.testing.assert_allclose(dense.predict(data['dev_x'][:64]),expected,atol=2e-5,rtol=0)
    full=empty_model(reference_class_model(dense));items=[]
    fit=dict(case=case['dataset'],seed=case['seed'],depth=request['depth'],minimum_leaf=50,
        rounds=800,seconds=perf_counter()-start,own_pool_peak_bytes=int(native.peak),gpu_probe=observations,
        observed_minimum_leaf_support=minimum_support,
        actual_minimum_leaf=native.params['min_data_in_leaf'],new_model_fits=1,new_boosting_rounds=800,
        interpreter=sys.executable,test_targets_used=False)
    for rounds in (40,80,160,200,400,800):
        identity=f'w50-d{request["depth"]}-t{rounds}';model=prefix(full,rounds)
        path=output/(identity+'.npz');model.save(path)
        logits=model.prepare_inference().raw(data['dev_x'])
        development=output/(identity+'-development.npz')
        a,b=np.cumsum(data['sizes'])[:2]
        np.savez_compressed(development,calibration=logits[:a],control=logits[a:b],selection=logits[b:])
        ledger=array_ledger(model)
        items.append(dict(identity=identity,artifact=str(path),sha256=digest(path),depth=request['depth'],
            rounds=rounds,origin='new',minimum_leaf=50,development_file=str(development),
            development_sha256=digest(development),ledger=ledger,leaves=len(model.base.coefficients),
            internal_nodes=len(model.base.nodes)))
    for path,sha in locks.items():
        if digest(path)!=sha:raise RuntimeError('input changed during width training')
    write(output/'trajectory.json',dict(status='complete',models=items,fit=fit,input_locks=locks))
    event('abl_width_trajectory_saved',**fit)
    del native,full,dense;gc.collect();cp.get_default_memory_pool().free_all_blocks()


if __name__=='__main__':main()
