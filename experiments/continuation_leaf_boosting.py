"""Native continuation fits initialized from actual checkpoint predictions."""
import numpy as np
import cupy as cp
from py_boost import GradientBoosting
from experiments.online_leaf_core import compact_topology,stitch


class ContinuationLeafBoosting(GradientBoosting):
    def __init__(self,*,initial_train,initial_validation,task,pool_limit_bytes=None,**arguments):
        super().__init__(**arguments)
        self.initial_train = np.asarray(initial_train,np.float32)
        self.initial_validation = [np.asarray(a,float) for a in initial_validation]
        self.study_task = task
        self.pool_limit_bytes = pool_limit_bytes

    def _fit(self,builder,build_info):
        train,valid = build_info['data']['train'],build_info['data']['valid']
        if train['ensemble'].shape != self.initial_train.shape or len(valid['ensemble']) != len(self.initial_validation):
            raise ValueError('continuation ensemble shape mismatch')
        train['ensemble'][:] = cp.asarray(self.initial_train)
        for state,initial in zip(valid['ensemble'],self.initial_validation):
            if state.shape != initial.shape:
                raise ValueError('continuation validation shape mismatch')
            state[:] = cp.asarray(initial)
        pool,allocator = build_info['mempool'],cp.cuda.get_allocator()
        self.training_pool_peak = pool.used_bytes()
        def tracked(size):
            pointer = allocator(size)
            self.training_pool_peak = max(self.training_pool_peak,pool.used_bytes())
            if self.pool_limit_bytes is not None and self.training_pool_peak > self.pool_limit_bytes:
                raise MemoryError('continuation pool exceeded explicit resource cap')
            return pointer
        with cp.cuda.using_allocator(tracked):
            super()._fit(builder,build_info)
        self.train_final_raw = train['ensemble'].get().copy()
        self.validation_final_raw = [a.get().copy() for a in valid['ensemble']]

    def export_increment(self):
        parts = [compact_topology(t.test_format,t.values)[0] for t in self.models]
        return stitch(parts,None,np.zeros(self.initial_train.shape[1],np.float32),self.nfeats,self.study_task)

    def predict(self,X,batch_size=100_000):
        # The inherited native predictor would add its old class prior, not
        # the checkpoint function. This fitting primitive must not silently
        # masquerade as an entire continued ensemble on new objects.
        raise NotImplementedError('Continuation owns only increments; combine export_increment() '
                                  'with the original checkpoint and predict with that artifact')

    def predict_staged(self,X,iterations=None,batch_size=100_000):
        raise NotImplementedError('Continuation staged prediction requires the original checkpoint; '
                                  'use saved combined checkpoint artifacts')
