"""GaLore-inspired output update spaces, not an Adam implementation."""
import time
import numpy as np
import cupy as cp

from experiments.online_leaf_boosting import CompactLeafBoosting
from experiments.online_leaf_core import stitch
from experiments.block_leaf_model import BlockLeafModel
from experiments.gradient_basis_core import GradientBasisState,gradient_signal,captured_energy


class HistoryLeafBoosting(CompactLeafBoosting):
    def __init__(self,*,basis_source='history',basis_period=20,observation_period=5,beta=.8,**arguments):
        if basis_source not in ('current','mean','history') or basis_period < 1 or observation_period < 1:
            raise ValueError('invalid history basis schedule')
        if basis_period%observation_period:
            raise ValueError('observation period must divide basis period')
        super().__init__(compact_mode='hybrid',**arguments)
        self.basis_source,self.basis_period,self.observation_period,self.beta = basis_source,basis_period,observation_period,beta
        self._reset_history()

    def _reset_history(self):
        self.estimator = GradientBasisState(self.basis_source,self.beta)
        self.basis_registry,self.part_basis_ids,self.basis_events,self.observer_trace = [],[],[],[]
        self.active_basis_id,self.observer_seconds = 0,0.

    def compact_part_added(self,step):
        self.part_basis_ids.append(self.active_basis_id)

    def compact_basis_ready(self):
        self.basis_registry.append(self.basis)
        self.basis_events.append(dict(tree_start=1,basis_id=0,source='shared_functional_warmup'))

    def before_compact_tree(self,step,grad,hess):
        if step < self.warmup:
            return
        refresh = step >= 2*self.warmup and step%self.basis_period == 0
        observe = step%self.observation_period == 0 and (self.basis_source != 'current' or refresh)
        if not observe:
            return
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        z = gradient_signal(grad.get(),hess.get(),self.lambda_l2,self.study_task)
        captured = captured_energy(z,self.basis)
        self.estimator.observe(z)
        self.observer_trace.append(dict(state_step=step,basis_id=self.active_basis_id,
            captured_by_previous_basis=captured,observations=self.estimator.observations,
            estimator_array_bytes=sum(a.nbytes for a in (self.estimator.mean,self.estimator.gram) if a is not None),
            diagnostic_array_bytes=sum(a.nbytes for a in (self.estimator.half_grams or []))))
        if refresh:
            b,diagnostics = self.estimator.estimate(self.study_task,self.basis.shape[1],self.basis)
            changed = not np.array_equal(b,self.basis)
            if changed:
                self.basis = b
                self.active_basis_id = len(self.basis_registry)
                self.basis_registry.append(b)
            self.basis_events.append(dict(tree_start=step+1,basis_id=self.active_basis_id,
                source=self.basis_source,changed=changed,**diagnostics))
        self.observer_seconds += time.perf_counter()-started

    def _fit(self,builder,build_info):
        self._reset_history()
        super()._fit(builder,build_info)
        if len(self.part_basis_ids) != len(self.parts):
            raise RuntimeError('part/basis registry mismatch')
        for row in self.trace:
            ids = set(self.part_basis_ids[:row['step']])
            basis_bytes = sum(self.basis_registry[j].nbytes for j in ids)
            row['basis_count'] = len(ids)
            row['retained_payload_bytes'] += basis_bytes-self.basis_registry[0].nbytes
            observed = [r for r in self.observer_trace if r['state_step'] < row['step']]
            row['estimator_array_bytes'] = observed[-1]['estimator_array_bytes'] if observed else 0
            row['diagnostic_array_bytes'] = observed[-1]['diagnostic_array_bytes'] if observed else 0

    def export(self,shifts=None):
        if len(self.part_basis_ids) != len(self.parts) or not self.basis_registry:
            raise RuntimeError('incomplete basis registry')
        blocks = []
        start = 0
        zero = np.zeros(self.outputs,np.float32)
        for stop in range(1,len(self.parts)+1):
            if stop == len(self.parts) or self.part_basis_ids[stop] != self.part_basis_ids[start]:
                blocks.append(stitch(self.parts[start:stop],self.basis_registry[self.part_basis_ids[start]],
                    zero,self.nfeats,self.study_task,zero))
                start = stop
        return BlockLeafModel(tuple(blocks),self.study_bias.copy(),
            np.zeros(self.outputs,np.float32) if shifts is None else np.asarray(shifts,np.float32))
