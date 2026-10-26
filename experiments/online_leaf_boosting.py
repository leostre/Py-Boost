"""Research-only PyBoost training loop with genuinely compact retained leaves.

Full G/H and full-output structure search are deliberately retained in OPT1.
One transient full leaf vector is built, projected and discarded per step.
Next gradients see the actual compact update, not a hidden full teacher.
"""
import hashlib
import time
import numpy as np
import cupy as cp

from py_boost import GradientBoosting
from py_boost.gpu.utils import calc_node_values
from experiments.online_leaf_core import compact_topology, project_part, stitch,residual_rank_allowance
from experiments.shared_leaf_basis import functional_basis, probabilities


class CompactLeafBoosting(GradientBoosting):
    def __init__(self, *, compact_mode='hybrid', rank_fraction=.5, residual_fraction=.05,
                 warmup=20, task='multilabel', force_identity=False, **arguments):
        super().__init__(**arguments)
        if compact_mode not in ('full','rank','hybrid','more_rank') or not 0 < rank_fraction <= 1:
            raise ValueError('invalid compact training mode')
        if not 0 <= residual_fraction <= 1 or warmup < 1:
            raise ValueError('invalid compact training budget')
        self.compact_mode,self.rank_fraction,self.residual_fraction = compact_mode,rank_fraction,residual_fraction
        self.warmup,self.study_task,self.force_identity = warmup,task,force_identity

    def before_compact_tree(self,step,grad,hess):
        """Research extension point; default fixed-basis path has no effects."""

    def compact_part_added(self,step):
        """Associate a new part with its basis, if a subclass needs a registry."""

    def compact_basis_ready(self):
        """Called once after the shared warmup representation is committed."""

    def _hessian_weights(self, hess, leaves, tree, canonical_rows):
        if hess.shape[1] != self.outputs:
            raise ValueError('OPT1 requires per-output diagonal Hessians')
        # Reuse the native fixed-order leaf reduction: -sum(H)/(0+1).
        # No atomics, no full N*C GPU->CPU transfer. Same all-row leaf domain.
        negative = calc_node_values(hess,cp.zeros((len(hess),1),cp.float32),leaves,canonical_rows,
            cp.zeros(self.outputs,cp.uint64),tree.max_leaves,lr=1.,lambda_l2=1.,deterministic=True)
        return -negative.get().astype(float)+self.lambda_l2

    def _fit(self,builder,build_info):
        if self.gd_steps != 1 or self.es != 0 or self.params['target_splitter'] != 'Single':
            raise ValueError('research compact trainer requires gd_steps1/es0/Single')
        train,valid = build_info['data']['train'],build_info['data']['valid']
        self.outputs = train['ensemble'].shape[1]
        if self.warmup >= self.ntrees:
            raise ValueError('warmup must be shorter than training')
        self.parts,self.full_weights,self.warm_memberships = [],[],[]
        self.basis,self.study_bias = None,self.base_score.get().copy()
        self.trace,self.stage_ms = [],{k:0. for k in ('gradient','build','statistics','projection','update','basis')}
        self.warm_fingerprint,self.warm_basis_full = None,None
        self.callbacks.before_train(build_info)
        canonical_rows = cp.arange(len(train['ensemble']),dtype=cp.uint64)
        pool = build_info['mempool']
        allocator = cp.cuda.get_allocator()
        self.training_pool_peak = pool.used_bytes()
        def tracked(size):
            pointer = allocator(size)
            self.training_pool_peak = max(self.training_pool_peak,pool.used_bytes())
            return pointer
        sync = cp.cuda.get_current_stream().synchronize
        def measured(name,fn):
            sync()
            started = time.perf_counter()
            result = fn()
            sync()
            self.stage_ms[name] += (time.perf_counter()-started)*1000
            return result
        with cp.cuda.using_allocator(tracked):
            for i in range(self.ntrees):
                build_info['num_iter'] = i
                train['grad'],train['hess'] = measured('gradient',lambda:self.loss(train['target'],train['ensemble']))
                self.callbacks.before_iteration(build_info)
                self.before_compact_tree(i,train['grad'],train['hess'])
                tree,leaves,pred,val_leaves,val_preds = measured('build',lambda:builder.build_tree(
                    train['features_gpu'],train['grad'],train['hess'],train['sample_weight'],
                    lambda x:self.loss(train['target'],train['ensemble']+x),*valid['features_gpu']))
                weights = measured('statistics',lambda:self._hessian_weights(train['hess'],leaves,tree,canonical_rows))
                tree.reformat(nfeats=self.nfeats,debug=False)
                part,lookup,ids = compact_topology(tree.test_format,tree.values)
                weights = weights[ids]
                lookup_gpu = cp.asarray(lookup)
                memberships = [lookup_gpu[a[:,0]] for a in [leaves,*val_leaves]]
                if any(bool((a < 0).any()) for a in memberships):
                    raise RuntimeError('object reached an unmapped leaf')
                if i < self.warmup:
                    self.warm_memberships.append(memberships)
                    self.full_weights.append(weights)
                elif self.compact_mode == 'full':
                    self.full_weights.append(weights)
                if self.basis is not None:
                    fraction = self.residual_fraction if self.compact_mode == 'hybrid' else 0.
                    part = measured('projection',lambda:project_part(part,weights,self.basis,fraction))
                    decoded = cp.asarray(part.decoded(self.basis))
                    pred,val_preds = decoded[memberships[0]],[decoded[a] for a in memberships[1:]]
                self.parts.append(part)
                self.compact_part_added(i)
                def update():
                    train['ensemble'][:] += pred
                    for old,new in zip(valid['ensemble'],val_preds):
                        old[:] += new
                measured('update',update)
                if i+1 == self.warmup:
                    self.warm_fingerprint = hashlib.sha256(b''.join(p.nodes.tobytes()+p.coefficients.tobytes()
                        for p in self.parts)+train['ensemble'].get().tobytes()).hexdigest()
                    def initialize_basis():
                        contribution = train['ensemble'].get().astype(float)-self.study_bias
                        self.warm_train_mean = contribution.mean(axis=0)
                        _,full_basis,_ = functional_basis(contribution,self.study_task)
                        self.warm_basis_full = full_basis.astype(np.float32)
                        if self.compact_mode == 'full':
                            return
                        rank = max(1,int(np.floor(full_basis.shape[1]*self.rank_fraction)))
                        if self.compact_mode == 'more_rank':
                            rank = min(full_basis.shape[1],rank+residual_rank_allowance(self.residual_fraction,self.outputs))
                        self.basis = (np.eye(self.outputs,dtype=np.float32) if self.force_identity else
                                      self.warm_basis_full[:,:rank].copy())
                        fraction = self.residual_fraction if self.compact_mode == 'hybrid' else 0.
                        self.parts = [project_part(p,w,self.basis,fraction) for p,w in zip(self.parts,self.full_weights)]
                        # Training-only global mean preservation for the shared prefix.
                        new = np.zeros_like(contribution)
                        for p,membership in zip(self.parts,self.warm_memberships):
                            new += p.decoded(self.basis)[membership[0].get()]
                        self.study_bias = (self.study_bias+(contribution-new).mean(axis=0)).astype(np.float32)
                        for old in [train['ensemble'],*valid['ensemble']]:
                            old[:] = cp.asarray(self.study_bias)
                        for p,memberships0 in zip(self.parts,self.warm_memberships):
                            decoded = cp.asarray(p.decoded(self.basis))
                            for old,membership in zip([train['ensemble'],*valid['ensemble']],memberships0):
                                old[:] += decoded[membership]
                    measured('basis',initialize_basis)
                    self.warm_memberships.clear()
                    if self.compact_mode != 'full':
                        self.full_weights.clear()
                        self.warm_basis_full = None
                    self.compact_basis_ready()
                self.models = self.parts
                train['last_tree'] = dict(leaves=leaves,preds=pred)
                valid['last_tree'] = dict(leaves=val_leaves,preds=val_preds)
                if self.callbacks.after_iteration(build_info):
                    raise RuntimeError('unexpected early stopping')
                if (i+1)%20 == 0 or i+1 == self.ntrees:
                    self.trace.append(dict(step=i+1,rank=self.outputs if self.basis is None else self.basis.shape[1],
                        retained_payload_bytes=sum(p.payload_bytes for p in self.parts)+self.study_bias.nbytes+
                            (0 if self.basis is None else self.basis.nbytes),
                        sparse_entries=sum(len(p.values) for p in self.parts),
                        training_pool_used_bytes=pool.used_bytes(),
                        training_pool_peak_bytes=self.training_pool_peak))
        self.callbacks.after_train(build_info)
        self.base_score = self.base_score.get()
        self.train_final_raw = train['ensemble'].get().copy()
        self.validation_final_raw = [a.get().copy() for a in valid['ensemble']]

    def export(self,shifts=None):
        return stitch(self.parts,self.basis,self.study_bias,self.nfeats,self.study_task,shifts)

    def predict(self,features,batch_size=None):
        return self.export().predict(features)
