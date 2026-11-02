"""Quantized-forward continuation; pinned parent equations, explicit deployment factors."""
from direct_leaf_gpu import *

class QuantizedDirectLeafBoosting(DirectLeafBoosting):
    def __init__(self,*args,forward_codec,fixed_a,**kwargs):
        super().__init__(*args,**kwargs)
        from six_quant_core import quantize_columns
        self.original_basis=self.direct_basis.copy();self.forward_codec=forward_codec
        self.fixed_a=np.asarray(fixed_a,np.float32);self.quant_clipped=0;self.quant_total=0
        qb,s=quantize_columns(self.original_basis,forward_codec)
        self.direct_basis=(qb.astype(np.float32)*(s if forward_codec=='int8' else 1.)).astype(np.float32)

    def forward_quantize(self,a):
        self.quant_total+=a.size
        if self.forward_codec=='fp16':return a.astype(cp.float16).astype(cp.float32)
        scale=cp.asarray(self.fixed_a);self.quant_clipped+=int(cp.sum(cp.abs(a)>127*scale).item())
        return (cp.clip(cp.rint(a/scale),-127,127).astype(cp.int8).astype(cp.float32)*scale).astype(cp.float32)

    def export_increment(self):
        # A topology/factor archive for preparing QuantSnapshot, NOT the deployed
        # floating function. Actual forward uses quantized original_basis.
        return stitch(self.parts,self.original_basis,np.zeros(self.initial_train.shape[1],np.float32),self.nfeats,self.direct_task)

    def _direct_loop(self,builder,info):
        train,valid=info['data']['train'],info['data']['valid'];b=cp.asarray(self.direct_basis);task=self.direct_task
        r=b.shape[1];c=b.shape[0];lam=float(self.params['lambda_l2']);lr=float(self.params['lr'])
        identity=self.direct_basis.shape==(c,c) and np.array_equal(self.direct_basis,np.eye(c,dtype=np.float32))
        self.callbacks.before_train(info)
        quota_atoms=quota_paid=0
        for iteration in range(self.ntrees):
            info['num_iter']=iteration;t=perf_counter()
            train['grad'],train['hess']=projected_arrays(train['ensemble'],train['target'],b,task,self.chunk_rows,identity)
            cp.cuda.get_current_stream().synchronize();projection_seconds=perf_counter()-t
            self.callbacks.before_iteration(info);t=perf_counter()
            tree,leaves,_,val_leaves,_=builder.build_tree(train['features_gpu'],train['grad'],train['hess'],train['sample_weight'],None,*valid['features_gpu'])
            cp.cuda.get_current_stream().synchronize();structure_seconds=perf_counter()-t;t=perf_counter()
            if leaves.shape[1]!=1:raise RuntimeError('direct pilot requires one shared tree structure')
            ids=leaves[:,0];stats=leaf_statistics_gpu(train['ensemble'],train['target'],ids,b,task,self.chunk_rows,identity)
            a=(-stats['full_gradient']/(stats['diagonal']+lam) if identity and task=='multilabel' else
                cp.linalg.solve(stats['hessian']+lam*(b.astype(cp.float64).T@b.astype(cp.float64))[None,:,:],-stats['gradient'][:,:,None])[:,:,0])
            residual=cp.zeros_like(stats['full_gradient']);active=int(cp.unique(ids).size)
            quota_atoms+=active*c
            target=int(np.floor(self.residual_fraction*quota_atoms))
            count=(target-quota_paid) if self.cumulative_quota else int(np.floor(self.residual_fraction*active*c))
            if self.cumulative_quota:quota_paid=target
            if count:
                v=a@b.astype(cp.float64).T
                residual_gradient=stats['full_gradient']+cp.einsum('lcr,lr->lc',stats['cross'],a)+lam*v
                importance=cp.asarray(self.class_importance)
                scores=residual_gradient**2/(stats['diagonal']+lam)*importance[None,:]
                used=cp.bincount(ids,minlength=len(a))>0;scores[~used]=0
                order=cp.asarray(np.argsort(-scores.get().ravel(),kind='stable')[:count])
                order=order[scores.ravel()[order]>1e-24]
                residual.ravel()[order]=-residual_gradient.ravel()[order]/(stats['diagonal']+lam).ravel()[order]
            a=(a*lr).astype(cp.float32);residual=(residual*lr).astype(cp.float32)
            a=self.forward_quantize(a)
            cp.cuda.get_current_stream().synchronize();coupled_seconds=perf_counter()-t;t=perf_counter()
            pred=(a[ids] if identity else a[ids]@b.T)+residual[ids]
            val_preds=[(a[v[:,0]] if identity else a[v[:,0]]@b.T)+residual[v[:,0]] for v in val_leaves]
            train['ensemble']+=pred
            for current,update in zip(valid['ensemble'],val_preds):current+=update
            train['last_tree']=dict(leaves=leaves,preds=pred);valid['last_tree']=dict(leaves=val_leaves,preds=val_preds)
            tree.set_node_values(a.get(),np.zeros(r,np.uint64));self.models.append(tree)
            stop=self.callbacks.after_iteration(info);tree.reformat(nfeats=self.nfeats,debug=self.params['debug'])
            part,lookup,_=compact_topology(tree.test_format,tree.values)
            residual_cpu=residual.get();rows,cols=np.nonzero(residual_cpu);keep=lookup[rows]>=0
            self.parts.append(TreePart(part.nodes,part.coefficients,lookup[rows[keep]].astype(np.int32),cols[keep].astype(np.int32),residual_cpu[rows[keep],cols[keep]]))
            cp.cuda.get_current_stream().synchronize();update_export_seconds=perf_counter()-t
            self.direct_trace.append(dict(iteration=iteration,rank=r,outputs=c,active_leaves=active,
                projection_seconds=projection_seconds,structure_seconds=structure_seconds,coupled_seconds=coupled_seconds,
                update_export_seconds=update_export_seconds,latent_derivative_bytes=2*len(ids)*r*4,
                coupled_statistics_bytes=sum({int(x.data.ptr):x.nbytes for x in stats.values() if x is not None}.values()),
                optimized_identity_coordinates=identity,nnz=len(self.parts[-1].values),
                cumulative_quota=self.cumulative_quota,quota_atoms=quota_atoms,requested_residual_count=count))
            del stats,pred,val_preds,residual
            if stop:break
        self.callbacks.after_train(info)
