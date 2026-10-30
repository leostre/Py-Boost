"""Direct latent increments on the pinned native tree engine, without edits to it."""
from dataclasses import replace
from time import perf_counter
import numpy as np
import cupy as cp
from py_boost import GradientBoosting
from experiments.online_leaf_core import compact_topology,TreePart,stitch
from experiments.shared_leaf_basis import reference_class_model


def probability(raw,task):
    if task=='multiclass':
        p=cp.exp(raw-raw.max(1,keepdims=True));return p/p.sum(1,keepdims=True)
    return 1/(1+cp.exp(cp.clip(-raw,-80,80)))


def gradient(p,y,task):
    g=p.copy()
    if task=='multiclass':g[cp.arange(len(g)),y.astype(cp.int32)]-=1
    else:g-=y
    return g


def projected_arrays(raw,y,basis,task,chunk,identity=False):
    n=len(raw);r=basis.shape[1];g=cp.empty((n,r),cp.float32);h=cp.empty_like(g)
    for a in range(0,n,chunk):
        b=min(a+chunk,n);p=probability(raw[a:b],task);gg=gradient(p,y[a:b],task)
        g[a:b]=gg if identity else gg@basis
        diagonal=p*(1-p) if identity else (p@(basis*basis)-(p@basis)**2 if task=='multiclass' else (p*(1-p))@(basis*basis))
        h[a:b]=cp.maximum(diagonal,1e-8)
    return g,h


def leaf_statistics_loop(raw,y,leaf_ids,basis,task,chunk,identity=False):
    """O(leaves*C*r+leaves*r*r), never samples*r*r curvature storage."""
    m=int(leaf_ids.max().item())+1;c=raw.shape[1];r=basis.shape[1]
    g=cp.zeros((m,c),cp.float64);diag=cp.zeros_like(g)
    cross=cp.zeros((m,c,r),cp.float64) if task=='multiclass' else None
    b=basis.astype(cp.float64)
    for a in range(0,len(raw),chunk):
        z=min(a+chunk,len(raw));p=probability(raw[a:z],task).astype(cp.float64);gg=gradient(p,y[a:z],task);ids=leaf_ids[a:z]
        for leaf in range(m):
            mask=ids==leaf
            if not bool(mask.any()):continue
            pp=p[mask];g[leaf]+=gg[mask].sum(0);diag[leaf]+=(pp*(1-pp)).sum(0)
            if task=='multiclass':cross[leaf]+=pp.sum(0)[:,None]*b-pp.T@(pp@b)
    if task=='multilabel' and identity:
        return dict(full_gradient=g,diagonal=diag,cross=None,gradient=g,hessian=None)
    if task=='multilabel':cross=diag[:,:,None]*b[None,:,:]
    h=cross if identity else cp.einsum('cr,lcs->lrs',b,cross,optimize=True)
    return dict(full_gradient=g,diagonal=diag,cross=cross,gradient=g if identity else g@b,hessian=h)


def leaf_statistics_gpu(raw,y,leaf_ids,basis,task,chunk,identity=False):
    """Grouped GEMMs; bounded L*chunk*C workspace, no sample*r*r tensor."""
    m=int(leaf_ids.max().item())+1;c=raw.shape[1];r=basis.shape[1]
    g=cp.zeros((m,c),cp.float64);diag=cp.zeros_like(g);b=basis.astype(cp.float64)
    cross=cp.zeros((m,c,r),cp.float64) if task=='multiclass' else None
    for a in range(0,len(raw),chunk):
        z=min(a+chunk,len(raw));p=probability(raw[a:z],task).astype(cp.float64)
        membership=(cp.arange(m)[:,None]==leaf_ids[a:z][None,:]).astype(cp.float64)
        g+=membership@gradient(p,y[a:z],task);diag+=membership@(p*(1-p))
        if task=='multiclass':
            cross+=(membership@p)[:,:,None]*b[None,:,:]-cp.matmul(
                (membership[:,:,None]*p[None,:,:]).transpose(0,2,1),p@b)
    if task=='multilabel' and identity:
        return dict(full_gradient=g,diagonal=diag,cross=None,gradient=g,hessian=None)
    if task=='multilabel':cross=diag[:,:,None]*b[None,:,:]
    h=cross if identity else cp.einsum('cr,lcs->lrs',b,cross,optimize=True)
    return dict(full_gradient=g,diagonal=diag,cross=cross,gradient=g if identity else g@b,hessian=h)


class DirectLeafBoosting(GradientBoosting):
    def __init__(self,*,initial_train,initial_validation,basis,task,residual_fraction=0.,
                 class_importance=None,full_coordinates=False,chunk_rows=1024,pool_limit_bytes=16*1024**3,cumulative_quota=False,**arguments):
        super().__init__(**arguments)
        self.initial_train=np.asarray(initial_train,np.float32);self.initial_validation=[np.asarray(x,float) for x in initial_validation]
        self.direct_basis=np.asarray(basis,np.float32);self.direct_task=task;self.residual_fraction=float(residual_fraction)
        self.class_importance=np.ones(len(basis)) if class_importance is None else np.asarray(class_importance,float)
        self.full_coordinates=full_coordinates;self.chunk_rows=chunk_rows;self.pool_limit_bytes=pool_limit_bytes
        self.cumulative_quota=bool(cumulative_quota)
        if (task not in ('multiclass','multilabel') or not 0<=self.residual_fraction<=.1 or chunk_rows<1
                or self.class_importance.shape!=(len(basis),) or not np.isfinite(self.class_importance).all() or np.any(self.class_importance<=0)):
            raise ValueError('invalid direct training domain')
        if self.direct_basis.shape[1]==self.direct_basis.shape[0] and self.residual_fraction:
            raise ValueError('full coordinates do not need a redundant sparse escape')
        if not np.allclose(self.direct_basis.T@self.direct_basis,np.eye(self.direct_basis.shape[1]),atol=2e-5):raise ValueError('orthonormal direct basis required')

    def _fit(self,builder,info):
        train,valid=info['data']['train'],info['data']['valid'];self.parts=[];self.direct_trace=[]
        if train['ensemble'].shape!=self.initial_train.shape or len(valid['ensemble'])!=len(self.initial_validation):
            raise ValueError('direct checkpoint domain mismatch')
        train['ensemble'][:]=cp.asarray(self.initial_train)
        for z,initial in zip(valid['ensemble'],self.initial_validation):z[:]=cp.asarray(initial)
        pool=info['mempool'];allocator=cp.cuda.get_allocator();self.training_pool_peak=pool.used_bytes()
        def tracked(size):
            pointer=allocator(size);self.training_pool_peak=max(self.training_pool_peak,pool.used_bytes())
            if self.training_pool_peak>self.pool_limit_bytes:raise MemoryError('direct own training-pool limit')
            return pointer
        with cp.cuda.using_allocator(tracked):self._direct_loop(builder,info)
        self.train_final_raw=train['ensemble'].get();self.validation_final_raw=[a.get() for a in valid['ensemble']]
        self.base_score=self.base_score.get()

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
                cp.linalg.solve(stats['hessian']+lam*cp.eye(r)[None,:,:],-stats['gradient'][:,:,None])[:,:,0])
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

    def export_increment(self):
        bias=np.zeros(self.initial_train.shape[1],np.float32)
        if self.full_coordinates:
            identity=self.direct_basis.shape==(len(bias),len(bias)) and np.array_equal(self.direct_basis,np.eye(len(bias),dtype=np.float32))
            decoded=[TreePart(p.nodes,p.coefficients.copy() if identity else (p.coefficients.astype(float)@self.direct_basis.T).astype(np.float32),p.rows,p.columns,p.values) for p in self.parts]
            model=stitch(decoded,None,bias,self.nfeats,self.direct_task)
            if self.direct_task=='multiclass':model=replace(model,base=reference_class_model(model.base))
            return model
        return stitch(self.parts,self.direct_basis,bias,self.nfeats,self.direct_task)

    def predict(self,*args,**kwargs):raise NotImplementedError('export_increment must be combined with its checkpoint')
