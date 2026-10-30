"""Shared compact-training proposal algebra; native_source stays immutable."""
from dataclasses import replace
import numpy as np
import budget_quality_worker
from abl_budget_revision_core import DenseView,decode_model,blank,pack,prune_tree,hessian
from abl_budget_revision import Group,group_model
from abl_budget_revision_core import fixed_support_projection
from abl_protected_core import as_protected,append_protected,block_cost
from experiments.shared_leaf_basis import reference_class_model,functional_basis
from experiments.gradient_basis_core import contrast_matrix
from experiments.sparse_leaf_core import contrast,training_weights,coordinate_scores,score_order


def prepare_training_group(view,basis,features,labels,teacher,hh):
    # BR0's group builder was intentionally multilabel-only. U2 supports both tasks.
    view=replace(view,values=contrast(view.values,view.task))
    basis=None if basis is None else np.asarray(basis,np.float32)
    designs={k:view.design(x) for k,x in features.items()}
    d=np.asarray(designs['train'].T@hh)+1.;frequency=np.asarray(designs['train'].mean(0)).ravel()
    if basis is None:
        a=view.values.astype(np.float32);residual=np.zeros_like(a);order=np.empty(0,np.int64)
    else:
        a,_,_,_=fixed_support_projection(view.values,d,basis)
        residual=(view.values-a.astype(float)@basis.astype(float).T).astype(np.float32)
        offset=frequency@residual.astype(float)
        error=contrast(np.asarray(designs['train']@residual.astype(float))-offset,view.task)
        weights=training_weights(labels['train'],teacher,view.task,'balanced_mse',.5)
        order=score_order(coordinate_scores(designs['train'],residual,error,'balanced_mse',weights),residual)
    decoded=a.astype(float) if basis is None else a.astype(float)@basis.astype(float).T
    return Group(view,basis,d,designs,frequency,a,residual,order,frequency@(view.values-decoded))


def canonical(model):
    if model.task!='multiclass':return model
    return pack([replace(b,base=reference_class_model(b.base)) if b.base.representation=='dense' else b
        for b in model.blocks],model.bias)


def coarsen(increment,x,teacher_raw,cap):
    protected=as_protected(increment);view,_=decode_model(protected)
    if cap>=8 and np.max(np.diff(view.leaf_offsets))<=cap:return canonical(protected)
    view=replace(view,values=contrast(view.values,view.task))
    design=view.design(x);weights=np.asarray(design.T@hessian(teacher_raw,view.task))+1.
    basis=np.eye(len(view.bias))
    nodes=[];values=[];no=[0];lo=[0]
    for t in range(view.trees):
        a,b=view.leaf_offsets[t:t+2];u,v=view.node_offsets[t:t+2]
        p=prune_tree(view.nodes[u:v],view.values[a:b],weights[a:b],basis,cap)
        nodes.append(p['nodes']);values.append(p['target']);no.append(no[-1]+len(p['nodes']));lo.append(lo[-1]+len(p['target']))
    result=DenseView(np.concatenate(nodes),np.asarray(no,np.int64),np.asarray(lo,np.int64),
        np.concatenate(values),view.bias,view.features,view.task)
    return canonical(pack([blank(result.base())],view.bias))


def rank_grid(outputs,task):
    k=outputs-int(task=='multiclass')
    if outputs==10 and task=='multiclass':return (4,6,8,9)
    if outputs>50:return tuple(sorted({16,32,50,64,k}&set(range(1,k+1))))
    return tuple(sorted({max(1,int(np.ceil(k*f))) for f in (.25,.5,.75,1.)}))


def candidate_grid(checkpoint,increment,features,labels,budget,*,protected=False,basis=None,
                   ranks=None,caps=None,residuals=(0.,.02,.05)):
    """One paid full proposal, multiple archived representations; no D labels in fitting."""
    if checkpoint is None and protected:raise ValueError('protected mode needs a checkpoint')
    original=as_protected(increment) if checkpoint is None else append_protected(checkpoint,increment)
    c=len(original.bias);task=original.task
    teacher=original.raw(features['train']);hh=hessian(teacher,task)
    caps=(4,6,8) if caps is None and c<=20 else ((8,) if caps is None else caps)
    ranks=rank_grid(c,task) if ranks is None else ranks
    for cap in caps:
        short=coarsen(increment,features['train'],teacher,cap)
        if protected:
            source=short;view,_=decode_model(source)
            target={key:checkpoint.raw(x)+source.raw(x) for key,x in features.items()}
        else:
            source=short if checkpoint is None else pack(checkpoint.blocks+short.blocks,checkpoint.bias+short.bias)
            # Append zero-bias native blocks exactly once; common checkpoint stays once.
            view,_=decode_model(source);target={key:source.raw(x) for key,x in features.items()}
        contribution=np.asarray(view.design(features['train'])@view.values)
        all_basis=functional_basis(contribution,task)[1] if basis is None else basis
        for rank in ranks:
            full=rank==c-int(task=='multiclass')
            b=None if full else all_basis[:,:rank]
            group=prepare_training_group(view,b,features,labels,teacher,hh)
            seen=set()
            for fraction in ((0.,) if full else residuals):
                count=min(len(group.order),int(np.floor(fraction*view.values.size)))
                if count in seen:continue
                seen.add(count)
                leaf,offset=group_model(group,group.order[:count])
                if protected:
                    model=pack(checkpoint.blocks+(leaf,),checkpoint.bias+view.bias+offset)
                else:model=pack([leaf],view.bias+offset)
                model=canonical(model);cost=block_cost(model)
                if budget is not None and cost['cost_bytes']>budget:continue
                raw={key:model.raw(x) for key,x in features.items()}
                error=float(np.mean(contrast(teacher-raw['train'],task)**2))
                yield dict(model=model,raw=raw,rank=rank,leaf_cap=cap,residual_fraction=fraction,
                    nnz=cost['nnz'],cost=cost,train_mse=error,basis_kind='functional' if basis is None else 'gradient',
                    uid=f'l{cap}-r{rank}-s{count}',new_increment_trees=increment.base.n_trees)


def shortlist(candidates,maximum=6):
    """Bound development queries explicitly; train-only shortlist, no quality gate."""
    candidates=list(candidates)
    if not candidates:return []
    selected={}
    for r in sorted({c['rank'] for c in candidates}):
        a=min((c for c in candidates if c['rank']==r),key=lambda c:(c['train_mse'],c['cost']['cost_bytes'],c['uid']))
        selected[a['uid']]=a
    a=min(candidates,key=lambda c:(c['cost']['cost_bytes'],c['train_mse'],c['uid']));selected[a['uid']]=a
    return sorted(selected.values(),key=lambda c:(c['train_mse'],c['cost']['cost_bytes'],c['uid']))[:maximum]


def scaled_increment(increment,weight):
    if not np.isfinite(weight) or weight<=0:raise ValueError('positive branch weight required')
    return replace(increment,base=replace(increment.base,coefficients=(increment.base.coefficients*weight).astype(np.float32),
        bias=(increment.base.bias*weight).astype(np.float32)),values=(increment.values*weight).astype(np.float32))


def merge_increment_views(increments,weights):
    if len(increments)!=len(weights) or not increments:raise ValueError('matching nonempty branches required')
    models=[as_protected(scaled_increment(inc,w)) for inc,w in zip(increments,weights)]
    combined=pack(tuple(b for m in models for b in m.blocks),sum((m.bias.astype(float) for m in models)).astype(np.float32))
    view,_=decode_model(combined)
    return blank(replace(view.base(),bias=view.bias.copy())),combined
