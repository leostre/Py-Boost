"""Protected old function plus compressed new increments; no refitting old bases."""
from dataclasses import replace
import numpy as np
from budget_quality_worker import native_load
from abl_quality_core import full_output_base,fingerprint,array_ledger
from abl_training_core import calibrated_local_variants
from experiments.block_leaf_model import BlockLeafModel
from experiments.parallel_leaf_core import blocks_of
from experiments.shared_leaf_basis import functional_basis,compress,quality_guard
from experiments.uniform_threshold_core import ThresholdModel
from experiments.sparse_leaf_core import (leaf_design,contrast,training_weights,coordinate_scores,score_order,make_sparse_model)


def append_protected(checkpoint,increment):
    blocks,bias,features=blocks_of(checkpoint)
    new,new_bias,new_features=blocks_of(increment)
    if features!=new_features or checkpoint.task!=increment.base.task:raise ValueError('protected increment domain mismatch')
    result=BlockLeafModel(tuple(blocks)+tuple(new),(bias+new_bias).astype(np.float32),np.zeros_like(bias))
    if [fingerprint(b) for b in result.blocks[:len(blocks)]]!=[fingerprint(b) for b in blocks]:
        raise RuntimeError('old protected block was changed')
    return result


def block_cost(model):
    # COO packed model and shared zero are one real representation. A reference
    # serving implementation retaining individually prepared blocks owns the
    # copied zero vectors, so count them instead of pretending they are shared.
    coo=model.resident_array_bytes+4
    prepared=sum(b.prepare_inference().resident_bytes for b in model.blocks)+model.bias.nbytes+model.decision_shifts.nbytes+4
    return dict(cost_bytes=max(coo,prepared),coo_resident_bytes=coo,prepared_blocks_bytes=prepared,
        parameters=model.parameter_count+1,blocks=len(model.blocks),trees=model.n_trees,
        basis_values=sum(b.base.basis.size for b in model.blocks),
        coefficient_values=sum(b.base.coefficients.size for b in model.blocks),nnz=sum(len(b.values) for b in model.blocks),
        scope='conservative max of packed COO runtime and separately prepared-block runtime; training history excluded from serving')


def load_protected(path):
    return BlockLeafModel.load(path)


def as_protected(model):
    blocks,bias,_=blocks_of(model)
    return BlockLeafModel(tuple(blocks),bias.copy(),np.zeros_like(bias))


def project_new_increment(checkpoint,increment,development,budget,config,basis_estimate=None):
    if set(development)!={'train','calibration','control'}:raise ValueError('only controller roles allowed')
    dense=full_output_base(increment);c=len(dense.bias);rank=max(1,(c-int(dense.task=='multiclass'))//2)
    designs={k:leaf_design(dense.leaf_indices(x),len(dense.coefficients)) for k,(x,_) in development.items()}
    protected={k:checkpoint.raw(x) for k,(x,_) in development.items()}
    contributions=np.asarray(designs['train']@dense.coefficients.astype(float))
    source='functional'
    if basis_estimate is None:
        mean,basis,spectrum=functional_basis(contributions,dense.task)
    else:
        mean=contributions.mean(0);spectrum=None
        basis=basis_estimate.basis;source=basis_estimate.diagnostics
        expected=c-int(dense.task=='multiclass')
        if basis.shape!=(c,expected) or not np.allclose(basis.T@basis,np.eye(expected),atol=2e-5):
            raise ValueError('invalid protected-increment basis')
    base=compress(dense,mean,basis[:,:rank]);zero=np.zeros(c,np.float32)
    pure=ThresholdModel(base,np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32),zero)
    teacher={k:protected[k]+np.asarray(d@dense.coefficients.astype(float))+dense.bias for k,d in designs.items()}
    residual=(contrast(dense.coefficients,dense.task)-contrast(base.coefficients.astype(float)@base.basis.T,dense.task)).astype(np.float32)
    error=contrast(teacher['train']-protected['train']-pure.raw_from_design(designs['train']),dense.task)
    weights=training_weights(development['train'][1],teacher['train'],dense.task,'balanced_mse',config['margin_scale'])
    order=score_order(coordinate_scores(designs['train'],residual,error,'balanced_mse',weights),residual)
    frequency=np.asarray(designs['train'].mean(axis=0)).ravel()
    def variants(raw):
        return calibrated_local_variants(raw['calibration'],development['calibration'][1],
            raw['control'],development['control'][1],dense.task,config['decision_grid'],config['decision_shrinkage'])
    reference=min(variants(teacher),key=lambda q:(-q['validation']['f1_macro'],q['validation']['loss'],q['threshold_fraction']))
    denominator=increment.parameter_count;seen=set();candidates=[];models=[]
    for percent in (0,1,2,5):
        count=min(len(order),percent*denominator//100)
        if count in seen:continue
        seen.add(count);sparse=make_sparse_model(base,residual,order,count,frequency)
        inc=ThresholdModel(sparse.base,sparse.rows,sparse.columns,sparse.values,zero)
        model=append_protected(checkpoint,inc);price=block_cost(model)
        if price['cost_bytes']>budget:continue
        rounding=model.bias.astype(float)-checkpoint.bias.astype(float)-inc.base.bias.astype(float)
        raw={k:protected[k]+inc.raw_from_design(designs[k])+rounding for k in development}
        mse=float(np.sum(weights*contrast(teacher['train']-raw['train'],dense.task)**2)/c)
        for v in variants(raw):
            row=dict(percent=percent,rank=rank,nnz=len(inc.values),denominator=denominator,cost=price,
                train_weighted_mse=mse,**v)
            candidates.append(row);models.append(model)
    eligible=[i for i,q in enumerate(candidates) if quality_guard(q['validation'],reference['validation'],.05)]
    if not eligible:return None,dict(status='no_memory_or_local_quality_candidate',candidates=candidates,reference=reference,basis_source=source)
    index=min(eligible,key=lambda i:(candidates[i]['train_weighted_mse'],-candidates[i]['validation']['f1_macro'],
        candidates[i]['validation']['loss'],candidates[i]['cost']['cost_bytes'],candidates[i]['percent']))
    return models[index],dict(status='accepted',chosen=candidates[index],candidates=candidates,
        reference=reference,basis_source=source,spectrum=spectrum.tolist() if basis_estimate is None else None,
        old_block_fingerprints=[fingerprint(b) for b in checkpoint.blocks])
