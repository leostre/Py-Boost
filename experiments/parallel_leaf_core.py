"""Pure continuation checkpoint, averaging and common functional subspace."""
from dataclasses import replace
import numpy as np
from experiments.block_leaf_model import BlockLeafModel
from experiments.online_leaf_core import dense_part,project_part,stitch
from experiments.shared_leaf_basis import functional_basis
from experiments.gradient_basis_core import projector_overlap
from experiments.history_basis_study import raw,task


def prefix(model,count):
    if not 1 <= count <= model.base.n_trees:
        raise ValueError('invalid prefix tree count')
    n,m = int(model.base.node_offsets[count]),int(model.base.leaf_offsets[count])
    keep = model.rows < m
    base = replace(model.base,nodes=model.base.nodes[:n].copy(),node_offsets=model.base.node_offsets[:count+1].copy(),
        leaf_offsets=model.base.leaf_offsets[:count+1].copy(),coefficients=model.base.coefficients[:m].copy())
    return replace(model,base=base,rows=model.rows[keep].copy(),columns=model.columns[keep].copy(),
        values=model.values[keep].copy(),decision_shifts=np.zeros_like(model.decision_shifts))


def blocks_of(model):
    if isinstance(model,BlockLeafModel):
        return model.blocks,model.bias,model.n_features
    zero = np.zeros_like(model.base.bias)
    local = replace(model,base=replace(model.base,bias=zero),decision_shifts=zero)
    return (local,),model.base.bias,model.base.n_features


def dense_parts(model):
    blocks,_,_ = blocks_of(model)
    parts = []
    for block in blocks:
        b = block.base
        values = b.coefficients.astype(float)
        if b.representation == 'low_rank':
            values = values@b.basis.T
            np.add.at(values,(block.rows,block.columns),block.values)
        elif b.representation == 'reference_class':
            values = np.column_stack((values,np.zeros(len(values))))
        for i in range(b.n_trees):
            parts.append(dense_part(b.nodes[b.node_offsets[i]:b.node_offsets[i+1]],
                                    values[b.leaf_offsets[i]:b.leaf_offsets[i+1]].astype(np.float32)))
    return parts


def average_increments(checkpoint,increments):
    if not increments:
        raise ValueError('no increments')
    old,bias,features = blocks_of(checkpoint)
    blocks = list(old)
    zero = np.zeros_like(bias)
    for increment in increments:
        if (np.any(increment.base.bias != 0) or np.any(increment.decision_shifts != 0)
                or increment.base.task != task(checkpoint) or increment.base.n_features != features):
            raise ValueError('invalid zero-bias branch increment')
        parts = dense_parts(increment)
        parts = [replace(p,coefficients=(p.coefficients/len(increments)).astype(np.float32)) for p in parts]
        blocks.append(stitch(parts,None,zero,features,task(checkpoint),zero))
    return BlockLeafModel(tuple(blocks),bias.copy(),zero)


def recompress(merged,candidates,features,rank,residual_fraction):
    """Union functional PCA; local visitation-weighted leaf projection.

    Reconstructs individual trees, not a global optimum of ensemble error.
    Native tree/output values are only transient during this synchronization.
    """
    if not candidates:
        raise ValueError('no functional candidates')
    matrices = [raw(m,features) for m in candidates]
    centered = [m-m.mean(axis=0,keepdims=True) for m in matrices]
    full_basis = functional_basis(np.concatenate(centered,axis=0),task(merged))[1]
    if not 1 <= rank <= full_basis.shape[1]:
        raise ValueError('invalid merge rank')
    basis = full_basis[:,:rank].astype(np.float32)
    parts = dense_parts(merged)
    _,bias,nfeats = blocks_of(merged)
    before = raw(merged,features)
    dense = stitch(parts,None,bias,nfeats,task(merged))
    leaves = dense.base.leaf_indices(features)
    projected,offset = [],0
    for part,ids in zip(parts,leaves):
        count = np.bincount(ids-offset,minlength=len(part.coefficients)).astype(float)+1.
        weights = np.broadcast_to(count[:,None],part.coefficients.shape)
        projected.append(project_part(part,weights,basis,residual_fraction))
        offset += len(part.coefficients)
    compact = stitch(projected,basis,bias,nfeats,task(merged))
    uncorrected = raw(compact,features)
    correction = (before-uncorrected).mean(axis=0)
    compact = replace(compact,base=replace(compact.base,bias=(bias+correction).astype(np.float32)))
    after = raw(compact,features)
    candidate_bases = [functional_basis(a,task(merged))[1][:,:rank].astype(np.float32) for a in centered]
    overlap = projector_overlap(*candidate_bases) if len(candidate_bases) == 2 else None
    denominator = float(np.sum((before-before.mean(axis=0))**2))
    return compact,dict(rank=rank,branches=len(candidates),branch_subspace_overlap=overlap,
        candidate_common_overlap=[projector_overlap(a,basis) for a in candidate_bases],
        relative_prediction_squared_error=float(np.sum((after-before)**2)/max(denominator,1e-30)),
        mean_preservation_max_error=float(np.max(np.abs(after.mean(axis=0)-before.mean(axis=0)))),
        parameters_before=merged.parameter_count,parameters_after=compact.parameter_count,
        payload_before=merged.payload_bytes,payload_after=compact.payload_bytes)
