"""Causal warning decisions and train-only projection under frozen B/r/s."""
from dataclasses import replace
import math
import numpy as np
from experiments.shared_leaf_basis import LeafEnsemble
from experiments.sparse_leaf_core import (leaf_design,contrast,training_weights,coordinate_scores,
                                         score_order,make_sparse_model)
from experiments.uniform_threshold_core import ThresholdModel


def spectral_view(signal,seed):
    z=np.asarray(signal,dtype=float)
    if z.ndim!=2 or len(z)<4 or not np.isfinite(z).all(): raise ValueError('invalid warning signal')
    permutation=np.random.default_rng(seed).permutation(len(z))
    halves=np.array_split(permutation,2)
    grams=[]
    for indices in halves:
        g=z[indices].T@z[indices]
        trace=float(np.trace(g))
        grams.append(g/trace if trace>1e-30 else np.zeros_like(g))
    return (grams[0]+grams[1])/2,float(np.linalg.norm(grams[0]-grams[1],'fro'))


def audit_decision(index,policy,middle_used,ratio,memory_blocked):
    # No quality, oracle risk, labels, or future values in this boundary.
    if isinstance(index,bool) or index not in range(5) or policy not in ('periodic','event'):
        raise ValueError('invalid warning schedule')
    if not math.isfinite(ratio) or ratio<0: raise ValueError('invalid causal drift/noise ratio')
    if index==0: return True,'warmup40'
    if index==4: return True,'final200'
    if middle_used: return False,'hold'
    if memory_blocked: return True,'shared_hard_memory_alarm'
    if policy=='periodic': return (True,'period120') if index==2 else (False,'hold')
    if ratio>=1: return True,'drift_above_simultaneous_noise'
    return (True,'deadline160') if index==3 else (False,'hold')


def hold_budget(dense,train_x,train_y,confirmed,margin_scale=.5):
    """Retain B/r/maximum s; no calibration/control/test capabilities."""
    b=confirmed.base.basis.astype(float)
    if confirmed.base.representation!='low_rank' or dense.task!=confirmed.base.task:
        raise ValueError('incompatible held basis')
    m,c=dense.coefficients.shape
    design=leaf_design(dense.leaf_indices(train_x),m)
    contribution=design@dense.coefficients
    mean=contribution.mean(axis=0)
    a=(dense.coefficients.astype(float)@b).astype(np.float32)
    base=LeafEnsemble(dense.nodes,dense.node_offsets,dense.leaf_offsets,a,b.astype(np.float32),
        (dense.bias+mean-(mean@b)@b.T).astype(np.float32),dense.n_features,dense.task,'low_rank')
    residual=(contrast(dense.coefficients,dense.task)-contrast(a.astype(float)@b.T,dense.task)).astype(np.float32)
    teacher=contrast(contribution+dense.bias,dense.task)
    error=teacher-contrast((design@a)@b.T+base.bias,dense.task)
    weights=training_weights(train_y,teacher,dense.task,'balanced_mse',margin_scale)
    order=score_order(coordinate_scores(design,residual,error,'balanced_mse',weights),residual)
    frequency=np.asarray(design.mean(axis=0)).ravel()
    leaf=make_sparse_model(base,residual,order,min(len(confirmed.values),len(order)),frequency)
    return ThresholdModel(leaf.base,leaf.rows,leaf.columns,leaf.values,confirmed.decision_shifts.copy())
