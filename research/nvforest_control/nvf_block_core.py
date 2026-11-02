"""Exact block coalescing and finite model-bank allocation; no GPU import."""
from dataclasses import replace
import numpy as np
from experiments.block_leaf_model import BlockLeafModel


def summary(values):
    return dict(zip(('p50_ms','p95_ms','p99_ms'),map(float,np.percentile(values,[50,95,99]))))


def quality_key(model,task):
    objective='loss' if task=='multiclass' else 'f1_macro'
    q=next(e['quality'] for e in model['evaluations'] if e['objective']==objective)
    return (q['loss'],-q['f1_macro'],model['cost_bytes']) if task=='multiclass' else (-q['f1_macro'],q['loss'],model['cost_bytes'])


def compatible(a,b):
    return (a.base.representation==b.base.representation and a.base.task==b.base.task
        and a.base.n_features==b.base.n_features and a.base.coefficients.shape[1]==b.base.coefficients.shape[1]
        and np.array_equal(a.base.basis,b.base.basis))


def join_blocks(blocks):
    first=blocks[0];noff=[];loff=[];rows=[];n=l=0
    for b in blocks:
        if not compatible(first,b):raise ValueError('decoders differ')
        noff.extend(b.base.node_offsets[:-1]+n);loff.extend(b.base.leaf_offsets[:-1]+l)
        rows.append(b.rows+l);n+=len(b.base.nodes);l+=len(b.base.coefficients)
    base=replace(first.base,nodes=np.concatenate([b.base.nodes for b in blocks]),
        node_offsets=np.asarray(noff+[n],np.int64),leaf_offsets=np.asarray(loff+[l],np.int64),
        coefficients=np.concatenate([b.base.coefficients for b in blocks]))
    return replace(first,base=base,rows=np.concatenate(rows).astype(np.int32),
        columns=np.concatenate([b.columns for b in blocks]),values=np.concatenate([b.values for b in blocks]))


def coalesce(model):
    groups=[]
    for b in model.blocks:
        if groups and compatible(groups[-1][0],b):groups[-1].append(b)
        else:groups.append([b])
    return BlockLeafModel(tuple(join_blocks(g) for g in groups),model.bias.copy(),model.decision_shifts.copy())
