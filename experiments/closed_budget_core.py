"""Pure resource plans; no predictions, labels, training, or hidden fallbacks."""
from dataclasses import dataclass, replace
import math

import numpy as np

from experiments.sparse_leaf_core import Cost


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError('invalid '+name)
    return int(value)


@dataclass(frozen=True)
class Topology:
    trees: int
    internal_nodes: int
    leaves: int

    def __post_init__(self):
        for name in ('trees', 'internal_nodes', 'leaves'):
            integer(getattr(self, name), name)
        if (self.trees == 0) != (self.leaves == 0 and self.internal_nodes == 0):
            raise ValueError('inconsistent empty topology')
        if self.trees and (self.leaves < self.trees or self.internal_nodes < self.trees):
            raise ValueError('each exported tree needs a node and a leaf')


@dataclass(frozen=True)
class Allocation:
    rank: int
    nnz: int
    cap_percent: int
    cost: Cost


def compact_cost(topology, outputs, rank, nnz):
    c = integer(outputs, 'outputs', 1)
    r = integer(rank, 'rank', 1)
    s = integer(nnz, 'nnz')
    if r > c or s > topology.leaves*c:
        raise ValueError('impossible compact dimensions')
    return Cost((topology.leaves+c)*r+s+2*c,
                16*topology.internal_nodes+16*(topology.trees+1)+4*(topology.leaves+c)*r+12*s+8*c)


def rank_limit(topology, outputs, budget, nnz=0, ceiling=None):
    c = integer(outputs, 'outputs', 1)
    s = integer(nnz, 'nnz')
    if s > topology.leaves*c:
        return 0
    maximum = c if ceiling is None else min(c, integer(ceiling, 'ceiling', 1))
    if budget.parameters is not None:
        maximum = min(maximum, (budget.parameters-s-2*c)//(topology.leaves+c))
    if budget.payload_bytes is not None:
        fixed = 16*topology.internal_nodes+16*(topology.trees+1)+12*s+8*c
        maximum = min(maximum, (budget.payload_bytes-fixed)//(4*(topology.leaves+c)))
    return max(0, maximum)


def extend_bound(topology, trees, depth):
    t = integer(trees, 'trees')
    d = integer(depth, 'depth', 1)
    if d > 16:
        raise ValueError('unbounded tree depth')
    return Topology(topology.trees+t, topology.internal_nodes+t*(2**d-1), topology.leaves+t*2**d)


def plan_depth(topology, remaining_trees, outputs, rank, nnz, budget):
    """Return a depth with space reserved through the final horizon, or None."""
    for depth in (5, 4, 3):
        future = extend_bound(topology, remaining_trees, depth)
        if budget.allows(compact_cost(future, outputs, rank, min(nnz, future.leaves*outputs))):
            return depth
    return None


def bounded_allocations(topology, outputs, budget, reference_parameters, ceiling, *, fixed=False):
    reference_parameters = integer(reference_parameters, 'reference_parameters', 1)
    c = integer(outputs, 'outputs', 1)
    maximum = rank_limit(topology, c, budget, ceiling=ceiling)
    if not maximum:
        return ()
    if fixed:
        rank = max(1, ceiling//2)
        if rank > maximum:
            return ()
        ranks = [rank]
    else:
        ranks = sorted({max(1, math.floor(maximum*f)) for f in (.5, .75, 1.)})
    result = []
    for rank in ranks:
        base = compact_cost(topology, c, rank, 0)
        limit = topology.leaves*c
        if budget.parameters is not None:
            limit = min(limit, budget.parameters-base.parameters)
        if budget.payload_bytes is not None:
            limit = min(limit, (budget.payload_bytes-base.payload_bytes)//12)
        percents = (5,) if fixed else (0, 1, 2, 5)
        seen = set()
        for percent in percents:
            count = min(limit, percent*reference_parameters//100)
            if count in seen:
                continue
            seen.add(count)
            result.append(Allocation(rank, count, percent, compact_cost(topology, c, rank, count)))
    return tuple(result)


def independent_snapshot(model):
    """Rejected proposals cannot mutate accepted array ownership through views."""
    base = replace(model.base, **{k: getattr(model.base, k).copy() for k in
        ('nodes', 'node_offsets', 'leaf_offsets', 'coefficients', 'basis', 'bias')})
    return replace(model, base=base, **{k: getattr(model, k).copy() for k in
        ('rows', 'columns', 'values', 'decision_shifts')})
