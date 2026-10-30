"""Array-only saved-tree pruning and fixed-support archive representations.

No fitting of split thresholds, file access, model training, or GPU imports.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import numpy as np
from experiments.shared_leaf_basis import LeafEnsemble, functional_basis, probabilities
from experiments.uniform_threshold_core import ThresholdModel
from experiments.block_leaf_model import BlockLeafModel
from experiments.sparse_leaf_core import leaf_design


@dataclass(frozen=True)
class DenseView:
    nodes: np.ndarray
    node_offsets: np.ndarray
    leaf_offsets: np.ndarray
    values: np.ndarray
    bias: np.ndarray
    features: int
    task: str

    @property
    def trees(self):
        return len(self.node_offsets)-1

    def base(self, values=None, basis=None, representation='dense'):
        return LeafEnsemble(self.nodes.copy(), self.node_offsets.copy(), self.leaf_offsets.copy(),
            np.asarray(self.values if values is None else values, np.float32),
            np.empty((0, 0), np.float32) if basis is None else np.asarray(basis, np.float32),
            np.zeros(len(self.bias), np.float32), self.features, self.task, representation)

    def design(self, x):
        return leaf_design(self.base().leaf_indices(x), len(self.values))

    def slice(self, begin, end):
        if not 0 <= begin < end <= self.trees:
            raise ValueError('invalid tree interval')
        lo, hi = self.leaf_offsets[[begin, end]]
        a, b = self.node_offsets[[begin, end]]
        return DenseView(self.nodes[a:b].copy(), self.node_offsets[begin:end+1]-a,
            self.leaf_offsets[begin:end+1]-lo, self.values[lo:hi].copy(),
            np.zeros_like(self.bias), self.features, self.task)


def decode_model(model):
    """Decode original factors in float64; only deployment coefficients use FP32."""
    if isinstance(model, BlockLeafModel):
        blocks, bias = model.blocks, model.bias
    else:
        blocks, bias = (model,), model.base.bias
    nodes = []; no = [0]; lo = [0]; values = []; groups = []
    for block in blocks:
        base = block.base; start = len(no)-1
        v = base.coefficients.astype(float)
        if base.representation == 'low_rank':
            v = v@base.basis.astype(float).T
        elif base.representation == 'reference_class':
            v = np.column_stack((v, np.zeros(len(v))))
        if len(block.values):
            v[block.rows, block.columns] += block.values
        for t in range(base.n_trees):
            n = base.nodes[base.node_offsets[t]:base.node_offsets[t+1]].copy()
            w = v[base.leaf_offsets[t]:base.leaf_offsets[t+1]].copy()
            nodes.append(n); values.append(w); no.append(no[-1]+len(n)); lo.append(lo[-1]+len(w))
        groups.append((start, len(no)-1))
    return DenseView(np.concatenate(nodes), np.asarray(no, np.int64), np.asarray(lo, np.int64),
        np.concatenate(values), bias.copy(), blocks[0].base.n_features, blocks[0].base.task), tuple(groups)


def blank(base):
    c = len(base.bias)
    return ThresholdModel(base, np.empty(0, np.int32), np.empty(0, np.int32),
        np.empty(0, np.float32), np.zeros(c, np.float32))


def pack(blocks, bias, shifts=None):
    c = len(bias); zero = np.zeros(c, np.float32)
    normalized = tuple(replace(b, base=replace(b.base, bias=zero), decision_shifts=zero) for b in blocks)
    return BlockLeafModel(normalized, np.asarray(bias, np.float32),
        zero if shifts is None else np.asarray(shifts, np.float32))


def hessian(raw, task):
    p = probabilities(raw, task)
    return p*(1-p)*(p.shape[1]/(p.shape[1]-1) if task == 'multiclass' else 1.)


def fixed_support_projection(target, weights, basis, keys=()):
    """Pinned weighted_projection math with externally selected support.

    Removing supported coordinates can make the normal matrix singular.
    Same symmetric eigen cutoff 1e-12 as the pinned implementation.
    """
    v, d, b = (np.asarray(x, float) for x in (target, weights, basis))
    if v.shape != d.shape or b.shape[0] != v.shape[1] or np.any(d <= 0):
        raise ValueError('positive diagonal and matching target/basis required')
    identity = b.shape == (v.shape[1], v.shape[1]) and np.array_equal(b, np.eye(v.shape[1]))
    if not all(np.isfinite(a).all() for a in (v, d, b)) or (not identity and not np.allclose(b.T@b, np.eye(b.shape[1]), atol=2e-5)):
        raise ValueError('finite orthonormal basis required')
    keys = np.asarray(keys, np.int64)
    if keys.ndim != 1 or len(np.unique(keys)) != len(keys) or (len(keys) and (keys.min() < 0 or keys.max() >= v.size)):
        raise ValueError('invalid support')
    free = d.copy(); free.ravel()[keys] = 0.
    if identity:
        # Same minimum-norm solution and relative eigen cutoff, no C^3 solve.
        cutoff = 1e-12*np.maximum(free.max(axis=1, keepdims=True), np.finfo(float).tiny)
        a = np.where(free > cutoff, v, 0.).astype(np.float32)
        rows, cols = np.unravel_index(keys, v.shape)
        values = (v[rows, cols]-a[rows, cols].astype(float)).astype(np.float32)
        nonzero = values != 0
        return a, rows[nonzero].astype(np.int32), cols[nonzero].astype(np.int32), values[nonzero]
    # Chunking bounds simultaneous eigensolver workspace independently of m.
    arrays = []
    for begin in range(0, len(v), 64):
        w = free[begin:begin+64]; y = v[begin:begin+64]
        lhs = np.einsum('cr,lc,cs->lrs', b, w, b, optimize=True)
        rhs = (w*y)@b
        eigen, vectors = np.linalg.eigh(lhs)
        cutoff = 1e-12*np.maximum(eigen[:, -1:], np.finfo(float).tiny)
        inverse = np.divide(1., eigen, out=np.zeros_like(eigen), where=eigen > cutoff)
        coord = np.einsum('lri,lr->li', vectors, rhs)*inverse
        arrays.append(np.einsum('lri,li->lr', vectors, coord))
    a = np.concatenate(arrays).astype(np.float32)
    rows, cols = np.unravel_index(keys, v.shape)
    values = (v-a.astype(float)@b.T)[rows, cols].astype(np.float32)
    nonzero = values != 0
    return a, rows[nonzero].astype(np.int32), cols[nonzero].astype(np.int32), values[nonzero]


def fit_error(target, weights, basis):
    a, _, _, _ = fixed_support_projection(np.asarray(target)[None], np.asarray(weights)[None], basis)
    return a[0], float(np.sum(weights*(target-a[0].astype(float)@basis.T)**2))


def prune_tree(nodes, targets, weights, basis, leaf_cap):
    """Greedy bottom-up cherry collapse in a fixed diagonal reconstruction metric.

    Original leaf weights are the atoms of the static objective. Their sums
    at a merged node are not a replay of a fresh merged-leaf Newton step.
    """
    if leaf_cap < 2:
        raise ValueError('this native format requires at least two leaves')
    nodes = np.asarray(nodes); v = np.asarray(targets, float); d = np.asarray(weights, float)
    summaries = {}; active = set(); records = []
    def init(pointer):
        if pointer < 0:
            ell = -pointer-1; z = (d[ell].copy(), d[ell]*v[ell], d[ell]*v[ell]**2, [ell])
        else:
            left, right = (init(int(p)) for p in nodes[pointer, 2:])
            z = tuple(left[i]+right[i] for i in range(3))+(left[3]+right[3],)
        summaries[pointer] = z
        if pointer < 0:
            active.add(pointer)
        return z
    init(0)
    @lru_cache(maxsize=None)
    def loss(pointer):
        mass, first, square, _ = summaries[pointer]; target = first/mass
        a, _ = fit_error(target, mass, basis)
        decoded = a.astype(float)@basis.T
        return float(np.sum(square-2*first*decoded+mass*decoded**2))
    while len(active) > leaf_cap:
        choices = []
        for i in range(len(nodes)):
            if i in active:
                continue
            l, r = map(int, nodes[i, 2:])
            if l in active and r in active:
                choices.append((max(0., loss(i)-loss(l)-loss(r)), i, l, r))
        if not choices:
            raise ValueError('invalid pruning frontier')
        increase, i, l, r = min(choices)
        active.remove(l); active.remove(r); active.add(i)
        records.append(dict(node=i, increase=increase))
    new_nodes = []; values = []; masses = []; origin_nodes = []; mapping = np.full(len(v), -1, np.int64)
    def build(pointer):
        if pointer in active:
            mass, first, _, leaves = summaries[pointer]; index = len(values)
            values.append(first/mass); masses.append(mass); mapping[leaves] = index
            return -index-1
        index = len(new_nodes); new_nodes.append(nodes[pointer].copy()); origin_nodes.append(pointer)
        for side in (2, 3):
            new_nodes[index][side] = build(int(nodes[pointer, side]))
        return index
    assert build(0) == 0 and (mapping >= 0).all()
    targets = np.asarray(values); masses = np.asarray(masses)
    coefficients, _, _, _ = fixed_support_projection(targets, masses, basis)
    return dict(nodes=np.asarray(new_nodes, np.float32), target=targets, weights=masses,
        coefficients=coefficients, mapping=mapping, origin_nodes=np.asarray(origin_nodes, np.int64), merges=records)


def reference_basis(warm_contributions):
    """Preserve nested contrast subspaces while storing a reference-class gauge."""
    _, contrast_basis, _ = functional_basis(warm_contributions, 'multiclass')
    differences = contrast_basis[:-1]-contrast_basis[-1:]
    q, r = np.linalg.qr(differences)
    signs = np.where(np.diag(r) >= 0, 1., -1.)
    q = q*signs
    wire = np.vstack((q, np.zeros((1, q.shape[1])))).astype(np.float32)
    return wire, contrast_basis


def bounded_allocate(total, weights, capacities):
    weights = np.asarray(weights, float); caps = np.asarray(capacities, np.int64)
    if total < 0 or total > int(caps.sum()) or np.any(caps < 0) or weights.shape != caps.shape or np.any(weights < 0):
        raise ValueError('invalid bounded quota')
    assigned = np.zeros(len(caps), np.int64); remaining = int(total)
    while remaining:
        room = caps-assigned; w = np.where(room > 0, weights, 0.)
        if w.sum() == 0:
            w = room.astype(float)
        ideal = remaining*w/w.sum()
        saturated = (room > 0) & (ideal >= room)
        if saturated.any():
            # Redistribute a saturated share BEFORE rounding the remainder.
            assigned[saturated] += room[saturated]
            remaining -= int(room[saturated].sum())
            continue
        floor = np.floor(ideal).astype(np.int64)
        assigned += floor; remaining -= int(floor.sum())
        if remaining == 0:
            break
        order = sorted(np.flatnonzero(caps-assigned > 0), key=lambda j:(-(ideal[j]-np.floor(ideal[j])), int(j)))
        for j in order:
            if remaining == 0:
                break
            assigned[j] += 1; remaining -= 1
    return assigned


def support_order(order, outputs, deficits, count):
    order = np.asarray(order, np.int64); deficits = np.asarray(deficits, float)
    if count > len(order) or count < 0 or deficits.shape != (outputs,) or np.any(deficits < 0):
        raise ValueError('invalid deficit selection')
    if not count or deficits.sum() == 0:
        return order[:count].copy(), dict(active=False, quotas=np.bincount(order[:count]%outputs, minlength=outputs).tolist())
    caps = np.bincount(order%outputs, minlength=outputs)
    positive = deficits > 0
    targeted = min(count, int(caps[positive].sum()))
    q = bounded_allocate(targeted, deficits, np.where(positive, caps, 0))
    selected = np.concatenate([order[order%outputs == j][:q[j]] for j in range(outputs)])
    if len(selected) < count:
        selected = np.r_[selected, order[~np.isin(order, selected)][:count-len(selected)]]
    return selected, dict(active=True, quotas=np.bincount(selected%outputs, minlength=outputs).tolist(),
        targeted=int(targeted), fallback=count-int(targeted))


def marginal_price(coo_zero, prepared_zero, leaves, quotas):
    quotas = np.asarray(quotas); leaves = np.asarray(leaves)
    return int(max(coo_zero+12*quotas.sum(), prepared_zero+8*quotas.sum()+4*np.sum((leaves+1)*(quotas > 0))))


def maximum_capacity(budget, coo_zero, prepared_zero, leaves, masses, capacities):
    if max(coo_zero, prepared_zero) > budget:
        return None
    upper = min(int(np.sum(capacities)), (budget-coo_zero)//12)
    # Descending enumeration is exact even at non-monotone Hamilton/CSR edges.
    for total in range(int(upper), -1, -1):
        q = bounded_allocate(total, masses, capacities)
        if marginal_price(coo_zero, prepared_zero, leaves, q) <= budget:
            return int(total)
    raise AssertionError('zero support was feasible')


def label_f1(labels, raw, shifts):
    labels = np.asarray(labels); pred = np.asarray(raw)-np.asarray(shifts) >= 0
    denominator = labels.sum(0)+pred.sum(0)
    return np.divide(2*(labels*pred).sum(0), denominator,
        out=np.zeros(labels.shape[1], float), where=denominator > 0)


def deficit(quality, reference):
    if min(reference['loss'], reference['f1_macro'], reference['f1_weighted']) <= 0:
        raise ValueError('positive fixed reference required')
    return float(max(0., quality['loss']/(1.05*reference['loss'])-1,
        1-quality['f1_macro']/(.95*reference['f1_macro']),
        1-quality['f1_weighted']/(.95*reference['f1_weighted'])))


def select_point(rows, role, reference, budget, require_control=False):
    candidates = [r for r in rows if r['cost_bytes'] <= budget and
        (not require_control or r['control_guard'])]
    if not candidates:
        return None
    d = min(deficit(r[role], reference) for r in candidates)
    candidates = [r for r in candidates if deficit(r[role], reference) <= d+1e-12]
    if role == 'selection':
        price = min(r['cost_bytes'] for r in candidates)
        candidates = [r for r in candidates if r['cost_bytes'] == price]
    loss = min(r[role]['loss'] for r in candidates)
    candidates = [r for r in candidates if r[role]['loss'] <= loss+1e-6]
    if role == 'control':
        return min(candidates, key=lambda r:(r['cost_bytes'], r.get('leaves', 0), r['uid']))
    return min(candidates, key=lambda r:(-r[role]['f1_macro'], r['cost_bytes'], r['uid']))
