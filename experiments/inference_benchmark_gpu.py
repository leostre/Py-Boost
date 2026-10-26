"""Unoptimized GPU port of the current design/COO inference, not a new method.

Keep stored coefficients float32 and current CPU design/accumulation float64.
Construct both CSR design and residual per request, as in the CPU reference.
"""
from types import SimpleNamespace


class GpuThresholdReference:
    def __init__(self, model):
        import cupy as cp
        self.task, self.representation = model.base.task, model.base.representation
        self.n_features = model.base.n_features
        self.trees = [SimpleNamespace(ngroups=1,
            test_format=cp.asarray(model.base.nodes[a:b].ravel()),
            test_format_offsets=cp.array([0], dtype=cp.int32))
            for a, b in zip(model.base.node_offsets[:-1], model.base.node_offsets[1:])]
        self.offsets = [int(v) for v in model.base.leaf_offsets[:-1]]
        self.coefficients, self.basis, self.bias = [cp.asarray(a) for a in
            (model.base.coefficients, model.base.basis, model.base.bias)]
        self.rows, self.columns, self.values, self.shifts = [cp.asarray(a) for a in
            (model.rows, model.columns, model.values, model.decision_shifts)]

    def route(self, x):
        import cupy as cp
        from py_boost.gpu.tree import Tree
        if x.dtype != cp.float32 or x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError('invalid resident features')
        leaves = cp.empty((len(self.trees), len(x)), dtype=cp.int32)
        scratch = cp.empty((len(x), 1), dtype=cp.int32)
        for t, (tree, offset) in enumerate(zip(self.trees, self.offsets)):
            Tree.predict_leaf(tree, x, scratch)
            leaves[t] = scratch[:, 0] + offset
        return leaves

    def design(self, leaves):
        import cupy as cp
        from cupyx.scipy import sparse
        n = leaves.shape[1]
        return sparse.coo_matrix((cp.ones(leaves.size, dtype=cp.float64),
            (cp.tile(cp.arange(n, dtype=cp.int32), len(leaves)), leaves.ravel())),
            shape=(n, len(self.coefficients))).tocsr()

    def raw(self, design):
        import cupy as cp
        from cupyx.scipy import sparse
        raw = design @ self.coefficients.astype(cp.float64)
        if self.representation == 'low_rank':
            raw = raw @ self.basis.T
        elif self.representation == 'reference_class':
            raw = cp.column_stack((raw, cp.zeros(len(raw))))
        raw = raw + self.bias
        if len(self.values):
            residual = sparse.coo_matrix((self.values, (self.rows, self.columns)),
                shape=(len(self.coefficients), len(self.bias))).tocsr()
            raw += (design @ residual.astype(cp.float64)).toarray()
        return raw

    def probability(self, raw):
        import cupy as cp
        if self.task == 'multiclass':
            p = cp.exp(raw-raw.max(axis=1, keepdims=True))
            return cp.clip(p/p.sum(axis=1, keepdims=True), 1e-6, 1-1e-6)
        z = cp.exp(-cp.abs(raw))
        return cp.clip(cp.where(raw >= 0, 1/(1+z), z/(1+z)), 1e-7, 1-1e-7)

    def predict_device(self, x):
        return self.probability(self.raw(self.design(self.route(x))))

    def predict(self, x):
        import cupy as cp
        return self.predict_device(cp.asarray(x, dtype=cp.float32)).get()
