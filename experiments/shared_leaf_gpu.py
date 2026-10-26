"""Standalone GPU inference; reuse only the native tree routing kernel."""
from types import SimpleNamespace


def device_features(features, n_features):
    """Transfer host input explicitly before CuPy-only layout normalization."""
    import cupy as cp
    x = cp.ascontiguousarray(cp.asarray(features, dtype=cp.float32))
    if x.ndim != 2 or x.shape[1] != n_features:
        raise ValueError('invalid feature shape')
    return x


class GpuLeafEnsemble:
    def __init__(self, ensemble):
        import cupy as cp
        self.task = ensemble.task
        self.representation = ensemble.representation
        self.n_features = ensemble.n_features
        self.trees = [SimpleNamespace(
            ngroups=1,
            test_format=cp.asarray(ensemble.nodes[a:b].ravel()),
            test_format_offsets=cp.array([0], dtype=cp.int32))
            for a, b in zip(ensemble.node_offsets[:-1], ensemble.node_offsets[1:])]
        self.leaf_offsets = tuple(int(x) for x in ensemble.leaf_offsets[:-1])
        self.coefficients = cp.asarray(ensemble.coefficients)
        self.basis = cp.asarray(ensemble.basis)
        self.bias = cp.asarray(ensemble.bias)

    def predict(self, features):
        import cupy as cp
        from py_boost.gpu.tree import Tree
        x = device_features(features, self.n_features)
        latent = cp.zeros((len(x), self.coefficients.shape[1]), dtype=cp.float32)
        leaves = cp.empty((len(x), 1), dtype=cp.int32)
        for tree, offset in zip(self.trees, self.leaf_offsets):
            Tree.predict_leaf(tree, x, leaves)
            latent += self.coefficients[leaves[:, 0] + offset]
        if self.representation == 'low_rank':
            raw = latent @ self.basis.T + self.bias
        elif self.representation == 'reference_class':
            raw = cp.concatenate((latent, cp.zeros((len(x), 1), dtype=cp.float32)), axis=1) + self.bias
        else:
            raw = latent + self.bias
        if self.task == 'multiclass':
            out = cp.exp(raw - raw.max(axis=1, keepdims=True))
            return cp.clip(out / out.sum(axis=1, keepdims=True), 1e-6, 1 - 1e-6).get()
        return cp.clip(1 / (1 + cp.exp(-raw)), 1e-7, 1 - 1e-7).get()
