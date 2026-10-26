"""Fresh CPU process: compact files must replay without importing a teacher."""
from pathlib import Path
import sys
import numpy as np
from experiments.shared_leaf_basis import LeafEnsemble


def main():
    count = 0
    for path in sorted(Path(sys.argv[1]).glob('*/compact.npz')):
        model = LeafEnsemble.load(path)
        with np.load(path.with_name('replay.npz'), allow_pickle=False) as data:
            np.testing.assert_allclose(model.predict(data['features']), data['probabilities'], atol=1e-12, rtol=0)
        count += 1
    if count != 15 or any(m == 'cupy' or m.startswith('py_boost') for m in sys.modules):
        raise RuntimeError('standalone replay did not cover 15 CPU-only models')
    print(f'Standalone CPU replay passed: {count} compact models', flush=True)


if __name__ == '__main__':
    main()
