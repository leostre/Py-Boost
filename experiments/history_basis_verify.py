"""CPU-only multibasis replay with no training or CuPy imports."""
import json
from pathlib import Path
import sys
import numpy as np
from experiments.block_leaf_model import load_prediction_model


def main():
    directory = Path(sys.argv[1])
    selection = json.loads((directory/'selection.json').read_text())
    errors = []
    with np.load(directory/'replay.npz',allow_pickle=False) as f:
        for choice in selection['choices']:
            model = load_prediction_model(directory/Path(choice['artifact']).name)
            actual = model.predict(f['features'])
            np.testing.assert_allclose(actual,f[choice['method']],rtol=0,atol=2e-14)
            np.testing.assert_array_equal(model.predict_labels(f['features']),f[choice['method']+'_labels'])
            errors.append(float(np.max(np.abs(actual-f[choice['method']]))))
    if 'cupy' in sys.modules or 'experiments.history_leaf_boosting' in sys.modules:
        raise RuntimeError('standalone verifier imported training/GPU runtime')
    print(json.dumps(dict(event='history_cpu_replay_passed',dataset=selection['dataset'],seed=selection['seed'],
        models=len(errors),maximum_probability_error=max(errors))),flush=True)


if __name__ == '__main__':
    main()
