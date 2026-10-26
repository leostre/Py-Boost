"""CPU-only reload of sealed optimization artifacts, no training classes."""
import json
from pathlib import Path
import sys
import numpy as np
from experiments.uniform_threshold_core import ThresholdModel


def main():
    directory = Path(sys.argv[1])
    selection = json.loads((directory/'selection.json').read_text())
    errors = []
    with np.load(directory/'replay.npz',allow_pickle=False) as f:
        for choice in selection['choices']:
            model = ThresholdModel.load(directory/Path(choice['artifact']).name)
            actual = model.predict(f['features'])
            np.testing.assert_allclose(actual,f[choice['method']],rtol=0,atol=2e-14)
            np.testing.assert_array_equal(model.predict_labels(f['features']),f[choice['method']+'_labels'])
            errors.append(float(np.max(np.abs(actual-f[choice['method']]))))
    if 'cupy' in sys.modules or 'experiments.online_leaf_boosting' in sys.modules:
        raise RuntimeError('standalone verifier imported GPU/training runtime')
    print(json.dumps(dict(event='online_cpu_replay_passed',dataset=selection['dataset'],seed=selection['seed'],
                         models=len(errors),maximum_probability_error=max(errors))),flush=True)


if __name__ == '__main__':
    main()
