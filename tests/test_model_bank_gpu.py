"""Synthetic no-fit multi-model and wide-output correctness, no new dependencies."""
from dataclasses import replace
import sys
import numpy as np
import pytest

from test_model_bank_core import tiny_model
from experiments.model_bank_core import gpu_items, owned_bytes, executor_name


@pytest.mark.parametrize('c,r', [(14,7), (128,64), (129,30), (159,40)])
def test_multiple_gpu_models_and_wide_fallback(c,r):
    if sys.platform != 'linux':
        pytest.skip('GPU checks are authorized only on the registered Linux A100')
    cp = pytest.importorskip('cupy')
    if 'A100' not in cp.cuda.runtime.getDeviceProperties(0)['name'].decode():
        pytest.skip('registered A100 required')
    from experiments.traversal_leaf_gpu import TraversalLeafGpu
    from experiments.unified_leaf_gpu import UnifiedLeafGpu
    from experiments.shared_leaf_basis import reference_class_model
    from experiments.uniform_threshold_core import ThresholdModel
    source = tiny_model(c,r)
    dense = replace(source.base, coefficients=(source.base.coefficients@source.base.basis.T).astype(np.float32),
                    basis=np.empty((0,0),np.float32), representation='dense', task='multiclass')
    full = ThresholdModel(reference_class_model(dense),np.empty(0,np.int32),np.empty(0,np.int32),
                          np.empty(0,np.float32),np.zeros(c,np.float32))
    factory = (lambda m: TraversalLeafGpu(m,'groups')) if executor_name(c)=='groups' else UnifiedLeafGpu
    runtimes = [factory(source), factory(full), factory(source)]
    inputs = np.array([[-1],[0],[1],[np.nan]],np.float32)
    all_arrays = [i for runtime in runtimes for i in gpu_items(runtime)]
    assert owned_bytes(all_arrays) == sum(t.resident_bytes for t in runtimes)
    expected = [source.predict(inputs),full.predict(inputs),source.predict(inputs)]
    for turn in (2,0,1,2,1,0):
        np.testing.assert_allclose(runtimes[turn].predict(inputs), expected[turn], atol=2e-12, rtol=0)
    assert not any(hasattr(t,'source') or hasattr(t,'model') for t in runtimes)
