from dataclasses import replace
import numpy as np
import pytest
from test_abl_quality_core import fixture
from abl_protected_core import as_protected
from saved_serving_core import load_saved_serving

@pytest.mark.parametrize('block',[False,True])
def test_legacy_and_shared_block_archives_keep_function_and_controls(block,tmp_path):
    model,x,_,_=fixture(8,4,3);model=replace(model,decision_shifts=np.array([.2,-.3,.1,.8],np.float32))
    if block:model=replace(as_protected(model),decision_shifts=model.decision_shifts.copy())
    p=tmp_path/'model.npz';model.save(p);loaded=load_saved_serving(p)
    np.testing.assert_array_equal(loaded.decision_shifts,model.decision_shifts)
    expected=model.raw(x) if block else model.prepare_inference().raw(x)
    np.testing.assert_allclose(loaded.raw(x),expected,atol=1e-12)
