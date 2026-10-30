from dataclasses import replace
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'HASBoost-A100-External/tests'))
import budget_quality_worker
from abl_training_worker import canonical_full,initial_scores,reserve_fit,completed_fit
from test_abl_quality_core import fixture
from budget_quality_core import probabilities


def test_training_initialization_uses_raw_actual_model_and_ignores_output_shifts():
    model,x,_,_=fixture(17,5,6)
    shifted=replace(model,decision_shifts=np.array([9,-7,3,1,-1],np.float32))
    expected=model.prepare_inference().raw(x)
    np.testing.assert_array_equal(initial_scores(shifted,x,5),expected)
    canonical=canonical_full(shifted)
    np.testing.assert_allclose(probabilities(initial_scores(canonical,x,5)),probabilities(expected),atol=2e-7)
    assert canonical.base.representation=='reference_class'
    assert canonical.base.coefficients.shape[1]==4
    np.testing.assert_array_equal(initial_scores(None,x,5),np.zeros((len(x),5)))


def test_interrupted_fit_reservation_cannot_disappear_or_exceed_global_tree_cap(tmp_path):
    path=tmp_path/'ledger.json';path.write_text(json.dumps(dict(reserved_rounds=21560,attempts=[])))
    index=reserve_fit(path,'case','joint',0,3)
    reserved=json.loads(path.read_text());assert reserved['reserved_rounds']==21600
    assert reserved['attempts'][index]['state']=='reserved'
    with pytest.raises(RuntimeError,match='cap exhausted'):reserve_fit(path,'case','joint',1,3)
    completed_fit(path,index,dict(seconds=1.,trees=40),320)
    complete=json.loads(path.read_text());assert complete['reserved_rounds']==21600
    assert complete['attempts'][index]['state']=='complete'
    assert complete['attempts'][index]['fitted_full_output_leaf_values']==320
