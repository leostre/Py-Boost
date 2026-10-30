import numpy as np
import pytest
from broad_core import grouped_folds,assess
from broad_final import oof_audit,independent_metrics,constant_diagnostics,decision_grid_coverage
from budget_quality_worker import digest


def test_independent_metric_definition_with_absent_label():
    y=np.array([[1,0,0],[1,1,0],[0,0,0]]);p=np.array([[1,0,0],[0,1,0],[0,1,0]])
    q=independent_metrics(y,p,np.array([.1,.2,.3]),np.array([0.,.1,.2]))
    assert q['f1_macro']==pytest.approx(4/9) and q['f1_weighted']==pytest.approx(2/3)


def test_constant_control_uses_train_probabilities_and_fixed_decisions():
    train=np.array([[1,0],[0,1]]);test=np.array([[1,0],[0,0]])
    q=constant_diagnostics(train,test)
    assert q['all_positive']['f1_macro']==pytest.approx(1/3)
    assert q['all_negative']['f1_macro']==0
    assert all(r['loss']==pytest.approx(np.log(2)) and r['brier']==.25 for r in q.values())


def test_grid_diagnostic_distinguishes_unsupported_label_from_unreachable_true_positive():
    z=np.array([[-5.,0.,-9.],[-4.,-1.,-8.]])
    y=np.array([[1,1,0],[0,0,0]])
    q=decision_grid_coverage(z,y,[-2,0,2])
    assert q['D_zero_support_labels']==1 and q['D_positive_labels']==2
    assert q['positive_labels_without_any_reachable_TP']==1
    assert q['positive_labels_without_any_positive_prediction']==1


def test_audit_rejects_corrupt_cached_predictions_even_with_updated_file_hash(tmp_path):
    rng=np.random.default_rng(23);z=rng.normal(size=(31,4));y=rng.integers(0,2,z.shape);x=rng.normal(size=(31,2))
    assessed,arrays=assess(z,y,grouped_folds(x,71),[0,.5,-.5],[.5,1])
    raw=tmp_path/'raw.npz';cache=tmp_path/'oof.npz';np.savez(raw,raw=z);np.savez(cache,prediction=arrays[0],loss=arrays[1],brier=arrays[2])
    m=dict(raw=dict(path=str(raw),sha256=digest(raw)),oof=dict(path=str(cache),sha256=digest(cache)),assessment=assessed)
    assert oof_audit(m,y)<1e-12
    pred=arrays[0].copy();pred[0,0]=~pred[0,0];np.savez(cache,prediction=pred,loss=arrays[1],brier=arrays[2]);m['oof']['sha256']=digest(cache)
    with pytest.raises(AssertionError):oof_audit(m,y)
