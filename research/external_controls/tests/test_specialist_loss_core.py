import numpy as np
import pytest
from specialist_loss_core import object_losses,decompose
from experiments.output_calibration_core import calibrated_quality


def test_exact_existing_loss_and_exhaustive_partition():
    rng=np.random.default_rng(20261021)
    a,b=rng.normal(size=(20,3)),rng.normal(size=(20,3))
    y=np.arange(20,dtype=np.int32)%3
    shift=np.array([0,.2,-.1],np.float32)
    report,x,z=decompose(y,a,b,shift,shift)
    assert report['specialist_mean_loss']==pytest.approx(calibrated_quality(y,a,'multiclass',shift)['loss'],abs=1e-14)
    assert sum(g['count'] for g in report['groups'])==len(y)
    assert sum(g['contribution_to_mean_excess'] for g in report['groups'])==pytest.approx(report['mean_excess'],abs=1e-14)
    assert sum(g['excess_loss_sum'] for g in report['classes'])==pytest.approx((x-z).sum(),abs=1e-14)


def test_decision_shifts_change_groups_but_not_probability_loss():
    a=np.array([[1.,0.],[1.,0.]])
    y=np.array([0,1])
    first,x,_=decompose(y,a,a,np.zeros(2),np.zeros(2))
    second,z,_=decompose(y,a,a,np.array([2.,0.]),np.zeros(2))
    np.testing.assert_array_equal(x,z)
    assert first['groups']!=second['groups']
    assert second['mean_excess']==0


def test_permutation_and_extreme_logits_do_not_change_aggregate():
    a=np.array([[1000.,-1000.],[0.,0.],[-1000.,1000.]])
    b=-a
    y=np.array([0,1,0])
    one,_,_=decompose(y,a,b,[],[])
    two,_,_=decompose(y[::-1],a[::-1],b[::-1],[],[])
    assert one['groups']==two['groups']
    assert np.isfinite(object_losses(a,y)).all()


@pytest.mark.parametrize('target',[np.array([-1,0]),np.array([0,2]),np.array([0.,1.]),np.array([0])])
def test_invalid_label_domains_are_rejected(target):
    with pytest.raises(ValueError): object_losses(np.zeros((2,2)),target)
