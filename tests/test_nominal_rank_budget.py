from fractions import Fraction
import numpy as np
import pytest
from hypothesis import given,settings,strategies as st
from experiments.online_leaf_core import residual_rank_allowance


@given(st.integers(1,1000),st.decimals(min_value=0,max_value=1,places=4,allow_nan=False,allow_infinity=False))
@settings(max_examples=80,deadline=None)
def test_nominal_rank_budget_matches_independent_exact_rational_arithmetic(outputs,fraction):
    exact = 3*Fraction(str(fraction))*outputs
    expected = (exact.numerator+exact.denominator-1)//exact.denominator
    assert residual_rank_allowance(float(fraction),outputs)==expected


def test_integer_boundary_does_not_gain_a_rank_from_binary_roundoff():
    assert np.ceil(3*.05*100)==16  # identifies the historical boundary case
    assert residual_rank_allowance(.05,100)==15
    assert residual_rank_allowance(.05,101)==16
    assert residual_rank_allowance(.05,14)==3
    for bad in (float('nan'),float('inf'),-.1,1.1):
        with pytest.raises(ValueError):
            residual_rank_allowance(bad,100)


def test_nominal_hundred_output_rank_control_gpu_matches_posthoc_prefix():
    pytest.importorskip('cupy')
    from experiments.a100_runner import build_model
    from experiments.online_leaf_boosting import CompactLeafBoosting
    from experiments.online_leaf_study import posthoc
    rng = np.random.default_rng(179)
    x = rng.normal(size=(600,6)).astype(np.float32)
    y = np.arange(600,dtype=np.int32)%100
    config = dict(ntrees=4,lr=.1,max_depth=2,lambda_l2=1.,online_warmup=2,
                  online_rank_fraction=.5,online_residual_fraction=.05)
    args = build_model('full','multiclass',100,config,717,2,1.).params
    full = CompactLeafBoosting(**args,task='multiclass',compact_mode='full',warmup=2)
    full.fit(x,y)
    snapshot = dict(parts=full.parts,weights=full.full_weights,bias=full.study_bias,
        basis=full.warm_basis_full,warm_mean=full.warm_train_mean,task='multiclass')
    offline = posthoc(snapshot,x,'more_rank',config)
    online = CompactLeafBoosting(**args,task='multiclass',compact_mode='more_rank',warmup=2)
    online.fit(x,y)
    exported = online.export()
    assert exported.base.coefficients.shape[1]==offline.base.coefficients.shape[1]==64
    np.testing.assert_array_equal(exported.base.basis,offline.base.basis)
    end = int(exported.base.leaf_offsets[2])
    np.testing.assert_array_equal(exported.base.coefficients[:end],offline.base.coefficients[:end])
