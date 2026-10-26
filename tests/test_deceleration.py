"""Geometric forecast laws: reference sums, inverse, monotonicity and causality."""
import math
import pytest
from hypothesis import given, strategies as st
from experiments.risk_budget_core import projected_gain, required_trees, plan_depth, WindowGain


@given(st.floats(.0001, 1, allow_nan=False, allow_infinity=False),
       st.floats(.001, .999999, allow_nan=False, allow_infinity=False),
       st.integers(1, 200))
def test_matches_independent_discrete_sum(rate, rho, n):
    a = rho ** (1/20)
    expected = rate * math.fsum(a ** k for k in range(1, n+1))
    assert projected_gain(rate, rho, n) == pytest.approx(expected, rel=1e-9)


@given(st.floats(.0001, 1, allow_nan=False, allow_infinity=False),
       st.floats(.5, .999999, allow_nan=False, allow_infinity=False),
       st.integers(1, 200))
def test_inverse_where_distance_to_asymptote_is_resolvable(rate, rho, n):
    # Here a**n >= .5**10: inverse is well-conditioned in float64.
    a = rho ** (1/20)
    expected = rate * math.fsum(a ** k for k in range(1, n+1))
    assert required_trees(rate, rho, expected) == pytest.approx(n, rel=1e-8)


@given(st.floats(.001, 1, allow_nan=False, allow_infinity=False),
       st.floats(0, 1, allow_nan=False, allow_infinity=False),
       st.integers(0, 200))
def test_never_exceeds_constant_and_increases_with_trees(rate, rho, n):
    g = projected_gain(rate, rho, n)
    assert 0 <= g <= rate*n + 1e-12
    assert projected_gain(rate, rho, n+1) >= g - 1e-12


def test_boundary_and_unreachable_cases():
    assert projected_gain(.1, 1, 20) == 2
    assert required_trees(.1, 1, 2) == 20
    assert required_trees(.1, 0, .01) is None
    assert required_trees(0, 1, .01) is None
    assert required_trees(0, 0, 0) == 0
    rho = .5
    limit = .01 / math.expm1(-math.log(rho)/20)
    assert required_trees(.01, rho, limit) is None
    assert required_trees(.01, rho, limit*1.01) is None
    assert required_trees(.01, rho, limit*.99) is not None
    almost_constant = math.nextafter(1., 0.)
    assert projected_gain(.01, almost_constant, 200) == pytest.approx(2.)


@pytest.mark.parametrize('rate,rho', [(.1,.5),(1.,.001),(.01,.99)])
def test_exact_asymptote_is_not_reachable_in_finite_time(rate,rho):
    limit = rate / math.expm1(-math.log(rho)/20)
    assert required_trees(rate,rho,limit) is None


def decision(windows, **kw):
    return plan_depth(step=140, remaining_parameters=10000, remaining_trees=60,
                      current_deficit=.1, windows=windows, history=2, incumbent=5, **kw)


def windows(old=.2, new=.1):
    return [WindowGain(3,60,80,1000,.5,.5-old), WindowGain(5,80,100,3000,.5,.4),
            WindowGain(3,100,120,1000,.3,.3-new), WindowGain(5,120,140,3000,.3,.2)]


def test_observed_decay_and_cost_conversion():
    a = decision(windows(), deceleration=True).forecasts[0]
    b = decision(windows()).forecasts[0]
    assert a.block_decay == pytest.approx(.5)
    assert a.required_trees > b.required_trees
    assert a.required_parameters == pytest.approx(a.required_trees*50)


def test_recent_negative_gain_can_remove_positive_projection():
    a = decision(windows(.2,-.05), deceleration=True).forecasts[0]
    assert a.tree_rate > 0 and a.block_decay == 0
    assert a.required_trees is None and a.score is None and not a.feasible


def test_no_decay_if_old_nonpositive_or_recent_accelerates():
    for old, new in ((-.05,.2),(.1,.2)):
        a = decision(windows(old,new), deceleration=True).forecasts[0]
        b = decision(windows(old,new)).forecasts[0]
        assert a.block_decay == 1
        assert a.required_trees == pytest.approx(b.required_trees)


def test_single_window_fallback_and_original_unchanged():
    ws=windows()[:2]
    a=decision(ws,deceleration=True)
    b=decision(ws)
    for x,y in zip(a.forecasts,b.forecasts):
        assert x.block_decay == 1
        assert x.required_parameters == pytest.approx(y.required_parameters)
        assert x.required_trees == pytest.approx(y.required_trees)


def test_future_windows_rejected_and_order_does_not_matter():
    ws=windows()
    assert decision(ws,deceleration=True) == decision(list(reversed(ws)),deceleration=True)
    with pytest.raises(ValueError, match='future'):
        decision(ws+[WindowGain(3,140,160,1000,.1,.05)],deceleration=True)
    with pytest.raises(ValueError, match='two-window'):
        plan_depth(step=140, remaining_parameters=1,remaining_trees=1,current_deficit=.1,
                   windows=ws,history=1,incumbent=5,deceleration=True)


@pytest.mark.parametrize('args', [(-1,1,20),(.1,-.1,20),(.1,1.1,20),(.1,.5,-1),
                                 (float('nan'),1,20),(.1,1,float('inf'))])
def test_invalid_projection(args):
    with pytest.raises(ValueError): projected_gain(*args)
