"""Generated causal and budget invariants for the pure controller."""
import math
import pytest
from hypothesis import given, strategies as st
from experiments.risk_budget_core import Quality, WindowGain, deficit, plan_depth


def plan(windows=(), **kw):
    args = dict(step=100, remaining_parameters=10000, remaining_trees=100,
                current_deficit=.1, windows=windows, history=2, incumbent=5)
    return plan_depth(**(args | kw))


def w(depth, end, cost, a=.5, b=.4):
    return WindowGain(depth,end-20,end,cost,a,b)


@given(st.floats(.001,100), st.floats(0,1), st.floats(0,1))
def test_reference_satisfies_own_target(loss, macro, weighted):
    q = Quality(loss,macro,weighted)
    assert deficit(q,q) == 0


@given(st.floats(.001,100), st.floats(.001,1), st.floats(.001,1), st.floats(1.06,3))
def test_loss_failure_cannot_be_hidden_by_f1(loss, macro, weighted, factor):
    assert deficit(Quality(loss*factor,1,1),Quality(loss,macro,weighted)) > 0


def test_zero_reference_f1_is_vacuous():
    assert deficit(Quality(1,0,0), Quality(1,0,0)) == 0


@pytest.mark.parametrize('loss,macro,weighted',[(0,.1,.1),(float('nan'),.1,.1),
    (1,-.1,.5),(1,.5,1.1),(1,.5,float('inf'))])
def test_invalid_quality_rejected(loss,macro,weighted):
    with pytest.raises(ValueError): Quality(loss,macro,weighted)


@given(st.integers(1,10000), st.integers(1,200))
def test_initial_probes_do_not_depend_on_budget_score(p,t):
    assert plan(step=60,remaining_parameters=p,remaining_trees=t).depth == 3
    assert plan(step=80,remaining_parameters=p,remaining_trees=t).depth == 5


def test_future_information_rejected():
    with pytest.raises(ValueError): plan([w(3,120,100)])


def test_signed_history_not_optimistically_clipped_per_window():
    d = plan([w(3,80,100,.5,.3), w(3,100,100,.3,.6), w(5,100,400,.5,.4)])
    assert d.forecasts[0].parameter_rate == 0
    assert d.depth == 5


def test_history_is_not_the_same_as_latest():
    windows = [w(3,80,100,.5,.1), w(3,100,100,.1,.2), w(5,100,100,.5,.4)]
    assert plan(windows,history=2).depth == 3
    assert plan(windows,history=1).depth == 5


def test_feasibility_beats_cheaper_but_too_slow_forecast():
    d = plan([w(3,80,10,.5,.49), w(5,100,500,.5,.3)], remaining_trees=50)
    assert d.depth == 5
    assert not d.forecasts[0].feasible and d.forecasts[1].feasible


def test_stale_refresh_is_labelled_not_adaptive():
    d = plan([w(3,80,100),w(5,140,400)],step=140)
    assert d.depth == 3 and d.reason == 'scheduled_refresh'


def test_small_improvement_is_held():
    d = plan([w(3,80,98),w(5,100,100)])
    assert d.depth == 5 and d.reason == 'hysteresis_hold'


@given(st.floats(.001,100),st.integers(1,10000),st.integers(1,10000))
def test_cheaper_equal_tree_gain_selected_unless_hysteresis(gain,c3,c5):
    d = plan([w(3,80,c3,gain,0),w(5,100,c5,gain,0)],remaining_parameters=10**12,
             remaining_trees=10**12,current_deficit=gain)
    assert d.depth == (3 if c3 < .95*c5 else 5)
    assert all(math.isfinite(f.score) for f in d.forecasts if f.score is not None)


def test_absent_gain_never_serializes_infinity():
    d = plan([w(3,80,100,.4,.5),w(5,100,100,.4,.4)])
    assert d.reason == 'no_positive_gain_hold'
    assert all(f.required_parameters is None and f.required_trees is None for f in d.forecasts)


@given(st.permutations([w(3,80,100),w(5,100,400)]))
def test_window_order_does_not_change_decision(windows):
    assert plan(windows) == plan(sorted(windows,key=lambda x:x.end))
