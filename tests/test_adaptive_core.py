import numpy as np
import pytest
from hypothesis import given, strategies as st
from experiments.adaptive_core import affordable_depth, growth_decision, spectral_profile, spectral_state, output_budget


@given(outputs=st.integers(1, 400), remaining=st.integers(0, 1000000),
       preferred=st.integers(1, 7))
def test_affordable_tree_never_exceeds_remaining_cap(outputs, remaining, preferred):
    d = affordable_depth(preferred, remaining, outputs)
    assert 0 <= d <= preferred
    if d:
        assert 2**d * outputs <= remaining
    if d < preferred:
        assert 2**(d + 1) * outputs > remaining


def test_plateau_increases_depth_only_after_warmup():
    args = dict(mode='adaptive', depth=3, losses=[1.] * 41,
                remaining=100000, outputs=100)
    assert growth_decision(step=20, **args).depth == 3
    assert growth_decision(step=40, **args).depth == 5
    assert growth_decision(step=41, **args).depth == 3


def test_budget_stop_and_validation_stop_are_distinct():
    a = growth_decision(mode='fixed', step=10, depth=3, losses=[1.], remaining=199, outputs=100)
    b = growth_decision(mode='adaptive', step=40, depth=7, losses=[1.] * 41,
                        remaining=100000, outputs=100)
    assert a.stop and a.reason == 'parameter_cap'
    assert b.stop and b.reason == 'validation_plateau_at_allowed_depth'


def test_improving_validation_does_not_increase_capacity():
    d = growth_decision(mode='adaptive', step=40, depth=3,
                        losses=np.linspace(2., 1., 41), remaining=100000, outputs=100)
    assert not d.stop and d.depth == 3


def test_spectral_profile_scale_invariance_and_low_rank_control():
    rng = np.random.default_rng(4)
    g = rng.normal(size=(64, 1)) @ rng.normal(size=(1, 10))
    h = np.zeros_like(g)
    p = spectral_profile(g, h)
    np.testing.assert_allclose(p, spectral_profile(g * 17, h), atol=1e-12)
    s = spectral_state(p, 64)
    assert s['energy_rank'] == 1
    assert abs(s['effective_rank'] - 1) < 1e-10
    assert output_budget(10, s, False) == 10
    assert output_budget(10, s, True) == 3


@pytest.mark.parametrize('bad', [np.full((3, 2), np.nan), np.full((3, 2), np.inf)])
def test_nonfinite_profile_is_rejected(bad):
    with pytest.raises(ValueError):
        spectral_profile(bad, np.ones_like(bad))

def test_actual_allowed_depth_not_unapplied_preference_controls_stop():
    d = growth_decision(mode='adaptive', step=40, depth=3, max_depth=5,
                        losses=[1.] * 41, remaining=100000, outputs=100)
    assert not d.stop and d.depth == 5
    d = growth_decision(mode='adaptive', step=60, depth=5, max_depth=5,
                        losses=[1.] * 61, remaining=100000, outputs=100)
    assert d.stop and d.reason == 'validation_plateau_at_allowed_depth'
def test_early_threshold_changes_only_plateau_response():
    values = [1. - .0003*i for i in range(41)]
    common = dict(mode='adaptive', step=40, depth=3, losses=values,
                  remaining=10000, outputs=4)
    assert growth_decision(**common, improvement_floor=.002).depth == 3
    assert growth_decision(**common, improvement_floor=.01).depth == 5

def test_latest_jump_cannot_be_certified_by_small_median():
    from experiments.adaptive_core import stable_profile_window
    assert not stable_profile_window([0., 0., 1.], .01, 60, 4)
    assert stable_profile_window([.01, .02, .01], .02, 60, 4)
    assert not stable_profile_window([0., 0.], .02, 40, 3)
    assert not stable_profile_window([0., 0., 0.], .02, 20, 4)
