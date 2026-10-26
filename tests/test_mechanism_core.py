import numpy as np
import pytest
from hypothesis import given, strategies as st
from experiments.mechanism_core import channel_budget, static_budget, uniform_reference_settings
from experiments.mechanism_sketch import IdentityAtFullTopK


@given(n=st.integers(1,400), d=st.integers(3,7), q=st.floats(min_value=.01,max_value=1,allow_nan=False))
def test_channel_disabling_is_independent(n, d, q):
    k = max(1, int(n*q))
    assert channel_budget(n,k,d,'outputs').depth_ceiling == 7
    assert channel_budget(n,k,d,'outputs').topk == k
    assert channel_budget(n,k,d,'depth').topk == n
    assert channel_budget(n,k,d,'depth').depth_ceiling == d
    assert (channel_budget(n,k,d,'joint').topk, channel_budget(n,k,d,'joint').depth_ceiling) == (k,d)


@given(n=st.integers(1,400), step=st.integers(0,59), q=st.sampled_from([.25,.5,1.]), d=st.sampled_from([3,5,7]))
def test_static_controls_share_full_warmup(n, step, q, d):
    b = static_budget(n,step,q,d)
    assert b.topk == n and b.depth_ceiling == 7


@given(n=st.integers(1,400), step=st.integers(60,200), q=st.sampled_from([.25,.5,1.]), d=st.sampled_from([3,5,7]))
def test_static_fraction_ceiling_and_rounding(n, step, q, d):
    b = static_budget(n,step,q,d)
    assert b.depth_ceiling == d and 1 <= b.topk <= n
    assert b.topk-1 < n*q <= b.topk


def test_exact_activation_boundary():
    assert static_budget(10,59,.25,3).topk == 10
    assert static_budget(10,60,.25,3).topk == 3


def reference(**kw):
    r = dict(dataset='yeast',seed=404,method='full',max_depth=5,lambda_l2=1.,
             boosting_steps=200,parameters=1000,candidate_id='fixed',data_sha256='fixeddata')
    return {**r,**kw}


@given(scores=st.dictionaries(st.text(max_size=5),st.floats(allow_nan=True)))
def test_reference_ignores_all_quality_fields(scores):
    expected = uniform_reference_settings([reference()], 'yeast',404)
    assert uniform_reference_settings([reference(train=scores,validation=scores,test=scores)],'yeast',404) == expected


def test_reference_requires_exact_metadata_and_unique_match():
    with pytest.raises(ValueError):
        uniform_reference_settings([reference(lambda_l2=10.)],'yeast',404)
    with pytest.raises(ValueError):
        uniform_reference_settings([reference(),reference()],'yeast',404)
    assert uniform_reference_settings([reference(),reference(max_depth=7,parameters=1)],'yeast',404)['parameters']==1000


def test_full_width_identity_preserves_objects_and_column_order():
    g=np.array([[3.,1.,2.],[0.,2.,1.]])
    h=np.ones_like(g)
    a,b=IdentityAtFullTopK(3)(g,h)
    assert a is g and b is h


def test_reduced_topk_preserves_existing_selection():
    g=np.array([[3.,1.,2.],[0.,2.,1.]])
    h=np.ones_like(g)
    a,b=IdentityAtFullTopK(1)(g,h)
    np.testing.assert_array_equal(a,g[:,[0]])
    np.testing.assert_array_equal(b,h[:,[0]])


class HostArray:
    def __init__(self, data):
        self.data = np.asarray(data)
    def __getitem__(self, idx):
        return HostArray(self.data[idx])
    def get(self):
        return self.data.copy()


def fake_info(step):
    from types import SimpleNamespace
    sketch = IdentityAtFullTopK(5)
    return dict(num_iter=step, model=SimpleNamespace(base_score=np.zeros(10)),
                builder=SimpleNamespace(multioutput_sketch=sketch, params={}),
                data={'train':{'target':np.zeros(100),
                               'grad':HostArray(np.arange(1000).reshape(100,10)),
                               'hess':HostArray(np.ones((100,1)))}})


@pytest.mark.parametrize('channels,expected', [('joint',(3,3)),('outputs',(3,7)),('depth',(10,3))])
def test_actual_callback_applies_only_enabled_channel(monkeypatch, channels, expected):
    import experiments.adaptive_callback as adapter
    monkeypatch.setattr(adapter,'stable_profile_window',lambda *a:True)
    monkeypatch.setattr(adapter,'spectral_state',lambda *a:dict(effective_rank=1.,energy_rank=1,above_noise_edge=0,noise_edge=0.))
    c=adapter.CompactnessController(mode='schedule',spectral='history',parameter_cap=10000,
                                   seed=1,task='multiclass',emit=lambda *a,**k:None,budget_channels=channels)
    info=fake_info(60)
    c.before_train(info)
    assert info['builder'].multioutput_sketch.topk==10
    c.before_iteration(info)
    assert (info['builder'].multioutput_sketch.topk,c.depth_ceiling)==expected
    assert c.confirmed is not None


def test_actual_static_callback_obeys_boundary():
    from experiments.adaptive_callback import CompactnessController
    c=CompactnessController(mode='schedule',spectral='none',parameter_cap=10000,
                           seed=1,task='multiclass',emit=lambda *a,**k:None,
                           static_fraction=.25,static_depth_ceiling=3)
    info=fake_info(59)
    c.before_train(info)
    c.before_iteration(info)
    assert (info['builder'].multioutput_sketch.topk,c.depth_ceiling)==(10,7)
    info['num_iter']=60
    c.before_iteration(info)
    assert (info['builder'].multioutput_sketch.topk,c.depth_ceiling)==(3,3)
