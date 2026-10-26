"""Additional generated checks where Hypothesis is already installed.

The frozen server has no Hypothesis; no dependency is installed for this pilot.
Deterministic boundary and oracle tests remain mandatory on the server.
"""
import numpy as np
import pytest

hypothesis=pytest.importorskip('hypothesis')
from hypothesis import given, settings, strategies as st
from calibration_pilot_core import make_plans, Scheme, choose_model


@settings(max_examples=60,deadline=None)
@given(sizes=st.tuples(st.integers(2,30),st.integers(2,30),st.integers(2,30)),
       classes=st.integers(2,15),repeat=st.integers(0,9))
def test_partition_conservation_for_dependent_sizes(sizes,classes,repeat):
    y=np.arange(sum(sizes))%classes
    for p in make_plans(y,sizes,repeat):
        if p.scheme==Scheme.CROSSFIT:
            seen=[i for f in p.folds for i in f.validation]
            assert sorted(seen)==list(range(len(y)))
        for f in p.folds:
            assert not set(f.validation)&(set(f.temperature)|set(f.shifts))


@settings(max_examples=60,deadline=None)
@given(costs=st.lists(st.integers(1,1000),min_size=1,max_size=20),
       lower=st.integers(1,1000),extra=st.integers(0,1000))
def test_relaxing_memory_cap_cannot_worsen_selected_validation_loss(costs,lower,extra):
    rows=[dict(model_id=str(i),cost_bytes=c,validation=dict(loss=(i+1)/len(costs),f1_macro=0.))
          for i,c in enumerate(costs)]
    a=choose_model(rows,'loss',lower);b=choose_model(rows,'loss',lower+extra)
    if a is not None:
        assert b is not None
        assert b['validation']['loss']<=a['validation']['loss']
