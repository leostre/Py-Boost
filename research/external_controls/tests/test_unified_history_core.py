import numpy as np
import pytest
from unified_history_core import observe_branches,replay_moments
from experiments.absolute_history_core import AbsoluteHistory


@pytest.mark.parametrize('task',['multiclass','multilabel'])
@pytest.mark.parametrize('kind',['current','mean','history'])
def test_duplicate_branches_preserve_curvature_scaling_and_moments(task,kind):
    rng=np.random.default_rng(87);n,c=45,5
    labels=np.arange(n)%c if task=='multiclass' else (rng.random((n,c))<.2).astype(int)
    one=AbsoluteHistory(kind,c,task);two=AbsoluteHistory(kind,c,task);frames=[]
    for _ in range(4):
        scores=rng.normal(size=(n,c));a,_,_=observe_branches(one,labels,[scores],1.,2)
        b,joined,_=observe_branches(two,labels,[scores,scores],1.,2)
        np.testing.assert_allclose(a.basis[:,:2]@a.basis[:,:2].T,b.basis[:,:2]@b.basis[:,:2].T,atol=3e-6)
        state=two.state
        frames.append((joined,dict(mean=np.empty((0,0)) if state.mean is None else state.mean.copy(),
            gram=np.empty((0,0)) if state.gram is None else state.gram.copy(),mass=state.mass,basis=b.basis)))
    assert replay_moments(frames,labels,task,kind,2,1.)['frames']==4
