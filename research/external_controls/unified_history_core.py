"""Aligned temporal/branch history with an independent moment replay."""
import numpy as np
import budget_quality_worker
from experiments.absolute_history_core import AbsoluteHistory


def observe_branches(history,labels,scores,regularization,rank):
    if not scores or any(s.shape!=scores[0].shape for s in scores):raise ValueError('aligned branches required')
    branches=len(scores);y=np.asarray(labels)
    joined=np.concatenate(scores)
    target=np.tile(y,branches) if y.ndim==1 else np.tile(y,(branches,1))
    # Preserve lambda/n per original object despite stacking branch replicas.
    estimate=history.observe(target,joined,regularization*branches,rank)
    return estimate,joined,target


def independent_signal(labels,scores,task,regularization):
    from scipy.special import expit,softmax
    y=np.asarray(labels);f=np.asarray(scores,float);c=f.shape[1]
    if task=='multiclass':
        p=np.clip(softmax(f,axis=1),1e-6,1-1e-6)
        g=p-np.eye(c)[y];h=p*(1-p)*(c/(c-1))
    else:
        p=np.clip(expit(f),1e-7,1-1e-7);g=p-y;h=p*(1-p)
    z=-g/np.sqrt(h+regularization/len(f))
    if task=='multiclass':z-=z.mean(1,keepdims=True)
    norm=np.linalg.norm(z)
    return z/norm if norm else np.zeros_like(z)


def replay_moments(frames,labels,task,kind,branches,regularization,beta=.8):
    """No AbsoluteHistory/GradientBasisState calls in this independent audit."""
    mass=0.;mean=None;gram=None;checks=0
    target=np.tile(labels,branches) if labels.ndim==1 else np.tile(labels,(branches,1))
    for scores,state in frames:
        z=independent_signal(target,scores,task,regularization*branches)
        if kind=='current':gram=z.T@z;mass=1.
        elif kind=='mean':
            mean=(np.zeros_like(z) if mean is None else beta*mean)+(1-beta)*z
            mass=beta*mass+(1-beta)
        elif kind=='history':
            gram=(np.zeros((z.shape[1],z.shape[1])) if gram is None else beta*gram)+(1-beta)*(z.T@z)
            mass=beta*mass+(1-beta)
        else:raise ValueError('unknown moments')
        expected=(mean/mass).T@(mean/mass) if kind=='mean' else gram/mass
        actual=state['mean'].T@state['mean']/mass**2 if kind=='mean' else state['gram']/mass
        np.testing.assert_allclose(actual,expected,atol=3e-12,rtol=3e-10)
        assert abs(float(state['mass'])-mass)<1e-14
        b=state['basis'];np.testing.assert_allclose(b.T@b,np.eye(b.shape[1]),atol=3e-5)
        if task=='multiclass':np.testing.assert_allclose(b.sum(0),0,atol=3e-5)
        checks+=1
    return dict(status='passed',frames=checks,kind=kind,branches=branches)
