"""Rebuild saved current/mean/Gram states from actual train-only trajectories."""
import numpy as np
from budget_quality_worker import locked,digest,native_load
from abl_protected_core import load_protected
from abl_transfer_study import data_before_test
from experiments.parallel_leaf_core import prefix
from scipy.special import expit,softmax


def independent_signal(y,f,task,regularization):
    f=np.asarray(f,dtype=np.float64)
    p=np.clip(softmax(f,axis=1),1e-6,1-1e-6) if task=='multiclass' else np.clip(expit(f),1e-7,1-1e-7)
    truth=np.eye(f.shape[1])[y] if task=='multiclass' else y
    h=p*(1-p)*(f.shape[1]/(f.shape[1]-1) if task=='multiclass' else 1)
    z=-(p-truth)/np.sqrt(h+regularization/len(y))
    if task=='multiclass':z-=z.mean(1,keepdims=True)
    norm=np.linalg.norm(z)
    return z/norm if norm else np.zeros_like(z)


def audit_history_states(entries):
    checks=[]
    for entry in entries:
        case=locked(entry['path'],entry['sha256']);data=data_before_test(case['record']);task=case['task'];y=data['train_y'];x=data['train_x']
        warm=locked(case['warmup']['path'],case['warmup']['sha256']);teacher=native_load(warm['warmup_threshold']['artifact'])
        for reference in case['trajectories']:
            kind=reference['mode']
            if kind not in ('current','mean','history'):continue
            t=locked(reference['path'],reference['sha256']);mass=0.;mean=None;gram=None;observations=0
            def observe(f):
                nonlocal mass,mean,gram,observations
                z=independent_signal(y,f,task,1.)
                if kind=='current':gram=z.T@z;mass=1.
                elif kind=='mean':mean=.8*(np.zeros_like(z) if mean is None else mean)+.2*z;mass=.8*mass+.2
                else:gram=.8*(np.zeros((z.shape[1],z.shape[1])) if gram is None else gram)+.2*(z.T@z);mass=.8*mass+.2
                observations+=1
            for trees in (0,10,20,30,40):
                f=np.broadcast_to(teacher.base.bias,(len(y),len(teacher.base.bias))).copy() if not trees else prefix(teacher,trees).prepare_inference().raw(x)
                observe(f)
            count=0
            for cycle in t['cycles']:
                if cycle['block']>1:
                    previous=next(m for m in t['models'] if m['block']==cycle['block']-1)
                    observe(load_protected(previous['artifact']).raw(x))
                path=cycle['history_state_path'];assert digest(path)==cycle['history_state_sha256']
                with np.load(path,allow_pickle=False) as f:
                    assert int(f['observations'])==observations
                    assert abs(float(f['mass'])-mass)<1e-14
                    np.testing.assert_allclose(f['mean'],np.empty((0,0)) if mean is None else mean,atol=2e-13,rtol=2e-12)
                    np.testing.assert_allclose(f['gram'],np.empty((0,0)) if gram is None else gram,atol=2e-13,rtol=2e-12)
                    b=f['basis'].copy()
                g=(mean/mass).T@(mean/mass) if kind=='mean' else gram/mass
                c=len(g);q=np.linalg.qr(np.vstack((np.eye(c-1),-np.ones(c-1))))[0] if task=='multiclass' else np.eye(c)
                spectrum=np.linalg.eigvalsh(q.T@g@q)[::-1]
                np.testing.assert_allclose(np.diag(b.T@g@b),spectrum,atol=2e-7,rtol=2e-6)
                assert cycle['algorithm_state_bytes']==(mean.nbytes if kind=='mean' else gram.nbytes)
                count+=1
            checks.append(dict(case=case['case_id'],kind=kind,snapshots=count,status='passed'))
    return dict(status='passed',checks=checks,test_accessed=False,gpu_execution=False,
        scope='independent sigmoid/softmax derivatives, aligned temporal first/second moments, saved basis Rayleigh spectrum and algorithm bytes')
