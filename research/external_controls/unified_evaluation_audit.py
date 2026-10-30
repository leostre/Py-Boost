"""Model-family-independent replay of frozen OOF controls and metrics."""
import numpy as np
from scipy.special import softmax,expit
from sklearn.metrics import f1_score
from budget_quality_worker import digest


def audit_evaluations(raw,y,task,evaluations):
    refs={e['evaluation']['path']:e['evaluation'] for e in evaluations};checks=0
    for path,ref in refs.items():
        assert digest(path)==ref['sha256'];states=ref['states']
        ids=np.concatenate([s['validation'] for s in states]);assert sorted(ids.tolist())==list(range(len(y)))
        for s in states:
            a,b=set(s['fitting']),set(s['validation']);assert not a&b and a|b==set(range(len(y)))
        with np.load(path,allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['validation'],ids)
            for name,q in ref['scores'].items():
                predictions=[];loss=[];brier=[]
                for s in states:
                    idx=np.asarray(s['validation']);control=s['policies'][name];z=raw[idx]/control['temperature'];target=y[idx]
                    p=np.clip(softmax(z,axis=1) if task=='multiclass' else expit(z),1e-7,1-1e-7)
                    predictions.append((z-control['shifts']).argmax(1) if task=='multiclass' else z>=control['shifts'])
                    if task=='multiclass':
                        loss.extend(-np.log(p[np.arange(len(idx)),target.astype(int)]))
                        brier.extend(np.sum((p-np.eye(raw.shape[1])[target.astype(int)])**2,axis=1))
                    else:
                        assert control['temperature']==1.;loss.extend(np.mean(-target*np.log(p)-(1-target)*np.log1p(-p),axis=1));brier.extend(np.mean((p-target)**2,axis=1))
                pred=np.concatenate(predictions);np.testing.assert_array_equal(pred,saved[name+'__prediction'])
                np.testing.assert_allclose(loss,saved[name+'__loss'],atol=2e-12);np.testing.assert_allclose(brier,saved[name+'__brier'],atol=2e-12)
                kwargs={'labels':np.arange(raw.shape[1])} if task=='multiclass' else {}
                computed=dict(loss=np.mean(loss),brier=np.mean(brier),f1_macro=f1_score(y[ids],pred,average='macro',zero_division=0,**kwargs),f1_weighted=f1_score(y[ids],pred,average='weighted',zero_division=0,**kwargs))
                for key,value in computed.items():assert abs(q[key]-value)<1e-10
                checks+=1
    return dict(status='passed',metric_policy_checks=checks,scope='independent OOF array/metric replay; model forward provenance separate')
