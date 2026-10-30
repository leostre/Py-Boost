"""Before/after observer on the SAME saved full trajectory; no new tree fit."""
from pathlib import Path
import gc,json,sys
import numpy as np
from scipy.special import softmax,expit
from budget_quality_worker import locked,digest,write,event,native_load
from abl_quality_worker import check_native_environment
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected,as_protected
from unified_training_core import candidate_grid,merge_increment_views
from unified_training import evaluate,serialize,quality_row
from unified_frontier_core import make_folds
from unified_recalibration import audit_case
from experiments.absolute_history_core import AbsoluteHistory
from experiments.sparse_leaf_core import contrast


def scores(ref):
    assert digest(ref['path'])==ref['sha256']
    with np.load(ref['path'],allow_pickle=False) as f:return f['scores'].copy()


def diagnose_case(cfg,case,directory):
    directory.mkdir(parents=True,exist_ok=True);data=data_before_test(case['record']);task=case['task'];c=case['record']['case']['outputs'];rank=max(1,(c-int(task=='multiclass'))//2)
    frames=next(p['history_states'] for p in case['paths'] if p['method']=='one_full')
    histories={kind:AbsoluteHistory(kind,c,task) for kind in ('current','mean','history')};estimates={}
    for frame in frames[:5]:
        z=scores(frame)
        for kind,h in histories.items():estimates[kind]=h.observe(data['train_y'],z,cfg['lambda_l2'],rank)
    folds=make_folds(data['dev_y'],task,case['record']['case']['seed']+20261028);models=[]
    features={'train':data['train_x'],'development':data['dev_x']};labels={'train':data['train_y']}
    full=sorted((m for m in case['models'] if m['family']=='one_full'),key=lambda m:m['block'])
    for source in full:
        block=source['block'];checkpoint=load_protected(source['initial_checkpoint']);fit=source['fits'][0]
        assert digest(fit['artifact'])==fit['sha256'];increment,_=merge_increment_views([native_load(fit['artifact'])],[1.])
        teacher=load_protected(source['artifact']);target=teacher.raw(data['train_x']);post=scores(frames[5+block])
        probability=lambda z:softmax(z,axis=1) if task=='multiclass' else expit(z)
        np.testing.assert_allclose(probability(target),probability(post),atol=2e-5,rtol=0)
        before={kind:e.basis.copy() for kind,e in estimates.items()}
        for kind,h in histories.items():estimates[kind]=h.observe(data['train_y'],post,cfg['lambda_l2'],rank)
        bases=[(phase,kind,basis) for phase,collection in (('before',before),('after',{kind:e.basis for kind,e in estimates.items()})) for kind,basis in collection.items()]
        bases.append(('increment','functional',None))
        delta=contrast(as_protected(increment).raw(data['train_x']),task);centered=delta-delta.mean(0);denom=max(float(np.sum(centered**2)),1e-30)
        for phase,kind,basis in bases:
            proposals=list(candidate_grid(checkpoint,increment,features,labels,None,protected=True,basis=basis,ranks=(rank,),caps=(8,),residuals=(.02,)))
            assert len(proposals)==1;candidate=proposals[0];model=candidate['model'];p=directory/f'b{block}-{phase}-{kind}';p.mkdir(exist_ok=True)
            ev=evaluate(candidate['raw']['development'],data['dev_y'],task,folds,cfg,p)
            b=model.blocks[-1].base.basis.astype(float);error=centered-centered@b@b.T
            actual_error=contrast(candidate['raw']['train']-target,task)
            models.append(serialize(model,candidate['raw']['development'],ev,p,f'b{block}-{phase}-{kind}',dict(family=f'{phase}_{kind}',block=block,rank=rank,phase=phase,kind=kind,
                source_proposal=source['uid'],history_frames_before=5+block,history_frames_after=6+block,
                source_full_quality=quality_row(source['evaluations'],task)['quality'],
                centered_increment_basis_error_ratio=float(np.sum(error**2)/denom),relative_actual_compression_mse=float(np.sum(actual_error**2)/denom))))
            del proposals,candidate;gc.collect()
    result=dict(case=case['case'],task=task,record=case['record'],models=models,audit=audit_case(dict(task=task,models=models),data),
        interpretation='same uncompressed prefixes and proposals; local representation diagnosis, not a new compact-feedback training trajectory')
    path=directory/'case.json';write(path,result);event('unified_history_diagnostic_case',case=case['case'],models=len(models),audit=result['audit'])
    return dict(case=case['case'],path=str(path),sha256=digest(path),audit=result['audit'])


def run(cfg,out):
    environment=check_native_environment(cfg);spec=cfg['unified_history_diagnostic'];source=locked(spec['source']['path'],spec['source']['sha256'])
    assert source['status']=='unified_history_complete';entries=[]
    for cid in spec['cases']:
        ref=next(r for r in source['cases'] if r['case']==cid);case=locked(ref['path'],ref['sha256'])
        assert case['audit']['status']=='passed';entries.append(diagnose_case(cfg,case,out/cid))
    count=sum(e['audit']['models'] for e in entries);assert count<=spec['maximum_representations']
    result=dict(status='unified_history_diagnostic_complete',cases=entries,models=count,new_tree_fits=0,test_evaluated=False,environment=environment,
        interpretation='exploratory mechanism check after U3F; independent online pre-step policy requires a later separate training test')
    write(out/'report.json',result);event('unified_history_diagnostic_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
