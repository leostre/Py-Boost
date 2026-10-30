"""Sealed broad test replay, CPU/GPU checks, independent OOF metric audit."""
from pathlib import Path
from time import perf_counter
import json,sys,gc,hashlib
import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment,memory_guard
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected
from residual_gate_core import sufficient,quality
from packed_serving_core import PackedPlan
from packed_serving_gpu import PackedGpu
from adaptive_basis_core import risk_basis,risk_weights
from broad_core import select_cases


def independent_metrics(y,pred,loss,brier):
    return dict(f1_macro=float(f1_score(y,pred,average='macro',zero_division=0)),
                f1_weighted=float(f1_score(y,pred,average='weighted',zero_division=0)),
                loss=float(np.mean(loss)),brier=float(np.mean(brier)))


def constant_diagnostics(train_y,test_y):
    prior=np.clip(np.asarray(train_y,float).mean(0),1e-7,1-1e-7);y=np.asarray(test_y)
    p=np.broadcast_to(prior,y.shape);loss=np.mean(-y*np.log(p)-(1-y)*np.log1p(-p),axis=1);brier=np.mean((p-y)**2,axis=1)
    return {name:independent_metrics(y,pred,loss,brier) for name,pred in
            [('train_prior_raw',p>=.5),('all_positive',np.ones_like(y,bool)),('all_negative',np.zeros_like(y,bool))]}


def decision_grid_coverage(raw,y,grid):
    raw=np.asarray(raw,float);positive=np.asarray(y).astype(bool);support=positive.sum(0);minimum=float(min(grid))
    maximum_positive=np.where(positive,raw,-np.inf).max(0)
    return dict(minimum_threshold=minimum,D_positive_labels=int((support>0).sum()),D_zero_support_labels=int((support==0).sum()),
        positive_labels_without_any_reachable_TP=int(((support>0)&(maximum_positive<minimum)).sum()),
        positive_labels_without_any_positive_prediction=int(((support>0)&(raw.max(0)<minimum)).sum()),
        scope='D-only diagnostic of fixed T1 grid; no new threshold fit or selection query')


def oof_audit(m,y):
    assert digest(m['raw']['path'])==m['raw']['sha256'] and digest(m['oof']['path'])==m['oof']['sha256']
    with np.load(m['raw']['path'],allow_pickle=False) as f:raw=f['raw'].copy()
    with np.load(m['oof']['path'],allow_pickle=False) as f:pred,loss,brier=[f[k].copy() for k in ('prediction','loss','brier')]
    replay=np.zeros_like(pred);cover=[];policy=m['assessment']['policy']
    for state in m['assessment']['states']:
        a,b=np.array(state['fitting']),np.array(state['validation'])
        assert not set(a)&set(b) and set(a)|set(b)==set(range(len(y))) and len(set(a))==len(a) and len(set(b))==len(b)
        assert state['policies'][policy]['temperature']==1.
        cover.extend(b)
        replay[b]=raw[b]>=np.asarray(state['policies'][policy]['shifts'])
    assert sorted(cover)==list(range(len(y))) and np.array_equal(replay,pred)
    p=np.clip(expit(raw),1e-7,1-1e-7);expected_loss=np.mean(-y*np.log(p)-(1-y)*np.log1p(-p),axis=1)
    np.testing.assert_allclose(loss,expected_loss,atol=1e-13,rtol=1e-13)
    np.testing.assert_allclose(brier,np.mean((p-y)**2,axis=1),atol=1e-13,rtol=1e-13)
    q=independent_metrics(y,pred,loss,brier);error=max(abs(q[k]-m['assessment']['quality'][k]) for k in q)
    assert error<1e-12
    return error


def run(cfg,out):
    import cupy as cp
    started=perf_counter();out.mkdir(parents=True,exist_ok=True);spec=cfg['broad_final'];environment=check_native_environment(cfg)
    training=locked(spec['training']['path'],spec['training']['sha256']);assert training['status']=='broad_training_complete'
    serving=locked(spec['serving']['path'],spec['serving']['sha256']);assert serving['status']=='wide_tuning_complete'
    seal=locked(training['pretest_seal']['path'],training['pretest_seal']['sha256'])
    cases=[locked(r['path'],r['sha256']) for r in training['cases']]
    assert select_cases(cases)['selected']==seal['selected']
    write(out/'input_seal.json',dict(training=spec['training'],serving=spec['serving'],selection_sha256=training['pretest_seal']['sha256'],test_opened=False))
    cp.get_default_memory_pool().set_limit(size=12*1024**3);entries=[];all_rows=[];audits=[]
    for case in cases:
        cid=case['case'];record=case['record'];data=data_before_test(record);cc=record['case'];dd=record['dataset'];directory=out/cid;directory.mkdir(exist_ok=True)
        assert digest(dd['path'])==dd['sha256'] and digest(cc['split_file'])==cc['split_sha256']
        with np.load(cc['split_file'],allow_pickle=False) as f:test_ids=f['test'].copy()
        with np.load(dd['path'],allow_pickle=False) as f:x=f['features'][test_ids].astype(np.float32);y=f['target'][test_ids].copy()
        constants=constant_diagnostics(data['train_y'],y)
        model_rows=[];max_gpu_error=0.;max_metric_error=0.;by_sha={};profile=None
        for index,m in enumerate(case['models']):
            max_metric_error=max(max_metric_error,oof_audit(m,data['dev_y']))
            if m['sha256'] in by_sha:
                row=dict(by_sha[m['sha256']],uid=m['uid'],family=m['family'],reused_from=by_sha[m['sha256']]['uid']);model_rows.append(row);continue
            assert digest(m['artifact'])==m['sha256'];model=load_protected(m['artifact']);c=len(model.bias)
            has=any(len(b.values) for b in model.blocks);width=max(b.base.coefficients.shape[1] for b in model.blocks)
            groups=4
            try:PackedPlan(groups,'aos',c,width,has)
            except ValueError:groups=2
            control=m['assessment']['refit_control'];device=PackedGpu(model,control['temperature'],control['shifts'],groups,'aos')
            raw=np.empty((len(x),c),float)
            for start in range(0,len(x),256):
                xx=cp.asarray(np.ascontiguousarray(x[start:start+256]));buffers=device.buffers(len(xx));device.launch_into(xx,*buffers,raw=True);raw[start:start+len(xx)]=buffers[0].get();del xx,buffers
            sample=np.linspace(0,len(x)-1,min(17,len(x))).astype(int);err=float(np.max(np.abs(raw[sample]-model.raw(x[sample]))));assert err<1e-8
            max_gpu_error=max(max_gpu_error,err)
            pred,loss,brier=sufficient(y,raw,'multilabel',control);q=quality(y,pred,loss,brier,'multilabel',c);ind=independent_metrics(y,pred,loss,brier)
            assert max(abs(q[k]-ind[k]) for k in q)<1e-12
            raw_pred=raw>=0;raw_q=independent_metrics(y,raw_pred,loss,brier)
            tp=(pred & y.astype(bool)).sum(0);support=y.sum(0);positive=pred.sum(0);den=support+positive
            f1=np.divide(2.*tp,den,out=np.zeros(c),where=den>0)
            file=directory/f'm{index:03d}-predictions.npz';np.savez_compressed(file,prediction=pred,loss=loss,brier=brier,tp=tp,support=support,predicted_positive=positive,f1=f1)
            row=dict(case=cid,dataset=cc['dataset'],seed=cc['seed'],uid=m['uid'],family=m['family'],quality=q,raw_quality=raw_q,
                     cost_bytes=m['cost_bytes'],packed_device_array_bytes=device.resident_device_array_bytes,trees=model.n_trees,
                     leaves=sum(len(b.base.coefficients) for b in model.blocks),nodes=sum(len(b.base.nodes) for b in model.blocks),blocks=len(model.blocks),rank=m.get('rank'),
                     residual_fraction=m.get('residual_fraction'),nnz=m['nnz'],parameters=m['parameters'],basis_values=m['basis_values'],
                     train_zero_support_labels=cc['train_zero_support_labels'],test_zero_support_labels=int((support==0).sum()),
                     f1_macro_test_active=float(f1[support>0].mean()) if np.any(support>0) else None,
                     predictions=dict(path=str(file),sha256=digest(file)),control=control,D_quality=m['assessment']['quality'],raw_gpu_cpu_error=err)
            model_rows.append(row);by_sha[m['sha256']]=row
            if m['uid']=='ordinary/d3/t200':
                # All profiling uses train only, even though this is the final replay phase.
                z=model.raw(data['train_x']);rw,cw=risk_weights(data['train_y'],z,'multilabel');_,geometry=risk_basis(z-model.bias,'multilabel',rw,cw,min(64,c))
                trace=geometry.get('spectral_trace',sum(geometry['weighted_eigenvalues']));eigen=np.asarray(geometry['weighted_eigenvalues']);energy=np.cumsum(eigen)/max(trace,1e-30)
                profile=dict(meta=cc['train_meta_profile'],geometry=geometry,rank90=(int(np.searchsorted(energy,.9))+1 if energy[-1]>=.9 else None),
                             rank90_lower_bound=(len(eigen)+1 if energy[-1]<.9 else None),top64_energy_fraction=float(energy[-1]))
                with np.load(m['raw']['path'],allow_pickle=False) as f:
                    profile['decision_grid_coverage']=decision_grid_coverage(f['raw'],data['dev_y'],cfg['decision_grid'])
            if index%24==0:event('broad_final_model',case=cid,index=index,total=len(case['models']))
            del device,model,raw;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        case_file=directory/'metrics.json';write(case_file,dict(case=cid,models=model_rows,train_profile=profile,record=record,
            diagnostic_baselines=constants,diagnostic_scope='fixed constant decisions, train-prior probabilities; no D/test fitted parameter and no method selection use'))
        entries.append(dict(case=cid,path=str(case_file),sha256=digest(case_file)));all_rows.extend(model_rows)
        audit=dict(case=cid,models=len(model_rows),maximum_cpu_gpu_error=max_gpu_error,maximum_OOF_metric_error=max_metric_error,status='passed');audits.append(audit)
        write(out/'progress.json',dict(cases=entries,audits=audits));event('broad_final_case',**audit);memory_guard()
    lookup={(r['case'],r['uid']):r for r in all_rows};points=[]
    for selected in seal['selected']:
        row=dict(selected)
        if row['uid'] is not None:row['model']=lookup[row['case'],row['uid']]
        points.append(row)
    write(out/'selected_points.json',dict(points=points,selection_source=training['pretest_seal']))
    report=dict(status='broad_final_complete',cases=entries,selected_points=dict(path=str(out/'selected_points.json'),sha256=digest(out/'selected_points.json')),
                audits=audits,paid_new_trees=training['paid_new_trees'],environment=environment,source_training=spec['training'],
                source_serving=spec['serving'],new_trees=0,elapsed_seconds=perf_counter()-started,
                interpretation='old8 exploratory; new4 first frozen test; three train/D partitions share one test, not36 independent datasets')
    write(out/'report.json',report);event('broad_final_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
