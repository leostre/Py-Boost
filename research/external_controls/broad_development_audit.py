"""Independent saved-D audit; no GPU, tree fitter or test-label slicing."""
from pathlib import Path
from time import perf_counter
import json,sys,gc
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_transfer_study import data_before_test
from abl_quality_worker import check_native_environment
from abl_protected_core import load_protected
from broad_final import oof_audit
from experiments.output_calibration_core import fit_decision_shifts


def profile_saved(cfg,out):
    import cProfile,pstats,io
    from budget_quality_worker import native_load
    from abl_protected_core import append_protected
    from adaptive_controller_core import Action,candidates,diagnose,local_actions
    from broad_core import ranks_for
    spec=cfg['broad_audit'];case=locked(spec['case']['path'],spec['case']['sha256']);data=data_before_test(case['record'])
    by={m['uid']:m for m in case['models']};steps=[s for s in case['traces'][0]['steps'] if 'uid' in s];assert len(steps)>=2
    previous=by[steps[-2]['uid']];last=by[steps[-1]['uid']];checkpoint=load_protected(previous['artifact'])
    assert digest(last['fit']['artifact'])==last['fit']['sha256'];inc=native_load(last['fit']['artifact'])
    rank,cap,residual=previous['rank'],previous['leaf_cap'],previous['residual_fraction'];c=case['record']['case']['outputs']
    features=dict(train=data['train_x'],development=data['dev_x']);labels={'train':data['train_y']}
    def compute():
        probes=list(candidates(checkpoint,inc,features,labels,[Action(rank,8,0.,'keep'),Action(rank,4,0.,'keep')],
                               'risk_residual',max_basis_rank=min(c,64)))
        teacher=append_protected(checkpoint,inc);raw=teacher.raw(data['train_x'])
        diag=diagnose(raw-probes[0]['raw']['train'],raw-probes[1]['raw']['train'])
        b=checkpoint.blocks[0].base.basis if len(checkpoint.blocks)==1 and checkpoint.blocks[0].base.representation=='low_rank' else None
        actions=local_actions(rank,cap,residual,ranks_for(c),diag)
        pool=list(candidates(checkpoint,inc,features,labels,actions,'risk_residual',b,min(c,64)))
        return dict(actions=[a.uid for a in actions],candidates=len(pool),diagnostic=diag)
    prof=cProfile.Profile();started=perf_counter();result=prof.runcall(compute);elapsed=perf_counter()-started
    stream=io.StringIO();stats=pstats.Stats(prof,stream=stream).sort_stats('cumulative');stats.print_stats(35)
    out.mkdir(parents=True,exist_ok=True);(out/'profile.txt').write_text(stream.getvalue(),encoding='utf-8')
    top=[dict(file=f[0],line=f[1],function=f[2],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for f,v in
         sorted(stats.stats.items(),key=lambda item:-item[1][3])[:35]]
    report=dict(status='saved_cpu_profile_complete',case=case['case'],outputs=c,train_rows=len(data['train_x']),elapsed_seconds=elapsed,
                result=result,top=top,new_trees=0,test_evaluated=False,environment=check_native_environment(cfg),
                scope='one saved final-block candidate pass; excludes calibration/artifact writing; profiler overhead included; no runtime superiority claim')
    write(out/'report.json',report);event('saved_cpu_profile_complete',seconds=elapsed,top=top[:8]);return


def run(cfg,out):
    if cfg['broad_audit'].get('operation')=='profile':return profile_saved(cfg,out)
    started=perf_counter();out.mkdir(parents=True,exist_ok=True);spec=cfg['broad_audit'];environment=check_native_environment(cfg)
    snapshot=locked(Path(__file__).parent/spec['snapshot'],spec['snapshot_sha256']);cases=[];total_models=0;total_fits=0
    for ref in snapshot['cases']:
        case=locked(ref['path'],ref['sha256']);data=data_before_test(case['record']);by={m['uid']:m for m in case['models']}
        max_replay=max_oof=0.;fits={};threshold_families=set();checkpoint_checks=[]
        for m in case['models']:
            assert digest(m['artifact'])==m['sha256'];model=load_protected(m['artifact']);max_oof=max(max_oof,oof_audit(m,data['dev_y']))
            assert model.n_trees==m['trees']
            with np.load(m['raw']['path'],allow_pickle=False) as f:raw=f['raw'].copy()
            ids=np.linspace(0,len(raw)-1,min(17,len(raw))).astype(int);error=float(np.max(np.abs(raw[ids]-model.raw(data['dev_x'][ids]))));assert error<2e-5;max_replay=max(max_replay,error)
            if m['family'] not in threshold_families:
                threshold_families.add(m['family']);control=m['assessment']['refit_control'];name=m['assessment']['policy'];fraction=0. if name=='raw' else float(name.removeprefix('raw_shift_'))
                shifts=(fit_decision_shifts(raw,data['dev_y'],'multilabel',cfg['decision_grid'])*fraction).astype(np.float32)
                np.testing.assert_array_equal(shifts,np.asarray(control['shifts'],np.float32));assert control['temperature']==1.
                for state in m['assessment']['states']:
                    ids=np.asarray(state['fitting']);fitted=fit_decision_shifts(raw[ids],data['dev_y'][ids],'multilabel',cfg['decision_grid'])
                    for policy,ctrl in state['policies'].items():
                        factor=0. if policy=='raw' else float(policy.removeprefix('raw_shift_'))
                        np.testing.assert_array_equal((fitted*factor).astype(np.float32),np.asarray(ctrl['shifts'],np.float32))
            if 'fit' in m:fits[m['fit']['artifact']]=m['fit']
            del model,raw
        for artifact,fit in fits.items():
            path=Path(artifact);assert digest(path)==fit['sha256'];initial=path.parent/'initial.npz';source=path.parent/'data.npz'
            assert digest(initial)==fit['initial_sha256'] and digest(source)==fit['data_sha256']
            with np.load(source,allow_pickle=False) as f:
                for k,key in [('train_x','train_x'),('train_y','train_y'),('validation_x','dev_x'),('validation_y','dev_y')]:np.testing.assert_array_equal(f[k],data[key])
            assert max(fit['train_replay_error'],fit['validation_replay_error'])<2e-5
        previous=case['warmup']
        for step in case['traces'][0]['steps']:
            if 'uid' not in step:continue
            chosen=by[step['uid']];fit=chosen['fit'];checkpoint=load_protected(previous['artifact'])
            with np.load(Path(fit['artifact']).parent/'initial.npz',allow_pickle=False) as f:
                train_error=float(np.max(np.abs(f['train']-checkpoint.raw(data['train_x']).astype(np.float32))))
                dev_error=float(np.max(np.abs(f['validation']-checkpoint.raw(data['dev_x']))))
            assert max(train_error,dev_error)<2e-6
            checkpoint_checks.append(dict(block=step['block'],previous_uid=previous['uid'],chosen_uid=chosen['uid'],train_initial_error=train_error,development_initial_error=dev_error))
            previous=chosen
        assert all(n<=24 for n in case['queries'].values())
        row=dict(case=case['case'],models=len(case['models']),unique_fits=len(fits),threshold_families=sorted(threshold_families),
                 max_replay_error=max_replay,max_OOF_metric_error=max_oof,checkpoint_checks=checkpoint_checks,status='passed')
        cases.append(row);total_models+=row['models'];total_fits+=len(fits);write(out/'progress.json',dict(cases=cases));event('development_audit_case',**row);gc.collect()
    result=dict(status='broad_development_audit_complete',cases=cases,models=total_models,unique_fits=total_fits,new_trees=0,
                test_evaluated=False,environment=environment,source_snapshot_sha256=spec['snapshot_sha256'],elapsed_seconds=perf_counter()-started)
    write(out/'report.json',result);event('broad_development_audit_complete',models=total_models,cases=len(cases))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
