"""S0/S1: train-D-only tuning, then paired serving on frozen saved models."""
from pathlib import Path
from time import perf_counter
import json,sys,subprocess,gc
import numpy as np
from budget_quality_worker import locked,digest,write,event
from abl_quality_worker import check_native_environment
from abl_transfer_study import data_before_test
from abl_protected_core import load_protected
from packed_serving_core import PackedPlan
from broad_core import budget_reference
from unified_serving_core import bank_frontier


def scenarios(report):
    seal=locked(report['pretest_seal']['path'],report['pretest_seal']['sha256']);chosen=[];seen=set()
    for ref in report['cases']:
        case=locked(ref['path'],ref['sha256']);name=case['record']['case']['dataset']
        if name in seen:continue
        seen.add(name);by={m['uid']:m for m in case['models']};roles={'full200':'ordinary/d3/t200','direct_AB200':'direct_AB/t200','direct_ABL200':'direct_ABL/t200'}
        for family in ('ordinary','posthoc','controller','mixed24','direct_AB','direct_ABL'):
            for ratio in ((.25,.5,.7,1.) if family in ('ordinary','mixed24') else (.7,)):
                row=next(r for r in seal['selected'] if r['case']==case['case'] and r['family']==family and r['ratio']==ratio)
                if row['uid'] is not None:roles[family+str(int(round(100*ratio)))]=row['uid']
        unique={}
        for role,uid in roles.items():unique.setdefault(uid,[]).append(role)
        for uid,labels in unique.items():
            m=by[uid];chosen.append(dict(case=case['case'],dataset=name,uid=uid,roles=labels,model=m,record=case['record']))
    return chosen


def bank_selections(entries):
    cases=list(dict.fromkeys(e['case'] for e in entries));reference=sum(budget_reference(next(e for e in entries if e['case']==c)['record']['case']['outputs']) for c in cases)
    budgets=[int(reference*r) for r in (.25,.5,.7,1.)];banks=[]
    for family in ('ordinary','mixed24'):
        options=[]
        for c in cases:
            rows=[]
            for entry in entries:
                if entry['case']!=c or not any(role.startswith(family) for role in entry['roles']):continue
                rows.append(dict(uid=entry['uid'],cost_bytes=entry['candidates'][0]['resident_bytes']+8,utility=entry['model']['assessment']['quality']['f1_macro']))
            options.append(rows)
        for ratio,row in zip((.25,.5,.7,1.),bank_frontier(options,budgets)):
            banks.append(dict(row,name=family+str(int(round(100*ratio))),family=family,ratio=ratio,cases=cases))
    full=[next(e for e in entries if e['case']==c and 'full200' in e['roles']) for c in cases]
    banks.append(dict(name='full200',family='full200',ratio=None,status='feasible',cases=cases,choices=[e['uid'] for e in full],
                      cost_bytes=sum(e['candidates'][0]['resident_bytes']+8 for e in full),utility=float(np.mean([e['model']['assessment']['quality']['f1_macro'] for e in full]))))
    return dict(banks=banks,analytic_reference_bytes=reference,objective='equal-task mean OOF macroF1; actual packed arrays plus8byte temperature per model',
                catalog='four frozen per-task budget choices from the same24query ordinary or mixed catalogue; no new D fits')


def tune(cfg,out):
    import cupy as cp
    from packed_preflight import gpu_preflight
    from packed_serving_gpu import PackedGpu
    from unified_serving_gpu import BlockGpu
    spec=cfg['wide_serving'];source=locked(spec['training']['path'],spec['training']['sha256']);assert source['status']=='broad_training_complete'
    preflight=gpu_preflight();write(out/'synthetic_preflight.json',preflight)
    entries=[];cp.get_default_memory_pool().set_limit(size=12*1024**3)
    for index,entry in enumerate(scenarios(source)):
        m=entry['model'];assert digest(m['artifact'])==m['sha256'];model=load_protected(m['artifact']);data=data_before_test(entry['record'])
        control=m['assessment']['refit_control'];t=control['temperature'];shifts=control['shifts'];x=data['dev_x'];sample=x[np.linspace(0,len(x)-1,min(19,len(x))).astype(int)]
        expected=model.raw(sample);begin=perf_counter();reference=BlockGpu(model,t,shifts);cp.cuda.get_current_stream().synchronize()
        legacy_prepare_seconds=perf_counter()-begin;begin=perf_counter();ep,el=reference.request(sample);legacy_first_request_seconds=perf_counter()-begin
        plans=[];candidates=[];skipped=[]
        for groups in (1,2,4):
            for layout in ('aos','soa'):
                begin=perf_counter()
                try:device=PackedGpu(model,t,shifts,groups,layout)
                except ValueError as exc:skipped.append(dict(groups=groups,layout=layout,reason=str(exc)));continue
                cp.cuda.get_current_stream().synchronize();prepare_seconds=perf_counter()-begin;first_begin=perf_counter()
                dx=cp.asarray(sample,order='C');buf=device.buffers(len(sample));device.launch_into(dx,*buf,raw=True)
                actual_raw=buf[0].get();first_raw_seconds=perf_counter()-first_begin
                error=float(np.max(np.abs(actual_raw-expected)));assert error<1e-8,(entry['case'],m['uid'],error)
                p,l=device.request(sample);np.testing.assert_allclose(p,ep,atol=2e-9,rtol=2e-9)
                difference=(l!=el);ambiguous=np.abs(expected/t-np.asarray(shifts))<1e-8 if model.task=='multilabel' else np.zeros_like(difference)
                assert not np.any(difference & ~ambiguous)
                compile_seconds=perf_counter()-begin;timings={}
                for batch in cfg['bank_batches']:
                    features=np.ascontiguousarray(x[np.arange(batch)%len(x)]);xx=cp.asarray(features);buffers=device.buffers(batch)
                    for _ in range(3):device.launch_into(xx,*buffers)
                    cp.cuda.get_current_stream().synchronize();samples=[]
                    for _ in range(16):
                        a,b=cp.cuda.Event(),cp.cuda.Event();a.record();device.launch_into(xx,*buffers);b.record();b.synchronize();samples.append(float(cp.cuda.get_elapsed_time(a,b)))
                    timings[str(batch)]=samples;del xx,buffers
                row=dict(groups=groups,layout=layout,raw_error=error,label_ties=int(np.sum(difference)),shared_bytes=device.plan.shared_bytes,
                         resident_bytes=device.resident_device_array_bytes,prepare_seconds=prepare_seconds,first_raw_request_seconds=first_raw_seconds,
                         cold_request_scope='includes buffer creation, possible JIT/cache loading and D2H; not a compile-only measurement',
                         compile_and_check_seconds=compile_seconds,kernel_attributes=device.kernel.attributes,timings=timings)
                candidates.append(row);del device,dx,buf;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        for batch in cfg['bank_batches']:
            best=min(candidates,key=lambda r:(float(np.median(r['timings'][str(batch)])),r['groups'],r['layout']))
            plans.append(dict(batch=batch,groups=best['groups'],layout=best['layout'],D_gpu_p50_ms=float(np.median(best['timings'][str(batch)]))))
        entry.update(plans=plans,candidates=candidates,skipped=skipped,legacy_prepare_seconds=legacy_prepare_seconds,
                     legacy_first_request_seconds=legacy_first_request_seconds);entries.append(entry)
        write(out/'tuning_progress.json',dict(entries=entries,test_metrics_evaluated=False));event('packed_tune_model',index=index,case=entry['case'],uid=entry['uid'],plans=plans)
        del reference,model,data;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    seal=out/'tuning_seal.json';write(seal,dict(entries=entries,banks=bank_selections(entries),source_training=spec['training'],test_metrics_evaluated=False,
        rule='minimum median16D GPU-resident requests per batch; choose before final measurement; equal snapshot precision; no test labels'))
    return dict(status='wide_tuning_complete',entries=len(entries),tuning_seal=dict(path=str(seal),sha256=digest(seal)),preflight=dict(path=str(out/'synthetic_preflight.json'),sha256=digest(out/'synthetic_preflight.json')))


def measure(cfg,out):
    spec=cfg['wide_serving'];seal=locked(spec['tuning']['path'],spec['tuning']['sha256']);results=[]
    for i,entry in enumerate(seal['entries']):
        for mode in ('latency','memory_packed','memory_legacy'):
            directory=out/f'm{i:03d}'/mode;directory.mkdir(parents=True,exist_ok=True);request=directory/'request.json'
            write(request,dict(entry=entry,mode=mode,batches=cfg['bank_batches'],requests=32,sessions=3,seed=202610290+i,output=str(directory)))
            subprocess.run([sys.executable,str(Path(__file__).parent/'wide_probe.py'),str(request)],check=True,timeout=1800)
            file=directory/'report.json';results.append(dict(case=entry['case'],uid=entry['uid'],mode=mode,path=str(file),sha256=digest(file)))
        write(out/'progress.json',dict(results=results));event('wide_measure_model',index=i,case=entry['case'],uid=entry['uid'])
    return dict(status='wide_measurements_complete',results=results,source_tuning=spec['tuning'])


def banks(cfg,out):
    spec=cfg['wide_serving'];seal=locked(spec['tuning']['path'],spec['tuning']['sha256']);results=[]
    selected=[b for b in seal['banks']['banks'] if b['name'] in ('ordinary70','mixed2470','full200') and b['status']=='feasible']
    jobs=[(b,backend) for b in selected for backend in ('packed','legacy')]
    order=np.random.default_rng(202610294).permutation(len(jobs))
    for index in order:
        bank,backend=jobs[index];directory=out/(bank['name']+'-'+backend);directory.mkdir(exist_ok=True)
        entries=[next(e for e in seal['entries'] if e['case']==c and e['uid']==uid) for c,uid in zip(bank['cases'],bank['choices'])]
        request=directory/'request.json';write(request,dict(bank=bank,entries=entries,backend=backend,batches=cfg['bank_batches'],requests=32,sessions=3,
                                                         seed=202610294,output=str(directory)))
        subprocess.run([sys.executable,str(Path(__file__).parent/'wide_bank_probe.py'),str(request)],check=True,timeout=1800)
        file=directory/'report.json';results.append(dict(bank=bank['name'],backend=backend,path=str(file),sha256=digest(file)))
        write(out/'progress.json',dict(results=results));event('wide_bank_complete',bank=bank['name'],backend=backend)
    return dict(status='wide_banks_complete',results=results,source_tuning=spec['tuning'])


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);started=perf_counter();environment=check_native_environment(cfg)
    operation=cfg['wide_serving']['operation']
    if operation not in ('preflight','tune','measure','banks'):raise ValueError('unknown serving operation')
    if operation!='preflight' and not cfg['wide_serving'].get('inputs_ready',False):
        raise RuntimeError('serving predecessor receipts are not sealed yet')
    if operation=='preflight':
        from packed_preflight import gpu_preflight
        result=gpu_preflight()
    else:result={'tune':tune,'measure':measure,'banks':banks}[operation](cfg,out)
    result.update(environment=environment,elapsed_seconds=perf_counter()-started,new_trees=0)
    write(out/'report.json',result);event('wide_serving_complete',**result)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
