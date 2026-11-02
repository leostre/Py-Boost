"""Read sealed numeric evidence only; no model fitting or new predictions."""
from pathlib import Path
from collections import defaultdict
import csv,json,sys,subprocess,shutil
import numpy as np
from budget_quality_worker import locked,write,digest,event
from abl_quality_worker import check_native_environment


def dataset(case):return case.rsplit('_',1)[0]
def export(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def equal_dataset_mean(rows,key):
    grouped=defaultdict(list)
    for r in rows:grouped[r['dataset']].append(r[key])
    means={k:float(np.mean(v)) for k,v in grouped.items()}
    return dict(mean=float(np.mean(list(means.values()))),positive_datasets=sum(x>0 for x in means.values()),datasets=means)


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);environment=check_native_environment(cfg);data=out/'data';data.mkdir(exist_ok=True)
    refs=cfg['six']['analysis_sources'];sources={k:locked(v['path'],v['sha256']) for k,v in refs.items()}
    total=sum(sources[k]['new_trees'] for k in ('R2','R2C','R5Q','R5QM'));assert total==21600
    structural=[(stage,c) for stage in ('R3','R3C') for c in sources[stage]['results']]
    full={c['case']:next(m for m in c['models'] if m['uid']=='prefix/t200') for _,c in structural}
    quality_rows=[]
    for stage in ('R2','R2C','R3','R3C'):
        for c in sources[stage]['results']:
            by={m['uid']:m for m in c['models']};reference=full[c['case']]
            for m in c['models']:
                if stage.startswith('R3'):same=by['prefix/t'+str(m['trees'])]
                else:same=by[('baseline-direct_ABL/t200' if 'ABL' in m['family'] else 'baseline-direct_AB/t200')]
                q=m['quality'];quality_rows.append(dict(stage=stage,case=c['case'],dataset=dataset(c['case']),uid=m['uid'],family=m['family'],
                    trees=m['trees'],packed_bytes=m['packed_bytes'],test_macro_f1=q['f1_macro'],test_loss=q['loss'],
                    reference_uid=same['uid'],delta_pp=100*(q['f1_macro']-same['quality']['f1_macro']),memory_ratio=m['packed_bytes']/same['packed_bytes'],
                    full200_f1=reference['quality']['f1_macro'],delta_full200_pp=100*(q['f1_macro']-reference['quality']['f1_macro']),memory_full200_ratio=m['packed_bytes']/reference['packed_bytes'],
                    below_half_full200=q['f1_macro']<.5*reference['quality']['f1_macro']))
    export(data/'quality_points.csv',quality_rows)
    quant=[]
    for stage in ('R5','R5A'):
        for c in sources[stage]['results']:
            for m in c['models']:
                for policy,q in m['quality'].items():
                    quant.append(dict(stage=stage,case=c['case'],dataset=dataset(c['case']),source_uid=m['source_uid'],codec=m['codec'],policy=policy,
                        packed_bytes=m['packed_bytes'],fp32_bytes=m['fp32_bytes'],memory_ratio=m['packed_bytes']/m['fp32_bytes'],test_macro_f1=q['f1_macro'],
                        original_macro_f1=m['original_quality']['f1_macro'],delta_pp=100*(q['f1_macro']-m['original_quality']['f1_macro'])))
    export(data/'quantization.csv',quant)
    controls={c['case']:{m['uid']:m for m in c['models']} for c in sources['R5Q']['results']};continuation=[]
    for stage in ('R5Q','R5QM'):
        for c in sources[stage]['results']:
            for m in c['models']:
                if not m['uid'].endswith('/QAT'):continue
                codec=m['uid'].split('/')[0];control=controls[c['case']][codec+'/PTQ']
                continuation.append(dict(stage=stage,case=c['case'],dataset=dataset(c['case']),codec=codec,test_macro_f1=m['quality']['f1_macro'],
                    control_macro_f1=control['quality']['f1_macro'],delta_pp=100*(m['quality']['f1_macro']-control['quality']['f1_macro']),
                    packed_bytes=m['packed_bytes'],control_bytes=control['packed_bytes'],clipped=m['fit']['clipped'],coefficients=m['fit']['coefficients']))
    export(data/'quantized_continuation.csv',continuation)
    inference=[];aggregate=[];compatibility=[]
    for stage in ('R4','R5S','R5O','N2V','N2F'):
        for ref in sources[stage]['rows']:
            report=locked(ref['path'],ref['sha256']);measurements=report['measured' if stage.startswith('N') else 'measurements'];grouped=defaultdict(lambda:dict(gpu=[],request=[],valid=True))
            for row in measurements:
                for backend,samples in row['samples'].items():
                    valid=not any(f['name']==backend for f in row.get('numerical_failures',[]))
                    uid=report.get('uid','ordinary/d3/t200');record=dict(stage=stage,case=report['case'],dataset=dataset(report['case']),uid=uid,batch=row['batch'],session=row['session'],backend=backend,
                        valid=valid,gpu_p50_ms=float(np.median(samples['gpu_ms'])),request_p95_ms=float(np.percentile(samples['request_ms'],95)))
                    inference.append(record);g=grouped[row['batch'],backend];g['gpu'].extend(samples['gpu_ms']);g['request'].extend(samples['request_ms']);g['valid'] &= valid
            for (batch,backend),g in grouped.items():
                memory=next(m for m in report['memory'] if m['batch']==batch and m['name']==backend)
                aggregate.append(dict(stage=stage,case=report['case'],dataset=dataset(report['case']),uid=report.get('uid','ordinary/d3/t200'),batch=batch,backend=backend,valid=g['valid'],
                    gpu_p50_ms=float(np.median(g['gpu'])),request_p95_ms=float(np.percentile(g['request'],95)),
                    model_bytes=memory['model_allocated_bytes'] if stage.startswith('N') else memory['model_arrays'],
                    model_plus_workspace_bytes=memory['model_plus_request'] if stage.startswith('N') else memory['model_plus_active_workspace']))
            if stage.startswith('N'):
                for row in report['tuning']:
                    for backend,check in row['compatibility'].items():compatibility.append(dict(stage=stage,case=report['case'],batch=row['batch'],backend=backend,**check))
    export(data/'inference_sessions.csv',inference);export(data/'inference_aggregate.csv',aggregate);export(data/'nvforest_compatibility.csv',compatibility)
    banks=[]
    for stage in ('R6D','R6E'):
        for b in sources[stage]['banks']:
            banks.append(dict(stage=stage,family=b['family'],traffic=b['traffic'],ratio=b['ratio'],status=b['status'],
                packed_bytes=b.get('cost_bytes'),test_macro_f1=b.get('test_equal_task_macro_f1'),test_weighted_macro_f1=b.get('test_weighted_macro_f1')))
    export(data/'bank_selection.csv',banks)
    serving=[]
    for row in sources['R6B']['rows']:
        if row['status']!='complete':continue
        service=[t for s in row['sessions'] for t in s['request_ms']];queued=[t for s in row['sessions'] for t in s['queued_request_ms']]
        serving.append({k:row[k] for k in ('family','traffic','batch','concurrency','ratio','total_budget','workspace_reserved','model_array_bytes','retained_pool_bytes','own_pool_peak_bytes','test_equal_task_macro_f1','test_weighted_macro_f1','larger_product_catalogue')}
                       |dict(request_p95_ms=float(np.percentile(service,95)),queued_request_p95_ms=float(np.percentile(queued,95)),requests_per_second=float(np.mean([s['requests_per_second'] for s in row['sessions']]))))
    export(data/'bank_serving.csv',serving);export(data/'bank_capacity.csv',sources['R6B']['capacity'])
    report=sources['R6T'];boost=locked(report['boost']['path'],report['boost']['sha256']);tfm=locked(report['tabpfn']['path'],report['tabpfn']['sha256']);cohort=[];cohort_quality=[]
    for row in boost['rows']:
        for m in row['metrics']:cohort.append(dict(family=row['family'],codec=row['codec'],batch=m['batch'],traffic=m['traffic'],request_p50_ms=m['p50_ms'],request_p95_ms=m['p95_ms'],model_bytes=row['model_arrays'],own_pool_peak_bytes=row['own_pool_peak']))
        for q in row['quality']:cohort_quality.append(dict(case=q['case'],family=row['family'],codec=row['codec'],**q['quality']))
    for m in tfm.get('metrics',[]):cohort.append(dict(family='TabPFN cached auto8',codec='auto',batch=m['batch'],traffic=m['traffic'],request_p50_ms=m['p50_ms'],request_p95_ms=m['p95_ms'],
                    model_bytes=None,own_pool_peak_bytes=m['cuda_peak_allocated_bytes'],own_reserved_peak_bytes=m['cuda_peak_reserved_bytes'],resident_torch_bytes=m['cuda_allocated_bytes'],gpu_pid_attribution_verified=tfm['gpu_ownership_verified']))
    for q in tfm.get('quality',[]):cohort_quality.append(dict(case=q['case_id'],family='TabPFN cached auto8',codec='auto',**q['quality']))
    export(data/'five_task_serving.csv',cohort);export(data/'five_task_quality.csv',cohort_quality)
    summary=dict(status='six_directions_complete',new_trees=total,datasets_multilabel=12,old_splits=36,structural_representations=sum(len(c['models']) for _,c in structural),
        matched_byte=sources['R1']['summary'],training_compact_start=equal_dataset_mean([r for r in quality_rows if r['stage'] in ('R2','R2C') and r['uid']=='compressed40_ABL/t200'],'delta_pp'),
        compact_start_memory=equal_dataset_mean([r for r in quality_rows if r['stage'] in ('R2','R2C') and r['uid']=='compressed40_ABL/t200'],'memory_ratio'),
        refit50=equal_dataset_mean([r for r in quality_rows if r['stage'] in ('R3','R3C') and r['uid']=='refit/t50'],'delta_pp'),
        refitABL50=equal_dataset_mean([r for r in quality_rows if r['stage'] in ('R3','R3C') and r['uid']=='refit_ABL/t50'],'delta_pp'),
        corrected_quantized_continuation={codec:equal_dataset_mean([r for r in continuation if r['stage']=='R5QM' and r['codec']==codec],'delta_pp') for codec in ('fp16','int8')},
        actual_bank_budget_checks=len(serving),all_actual_bank_budgets_passed=all(r['own_pool_peak_bytes']<=r['total_budget'] for r in serving),
        tabpfn_status=tfm['status'],tabpfn_context_fits=tfm['context_fit_calls'],tabpfn_weight_bytes=tfm.get('shared_weight_bytes'),
        tabpfn_context_fit_seconds=sum(f['seconds'] for f in tfm.get('fits',[])),tabpfn_gpu_pid_verified=tfm.get('gpu_ownership_verified'),
        environment=environment,input_reports=refs,scope='old inspected tests, three splits share test; per-task means, no universal superiority or new blind evaluation; PTR+driver excluded from own pools')
    write(out/'summary.json',summary);write(out/'input_manifest.json',refs)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    for name in ('plot_six.py','orx_figstyle.py'):shutil.copy2(Path(__file__).parent/'six_figures'/name,figures/name)
    plot_python=Path(cfg['environment_root'])/'venv/bin/python'
    subprocess.run([str(plot_python),'-B',str(figures/'plot_six.py'),str(data),str(figures)],check=True,timeout=600)
    write(out/'report.json',dict(status='six_direction_rollup_complete',summary=dict(path=str(out/'summary.json'),sha256=digest(out/'summary.json')),
            data=[dict(path=str(p),sha256=digest(p)) for p in sorted(data.glob('*.csv'))],figures=[dict(path=str(p),sha256=digest(p)) for p in sorted(figures.glob('*')) if p.suffix in ('.svg','.pdf')],new_trees=0,total_program_trees=total))
    event('R7_complete',sha256=digest(out/'report.json'),total_program_trees=total)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
