"""Replay recorded serving samples and costs; never starts a CUDA context."""
from collections import defaultdict
from pathlib import Path
import json

import numpy as np

from budget_quality_worker import digest,locked
from audit_abl_quality import control


def audit_serving(cfg):
    source=cfg['abl_serving_completed'];report=locked(source['path'],source['sha256'])
    root=Path(source['path']).parent
    receipt_path=Path(__file__).resolve().parent/cfg['abl_serving_receipt']['path']
    receipt=locked(receipt_path,cfg['abl_serving_receipt']['sha256'])
    assert receipt['run_id']=='6573312a-2ead-4275-95a1-3786014a51ea'
    expected={r['path']:r['sha256'] for r in receipt['jobs']}
    assert len(expected)==130
    plan=locked(root/'serving_plan.json',receipt['serving_plan_sha256'])
    assert plan['blocks']==3 and plan['requests']==32 and plan['warmup']==2
    assert plan['batches']==[1,32,256,1024] and plan['groups_variant']=='groups'
    assert plan['source']==cfg['abl_study_completed']
    primary=locked(cfg['abl_study_completed']['path'],cfg['abl_study_completed']['sha256'])
    primary_root=Path(cfg['abl_study_completed']['path']).parent
    sealed=locked(primary_root/'pretest_choices.json',primary['pretest_seal_sha256'])
    cases={entry['case_id']:locked(entry['path'],entry['sha256']) for entry in sealed['cases']}
    for item in plan['plans']:
        case=cases[item['case']]
        for role,model in item['models'].items():
            choice=next(c for c in case['budget_choices'] if c['scheme']=='crossfit' and c['repeat']==0
                and c['mode']=='temperature_shifts' and c['objective']==item['objective']
                and c['budget_ratio']==item['budget_ratio'] and c['group']==role)
            if choice['chosen'] is None:assert model is None;continue
            selected=choice['chosen'];assert model['uid']==selected['uid'] and model['policy']==selected['policy']
            row=next(r for r in case['rows'] if r['uid']==model['uid'] and r['plan']==choice['plan'])
            assert model['control']==control(row,model['policy'])
            assert model['cost_bytes']==case['metas'][model['uid']]['cost_bytes']
    summaries=report['summaries']
    times={(r['case'],r['objective'],r['budget_ratio'],r['role'],r['batch'],r['mode']):r for r in summaries['timings']}
    ratios={(r['case'],r['objective'],r['budget_ratio'],r['role'],r['batch'],r['mode']):r for r in summaries['paired_ratios']}
    memory_summary={(r['case'],r['uid'],r['batch'],r['mode'],r['state']):r for r in summaries['memory']}
    timing_jobs=0;memory_jobs=0;sample_count=0;profile_count=0
    for path,sha in expected.items():
        item=locked(path,sha);assert item['status']=='complete' and item['new_tree_fits']==0
        job=item['job']
        if job['mode']=='timing':
            timing_jobs+=1;roles={r for r,m in job['models'].items() if m is not None}
            assert len(item['samples'])==len(roles)*4*2*3*32
            unique=set();orders=defaultdict(list);samples=defaultdict(dict)
            for sample in item['samples']:
                role,batch,mode,block,request=(sample[k] for k in ('role','batch','mode','block','request'))
                key=(role,batch,mode,block,request);assert key not in unique;unique.add(key)
                assert role in roles and batch in (1,32,256,1024) and mode in ('host','device')
                assert 0<=block<3 and 0<=request<32
                assert set(sample['order'])==roles and len(sample['order'])==len(roles)
                assert np.isfinite(sample['milliseconds']) and sample['milliseconds']>0
                orders[batch,mode,block,request].append(tuple(sample['order']))
                samples[role,batch,mode][block,request]=sample['milliseconds']
            assert all(len(v)==len(roles) and len(set(v))==1 for v in orders.values())
            for (role,batch,mode),values in samples.items():
                array=np.array([v for _,v in sorted(values.items())])
                key=(job['case'],job['objective'],job['budget_ratio'],role,batch,mode)
                summary=times[key]
                assert abs(summary['median_ms']-float(np.median(array)))<1e-10
                assert abs(summary['q25_ms']-float(np.quantile(array,.25)))<1e-10
                if role!='native' and ('native',batch,mode) in samples:
                    baseline=np.array([v for _,v in sorted(samples['native',batch,mode].items())])
                    assert abs(ratios[key]['paired_median_compact_over_native']-float(np.median(array/baseline)))<1e-10
            for role in roles:assert item['setup'][role]['probability_parity_error']<=2e-5
            assert len(item['profiles'])==len(roles)*4*3
            for profile in item['profiles']:
                assert profile['fused_traverse_gather_decode_event_ms']>=0
                assert profile['calibration_probabilities_decisions_event_ms']>=0
            sample_count+=len(item['samples']);profile_count+=len(item['profiles'])
        elif job['mode']=='memory':
            memory_jobs+=1;assert item['model_bytes']==job['model']['cost_bytes']
            assert item['model_pool_used_bytes']>=item['model_bytes']
            assert len(item['records'])==4
            for record in item['records']:
                assert item['model_pool_used_bytes']<=record['pool_peak_used_bytes']<=16*1024**3
                assert record['pool_peak_reserved_bytes']>=record['pool_peak_used_bytes']
                key=(job['case'],job['model']['uid'],job['batch'],record['mode'],record['state'])
                # Same UID may have multiple calibrated controls: validate the
                # multiset below instead of silently overwriting a keyed row.
                candidates=[r for r in summaries['memory'] if
                    (r['case'],r['uid'],r['batch'],r['mode'],r['state'])==key]
                assert any(r['pool_peak_used_bytes']==record['pool_peak_used_bytes'] and
                    r['pool_peak_reserved_bytes']==record['pool_peak_reserved_bytes'] for r in candidates)
        else:raise AssertionError('unknown measurement job')
    assert timing_jobs==report['timing_jobs']==18 and memory_jobs==report['memory_jobs']==112
    assert sample_count==41472 and profile_count==648
    assert len(summaries['memory'])==448 and len(times)==432 and len(ratios)==288
    return dict(status='passed',source=source,timing_jobs=timing_jobs,memory_jobs=memory_jobs,
        request_samples=sample_count,profiles=profile_count,model_fits=0,gpu_execution=False)
