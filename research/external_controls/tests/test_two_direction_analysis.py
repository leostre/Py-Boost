import json
import pytest
from pathlib import Path
from budget_quality_worker import write,digest
from two_direction_analysis import inference_tables,representation_table


def test_latency_rollup_preserves_paired_sessions_and_fresh_memory_records(tmp_path):
    batches=[1,32,256,1024]
    def save(name,value):
        p=tmp_path/name;write(p,value);return dict(path=str(p),sha256=digest(p))
    tuning=save('tuning.json',{'entries':[dict(case='toy_1',uid='full',dataset='toy',roles=['full200','ordinary70'],record={'case':{'outputs':6}})]})
    sessions=[]
    for batch in batches:
        for session in range(3):sessions.append(dict(batch=batch,session=session,samples={
             'packed':dict(gpu_ms=[1.+session]*32,request_ms=[2.+session]*32),
             'legacy':dict(gpu_ms=[2.*(1.+session)]*32,request_ms=[2.*(2.+session)]*32)}))
    results=[dict(case='toy_1',uid='full',mode='latency',**save('latency.json',dict(status='wide_probe_complete',mode='latency',case='toy_1',uid='full',rows=sessions)))]
    for backend,factor in [('packed',1),('legacy',2)]:
        rows=[dict(batch=b,resident_device_array_bytes=100*factor,peak_pool_used_bytes=500*factor,incremental_peak_pool_bytes=400*factor) for b in batches]
        results.append(dict(case='toy_1',uid='full',mode='memory_'+backend,**save(backend+'.json',dict(status='wide_probe_complete',mode='memory_'+backend,case='toy_1',uid='full',rows=rows))))
    report=save('measurements.json',dict(status='wide_measurements_complete',source_tuning=tuning,results=results))
    cfg=dict(bank_batches=batches,broad_analysis=dict(tuning=tuning,measurements=report));out=tmp_path/'out';out.mkdir()
    rows,pairs,_=inference_tables(cfg,out)
    assert len(rows)==8 and len(pairs)==4
    assert all(p['gpu_speedup']==2 and p['request_p95_speedup']==2 and p['peak_reduction_percent']==50 for p in pairs)
    assert all(r['requests']==96 for r in rows)
    # Missing requests cannot silently become a seemingly valid p95.
    sessions[0]['samples']['packed']['gpu_ms'].pop()
    updated=save('latency.json',dict(status='wide_probe_complete',mode='latency',case='toy_1',uid='full',rows=sessions))
    results[0].update(updated);cfg['broad_analysis']['measurements']=save('measurements.json',dict(status='wide_measurements_complete',source_tuning=tuning,results=results))
    with pytest.raises(AssertionError):inference_tables(cfg,out)


def test_representation_effect_is_separate_from_executor_effect(tmp_path):
    common=dict(case='toy_1',dataset='toy',batch=32,backend='packed')
    rows=[dict(common,uid='full',roles=json.dumps(['full200']),gpu_p50_ms=8.,request_p95_ms=10.,resident_device_array_bytes=1000,peak_pool_used_bytes=2000),
          dict(common,uid='small',roles=json.dumps(['ordinary70']),gpu_p50_ms=2.,request_p95_ms=4.,resident_device_array_bytes=400,peak_pool_used_bytes=1000),
          dict(common,uid='hybrid',roles=json.dumps(['direct_ABL200','direct_ABL70']),gpu_p50_ms=4.,request_p95_ms=5.,resident_device_array_bytes=300,peak_pool_used_bytes=900)]
    lookup={('toy_1',u):dict(quality={'f1_macro':q},family=f,basis_values=b,nnz=n) for u,q,f,b,n in
            [('full',.8,'ordinary',0,0),('small',.7,'ordinary',0,0),('hybrid',.75,'direct_ABL',12,3)]}
    result=representation_table(rows,lookup,tmp_path)
    assert len(result)==2
    for r in result:
        assert r['gpu_vs_full_speedup']==2
        assert r['gpu_vs_small_ordinary_speedup']==.5
        assert r['test_delta_full_pp']==pytest.approx(-5)
        assert r['test_delta_ordinary70_pp']==pytest.approx(5)
