import numpy as np
from budget_quality_worker import locked,digest
from audit_abl_transfer import independent_task_quality
from audit_abl_protected import physical_cost
from abl_protected_core import load_protected
from audit_abl_quality import control,guard


def audit(report):
    cases=[]
    for entry in report['evaluations']:
        result=locked(entry['path'],entry['sha256']);case=locked(result['development']['path'],result['development']['sha256'])
        item=result['target'];assert digest(item['path'])==item['sha256']
        with np.load(item['path'],allow_pickle=False) as f:y=f['target'];ids=f['indices']
        with np.load(case['record']['case']['split_file'],allow_pickle=False) as f:np.testing.assert_array_equal(ids,f['test'])
        with np.load(case['record']['dataset']['path'],allow_pickle=False) as f:
            np.testing.assert_array_equal(y,f['target'][ids]);x=f['features'][ids].astype(np.float32)
        raw={}
        for uid,item in result['predictions'].items():
            assert digest(item['path'])==item['sha256'];meta=case['metas'][uid]
            assert digest(meta['artifact'])==meta['sha256'];model=load_protected(meta['artifact'])
            assert physical_cost(model)==meta['cost_bytes']
            with np.load(item['path'],allow_pickle=False) as f:raw[uid]=f['raw'].copy()
            np.testing.assert_allclose(raw[uid],model.raw(x),atol=1e-12,rtol=0)
        rows={r['uid']:r for r in case['rows']};assert len(result['points'])==len(case['choices'])==28
        for p,c in zip(result['points'],case['choices']):
            for key in ('mode','objective','group','budget_bytes'):assert p[key]==c[key]
            if c['chosen'] is None:assert p['status']=='no_feasible_candidate';continue
            uid=c['chosen']['uid'];policy=c['chosen']['policy'];assert p['uid']==uid and p['policy']==policy
            assert p['control']==control(rows[uid],policy)
            q=independent_task_quality(y,raw[uid],p['control'],case['task'])
            for k,v in q.items():assert abs(v-p['test'][k])<2e-10
            if p['group']!='full_unbounded':assert p['cost_bytes']<=case['budget_bytes']
            full=next(v for v in result['points'] if v['mode']==p['mode'] and v['objective']==p['objective'] and v['group']=='full_unbounded')
            assert p['full_reference_guard_5percent']==guard(q,full['test'])
            assert p['actual_compressed_increment']==bool(case['metas'][uid]['compressed_blocks'])
        cases.append(dict(case=result['case'],status='passed',points=28,model_replays=len(raw)))
    assert len(cases)==9
    return dict(status='passed',cases=cases,new_tree_fits=0,gpu_execution=False,
        scope='independent fixed choices/test metrics, protected model replay, unique/prepared byte sums and5percent guards')
