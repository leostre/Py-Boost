"""Independent frozen-choice, test metric and storage audit of T1 evaluation."""
from pathlib import Path
import numpy as np
from budget_quality_worker import locked,digest,native_load,write
from audit_calibration_pilot import independent_quality
from audit_abl_quality import guard,control


def audit(report):
    cases=[]
    for entry in report['evaluations']:
        result=locked(entry['path'],entry['sha256'])
        source=result['source_development'];case=locked(source['path'],source['sha256'])
        predictions={}
        target=result['target'];assert digest(target['path'])==target['sha256']
        with np.load(target['path'],allow_pickle=False) as f:y=f['target'];ids=f['indices']
        with np.load(case['record']['case']['split_file'],allow_pickle=False) as f:
            np.testing.assert_array_equal(ids,f['test'])
        with np.load(case['record']['dataset']['path'],allow_pickle=False) as f:
            np.testing.assert_array_equal(y,f['target'][ids])
            x=f['features'][ids].astype(np.float32)
        for uid,item in result['predictions'].items():
            assert digest(item['path'])==item['sha256']
            with np.load(item['path'],allow_pickle=False) as f:predictions[uid]=f['raw'].copy()
            meta=case['metas'][uid];assert digest(meta['artifact'])==meta['sha256']
            model=native_load(meta['artifact']);prepared=model.prepare_inference()
            # independent bytes, not the generating ledger function
            arrays=[getattr(prepared.base,k) for k in ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias')]
            arrays += [getattr(prepared,k) for k in ('indptr','columns','values','decision_shifts')]
            assert sum(a.nbytes for a in arrays)+4==meta['cost_bytes']
            np.testing.assert_allclose(predictions[uid],prepared.raw(x),atol=1e-12,rtol=0)
        assert len(result['points'])==len(case['choices'])==48
        rows={r['uid']:r for r in case['rows']}
        for point,choice in zip(result['points'],case['choices']):
            for name in ('mode','objective','group','budget_bytes'):
                assert point[name]==choice[name]
            if choice['chosen'] is None:assert point['status']=='no_feasible_candidate';continue
            chosen=choice['chosen'];uid=chosen['uid'];meta=case['metas'][uid]
            assert point['uid']==uid and point['policy']==chosen['policy']
            expected=control(rows[uid],chosen['policy']);assert point['control']==expected
            q=independent_quality(y,predictions[uid],expected)
            for metric,value in q.items():assert abs(value-point['test'][metric])<2e-10
            assert point['parameters']==native_load(meta['artifact']).parameter_count+1
            if point['group']!='full_unbounded':assert point['cost_bytes']<=case['budget_bytes']
            reference=next(p for p in result['points'] if p['mode']==point['mode'] and
                p['objective']==point['objective'] and p['group']=='full_unbounded')
            assert point['full_reference_guard_5percent']==guard(q,reference['test'])
            assert point['actual_compact_gradient_training']==bool(point['compact_gradient_blocks_before_snapshot'])
        for interval in result['bootstrap_intervals']:
            # Rebuild each interval from paired sufficient statistics with an
            # independent confusion-count implementation and the frozen RNG.
            summaries=[]
            for side in ('left','right'):
                z=predictions[interval[side+'_uid']].astype(np.float64)
                cp=interval[side+'_control'];scaled=z/cp['temperature'];e=np.exp(scaled-scaled.max(axis=1,keepdims=True))
                p=np.clip(e/e.sum(axis=1,keepdims=True),1e-6,1-1e-6);p/=p.sum(axis=1,keepdims=True)
                pred=(scaled-np.asarray(cp['shifts'])).argmax(axis=1)
                losses=-np.log(np.clip(p[np.arange(len(y)),y],1e-7,1-1e-7))
                summaries.append((pred,losses,z.shape[1]))
            rng=np.random.default_rng(20261026);deltas=[]
            for _ in range(500):
                w=rng.multinomial(len(y),np.full(len(y),1/len(y)));scores=[]
                for pred,loss,c in summaries:
                    cm=np.bincount(c*y+pred,weights=w,minlength=c*c).reshape(c,c)
                    denom=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denom,out=np.zeros(c),where=denom>0)
                    scores.append((float(w@loss/w.sum()),float(f1.mean())))
                deltas.append([scores[0][0]-scores[1][0],100*(scores[0][1]-scores[1][1])])
            bounds=np.quantile(deltas,[.025,.975],axis=0).T
            np.testing.assert_allclose(bounds[0],interval['delta_loss_ci95'],atol=2e-12,rtol=0)
            np.testing.assert_allclose(bounds[1],interval['delta_f1_pp_ci95'],atol=2e-10,rtol=0)
        cases.append(dict(case=result['case'],choices=48,predictions=len(predictions),bootstrap_pairs=len(result['bootstrap_intervals']),status='passed'))
    assert len(cases)==9 and report['new_tree_fits']==0 and not report['gpu_execution']
    return dict(status='passed',cases=cases,new_tree_fits=0,gpu_execution=False,
        scope='independent test metrics, byte sums, model replay, fixed choices and paired-bootstrap intervals')
