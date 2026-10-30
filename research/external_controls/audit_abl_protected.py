"""Independent actual-state and protected-prefix audit, before test access."""
from pathlib import Path
import numpy as np
from budget_quality_worker import locked,digest,native_load
from abl_protected_core import load_protected
from abl_transfer_study import data_before_test
from abl_quality_core import fingerprint
from audit_abl_transfer import independent_task_quality
from audit_abl_quality import control,guard
from experiments.shared_leaf_basis import probabilities


def physical_cost(model):
    arrays=[model.bias,model.decision_shifts,model._zero]
    for b in model.blocks:
        arrays.extend(getattr(b.base,k) for k in ('nodes','node_offsets','leaf_offsets','coefficients','basis'))
        arrays.extend(getattr(b,k) for k in ('rows','columns','values'))
    unique={(a.__array_interface__['data'][0],a.nbytes):a for a in arrays if a.nbytes}
    coo=sum(a.nbytes for a in unique.values())+4
    total=model.bias.nbytes+model.decision_shifts.nbytes+4
    for b in model.blocks:
        p=b.prepare_inference()
        total+=sum(getattr(p.base,k).nbytes for k in ('nodes','node_offsets','leaf_offsets','coefficients','basis','bias'))
        total+=sum(getattr(p,k).nbytes for k in ('indptr','columns','values','decision_shifts'))
    return max(coo,total)


def choose(rows,metas,choice,budget):
    from abl_training_development_audit import independent_training_choice
    # Reuse its independently written bands, mapping only domain membership.
    allowed={'full_unbounded':{'full'},'native_budget':{'full'},'deployed':set(m['family'] for m in metas.values())}.get(choice['group'],{choice['group']})
    subset=[r for r in rows if metas[r['uid']]['family'] in allowed]
    mapped={u:dict(m,family='full' if choice['group'] in ('full_unbounded','native_budget') else 'joint') for u,m in metas.items()}
    remapped=dict(choice,group=choice['group'] if choice['group'] in ('full_unbounded','native_budget') else 'joint')
    return independent_training_choice(subset,mapped,remapped,budget)


def audit_development(entries,ledger):
    observed={};results=[]
    assert ledger['reserved_rounds']==40*len(ledger['attempts'])<=8000
    assert all(a['state']=='complete' for a in ledger['attempts'])
    for entry in entries:
        case=locked(entry['path'],entry['sha256']);data=data_before_test(case['record']);ncal,nctrl=data['sizes'][:2]
        warm=locked(case['warmup']['path'],case['warmup']['sha256']);warm_item=warm['models'][0]
        assert digest(warm_item['artifact'])==warm_item['sha256']
        f=warm['cycles'][0]['fit'];observed[(case['case_id'],'warmup',0)]=f
        assert f['trees']==40 and f['observed_minimum_leaf_support']>=10
        with np.load(warm['cycles'][0]['initial_path'],allow_pickle=False) as z:
            expected=np.zeros(data['train_y'].shape[1] if case['task']=='multilabel' else case['record']['case']['outputs'])
            if case['task']=='multilabel':
                p=np.clip(data['train_y'].mean(0),1e-7,1-1e-7);expected=np.log(p/(1-p)).astype(np.float32)
            np.testing.assert_array_equal(z['train'],np.broadcast_to(expected,z['train'].shape).astype(np.float32))
        for reference in case['trajectories']:
            t=locked(reference['path'],reference['sha256']);previous=load_protected(warm_item['artifact']);by={m['block']:m for m in t['models']}
            for cycle in t['cycles']:
                block=cycle['block'];fit=cycle['fit'];observed[(case['case_id'],reference['mode'],block)]=fit
                assert digest(fit['artifact'])==fit['sha256'] and digest(cycle['initial_path'])==cycle['initial_sha256']
                assert max(fit['train_replay_error'],fit['validation_replay_error'])<=2e-5
                assert fit['observed_minimum_leaf_support']>=10 and fit['training_pool_peak_bytes']<=16*1024**3
                with np.load(cycle['initial_path'],allow_pickle=False) as z:
                    np.testing.assert_allclose(z['train'],previous.raw(data['train_x']).astype(np.float32),atol=2e-6,rtol=2e-7)
                    np.testing.assert_allclose(z['validation'],previous.raw(data['dev_x'][:ncal]),atol=2e-9,rtol=1e-10)
                if cycle['action']=='rejected':continue
                m=by[block];assert digest(m['artifact'])==m['sha256'];current=load_protected(m['artifact'])
                assert m['cost_bytes']==physical_cost(current)
                if reference['mode']!='full':
                    assert m['cost_bytes']<=case['budget_bytes']
                    assert [fingerprint(b) for b in current.blocks[:-1]]==[fingerprint(b) for b in previous.blocks]
                    info=cycle['projection'];selected=info['chosen'];assert guard(selected['validation'],info['reference']['validation'])
                    q=independent_task_quality(data['dev_y'][ncal:ncal+nctrl],current.raw(data['dev_x'][ncal:ncal+nctrl]),
                        dict(temperature=selected['temperature'],shifts=selected['decision_shifts']),case['task'])
                    for k,v in q.items():assert abs(v-selected['validation'][k])<2e-8
                previous=current
        end=ncal+nctrl
        for row in case['rows']:
            meta=case['metas'][row['uid']];cache=meta['development_cache'];assert digest(cache['path'])==cache['sha256']
            with np.load(cache['path'],allow_pickle=False) as f:z=f['raw'][end:].copy()
            for name,q in row['validation'].items():
                actual=independent_task_quality(data['dev_y'][end:],z,control(row,name),case['task'])
                for k,v in actual.items():assert abs(v-q[k])<2e-10
        for c in case['choices']:assert choose(case['rows'],case['metas'],c,case['budget_bytes'])==c['chosen']
        results.append(dict(case=case['case_id'],status='passed',choices=len(case['choices']),candidate_models=len(case['metas'])))
    assert observed=={(a['case'],a['mode'],a['block']):a['fit'] for a in ledger['attempts']}
    return dict(status='passed',cases=results,reserved_rounds=ledger['reserved_rounds'],new_tree_fits=0,gpu_execution=False,test_accessed=False)
