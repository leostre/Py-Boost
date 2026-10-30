"""Independent pre-test role, raw-initialization and byte-budget checks."""
from pathlib import Path
import json

import numpy as np

from budget_quality_worker import digest,locked,native_load,write,event
from budget_quality_core import probabilities
from audit_calibration_pilot import independent_quality,policy_meta
from audit_abl_quality import policies,guard
from abl_quality_worker import before_test_data
from experiments.closed_budget_study import combine_actual


def independent_training_choice(rows,metas,choice,budget):
    allowed={'full_unbounded':{'full'},'native_budget':{'full'},'posthoc':{'posthoc'},
        'fixed':{'fixed'},'joint':{'joint'},'deployed':{'full','posthoc','fixed','joint'}}[choice['group']]
    objective=choice['objective'];candidates=[]
    for row in rows:
        meta=metas[row['uid']]
        if meta['family'] not in allowed:continue
        if choice['group']!='full_unbounded' and meta['cost_bytes']>budget:continue
        names=policies(choice['mode'],objective)
        if objective=='f1_macro':
            value=max(row['validation'][name]['f1_macro'] for name in names)
            names=[name for name in names if row['validation'][name]['f1_macro']>=value-1e-12]
        value=min(row['validation'][name]['loss'] for name in names)
        names=[name for name in names if row['validation'][name]['loss']<=value+1e-6]
        def preference(name):
            t,f=policy_meta(name);return int(t)+int(f>0),f,name
        name=min(names,key=preference)
        candidates.append(dict(uid=row['uid'],policy=name,validation=row['validation'][name],cost_bytes=meta['cost_bytes']))
    if not candidates:return None
    if objective=='f1_macro':
        value=max(c['validation']['f1_macro'] for c in candidates)
        candidates=[c for c in candidates if c['validation']['f1_macro']>=value-1e-12]
    value=min(c['validation']['loss'] for c in candidates)
    return min((c for c in candidates if c['validation']['loss']<=value+1e-6),key=lambda c:(c['cost_bytes'],c['uid']))


def saved_control(row,name):
    temperature,fraction=policy_meta(name);fit=row['refit']
    shifts=np.asarray(fit['temperature_shifts' if temperature else 'raw_shifts'],np.float32)
    return dict(temperature=fit['temperature'] if temperature else 1.,shifts=(fraction*shifts).astype(np.float32).tolist())


def probe_indices(labels):
    y=np.asarray(labels);indices=list(np.linspace(0,len(y)-1,min(64,len(y)),dtype=int))
    indices.extend(int(np.flatnonzero(y==label)[0]) for label in np.unique(y))
    return np.unique(indices)


def audit_development(entries,ledger):
    results=[];observed_attempts={}
    assert ledger['reserved_rounds']==40*len(ledger['attempts'])<=21600
    assert all(a['state']=='complete' for a in ledger['attempts'])
    for entry in entries:
        case=locked(entry['path'],entry['sha256']);data=before_test_data(case['record'],{})
        ncal,nctrl=data['sizes'][:2];fit_count=ncal+nctrl;selection=np.arange(fit_count,len(data['dev_y']))
        cp=case['calibration_plan']
        assert cp['scheme']=='pooled' and cp['temperature_indices']==cp['shift_indices']==list(range(fit_count))
        np.testing.assert_array_equal(cp['validation_indices'],selection)
        train_probe=probe_indices(data['train_y']);cal_probe=probe_indices(data['dev_y'][:ncal])
        xtrain=data['train_x'][train_probe];xcal=data['dev_x'][:ncal][cal_probe]
        worst=0.;projection_count=0;compact_inputs=0
        for reference in case['trajectories']:
            trajectory=locked(reference['path'],reference['sha256']);mode=reference['mode'];previous=None;previous_sha=None
            assert trajectory['controller_used_roles']==['train','calibration','control']
            assert not trajectory['selection_and_test_used_by_controller']
            by_block={m['block']:m for m in trajectory['models']}
            work=Path(reference['path']).parent/'worker_data.npz'
            with np.load(work,allow_pickle=False) as f:
                assert set(f.files)=={'train_x','train_y','validation_x','validation_y'}
                np.testing.assert_array_equal(f['train_x'],data['train_x'])
                np.testing.assert_array_equal(f['train_y'],data['train_y'])
                np.testing.assert_array_equal(f['validation_x'],data['dev_x'][:ncal])
                np.testing.assert_array_equal(f['validation_y'],data['dev_y'][:ncal])
            for cycle in trajectory['cycles']:
                block=cycle['block'];directory=Path(reference['path']).parent/f'block{block:02d}'
                initial=directory/'initial.npz';assert digest(initial)==cycle['initial_sha256']
                assert cycle['initial_checkpoint_sha256']==previous_sha
                with np.load(initial,allow_pickle=False) as f:
                    train_initial=f['train'][train_probe].copy();cal_initial=f['validation'][cal_probe].copy()
                    assert f['train'].shape==(len(data['train_y']),case['record']['case']['outputs'])
                expected_train=np.zeros_like(train_initial) if previous is None else previous.prepare_inference().raw(xtrain)
                expected_cal=np.zeros_like(cal_initial) if previous is None else previous.prepare_inference().raw(xcal)
                np.testing.assert_allclose(train_initial,expected_train.astype(np.float32),atol=2e-6,rtol=2e-7)
                np.testing.assert_allclose(cal_initial,expected_cal,atol=2e-9,rtol=1e-10)
                if previous is not None and previous.base.representation=='low_rank':compact_inputs+=1
                if cycle['reused_first40']:
                    assert block==0 and cycle['fit'] is None
                    source=trajectory['config']['warmups'][str(cycle['depth'])]
                    assert digest(source['artifact'])==source['sha256'];increment=native_load(source['artifact'])
                else:
                    fit=cycle['fit'];assert fit['initial_sha256']==cycle['initial_sha256'] and fit['data_sha256']==digest(work)
                    assert fit['trees']==40 and max(fit['train_replay_error'],fit['validation_replay_error'])<=2e-5
                    assert fit['minimum_leaf']==case['hyper']['minimum_leaf'] and fit['observed_minimum_leaf_support']>=fit['minimum_leaf']
                    assert fit['training_pool_peak_bytes']<=16*1024**3
                    key=(case['case_id'],mode,block);assert key not in observed_attempts;observed_attempts[key]=fit
                    assert digest(fit['artifact'])==fit['sha256'];increment=native_load(fit['artifact'])
                prior=increment.base.bias.copy() if cycle['reused_first40'] else np.zeros(case['record']['case']['outputs'],np.float32)
                proposal=combine_actual(previous,increment,prior)
                expected=expected_train+increment.prepare_inference().raw(xtrain)
                actual=proposal.prepare_inference().raw(xtrain)
                error=float(np.max(np.abs(probabilities(expected)-probabilities(actual))))
                worst=max(worst,error);assert error<=2e-5
                if cycle['action']=='projection_rejected':continue
                model=by_block[block];assert digest(model['artifact'])==model['sha256']
                current=native_load(model['artifact']);prepared=current.prepare_inference()
                assert model['cost_bytes']==prepared.resident_bytes+4
                if mode!='full':assert max(current.payload_bytes,prepared.resident_bytes)+4<=case['budget_bytes']
                if cycle['action']=='projection':
                    projection_count+=1;info=cycle['projection'];chosen=info['chosen']
                    assert guard(chosen['validation'],info['proposal_control'])
                    assert len(current.values)<=5*(len(proposal.base.coefficients)*(case['record']['case']['outputs']-1)+2*case['record']['case']['outputs'])//100
                    raw=prepared.raw(data['dev_x'][ncal:ncal+nctrl])
                    measured=independent_quality(data['dev_y'][ncal:ncal+nctrl],raw,
                        dict(temperature=chosen['temperature'],shifts=current.decision_shifts.tolist()))
                    for metric,value in measured.items():assert abs(value-chosen['validation'][metric])<2e-8
                else:
                    np.testing.assert_allclose(probabilities(prepared.raw(xtrain)),probabilities(expected),atol=2e-5,rtol=0)
                previous=current;previous_sha=model['sha256']
        for row in case['rows']:
            meta=case['metas'][row['uid']];cache=meta['development_cache']
            assert digest(cache['path'])==cache['sha256']
            with np.load(cache['path'],allow_pickle=False) as f:z=f['raw'][selection]
            for name in row['validation']:
                q=independent_quality(data['dev_y'][selection],z,saved_control(row,name))
                for metric,value in q.items():assert abs(value-row['validation'][name][metric])<2e-10
        for choice in case['choices']:
            assert independent_training_choice(case['rows'],case['metas'],choice,case['budget_bytes'])==choice['chosen']
        results.append(dict(case=case['case_id'],status='passed',choices=len(case['choices']),
            candidate_models=len(case['metas']),projection_events=projection_count,
            compact_initializations=compact_inputs,worst_proposal_probability_error=worst,
            gradient_audit_scope='all initial-file hashes plus class-covering/64-uniform-row numerical probes; no test data'))
    expected={(a['case'],a['mode'],a['block']):a['fit'] for a in ledger['attempts']}
    assert expected==observed_attempts
    return dict(status='passed',cases=results,reserved_rounds=ledger['reserved_rounds'],
        fit_attempts=len(observed_attempts),new_model_fits=0,gpu_execution=False,test_accessed=False)


def main():
    import sys
    from threadpoolctl import threadpool_limits
    source=locked(sys.argv[1],sys.argv[2])
    with threadpool_limits(limits=1):result=audit_development(source['cases'],source['training_ledger'])
    write(Path(sys.argv[3]),result)
    event('abl_training_development_audit_passed',cases=len(result['cases']),
        reserved_rounds=result['reserved_rounds'],path=sys.argv[3],sha256=digest(sys.argv[3]),test_accessed=False)


if __name__=='__main__':main()
