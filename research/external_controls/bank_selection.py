"""Development-only five-task banks and task-ID specialist subset selection."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from matched_data import digest
from quality_worker import load_inputs,write
from external_quality_core import choose
from experiments.uniform_threshold_core import ThresholdModel,choice_key
from experiments.output_calibration_core import calibrated_quality
from experiments.shared_leaf_basis import quality_guard
from external_bank_core import useful_specialist,select_subsets


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    output=Path(sys.argv[2])
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_bank_selection.py'],check=True,timeout=300)
    locks={}
    def locked(path,value):
        if digest(path)!=value: raise RuntimeError('bank development source changed')
        locks[str(path)]=value
        return Path(path)
    sources={}
    for key,spec in cfg['bank_sources'].items():
        sources[key]=json.loads(locked(spec['path'],spec['sha256']).read_text(encoding='utf-8'))
    records,specialists,eligibility=[],[],[]
    with threadpool_limits(limits=1):
        for case_id in cfg['quality_cases']:
            case,entry,data=load_inputs(cfg,case_id)
            locked(entry['path'],entry['sha256'])
            locked(case['split_file'],case['split_sha256'])
            native=next(r for r in sources['native']['records'] if r['case_id']==case_id and r['family']=='full')
            tfm=next(r for r in sources['external']['records'] if r['case_id']==case_id and r['family']=='tabpfn_auto')
            if native['case']['split_sha256']!=case['split_sha256'] or tfm['case']['split_sha256']!=case['split_sha256']:
                raise RuntimeError('bank model roles differ')
            pool=[]
            base_cache,raw_cache={},{}
            def add(identity,path,value,shifts=None,origin=None):
                if path not in base_cache: base_cache[path]=ThresholdModel.load(locked(path,value))
                elif locks[str(path)]!=value: raise RuntimeError('same model path has conflicting hashes')
                model=base_cache[path]
                if shifts is not None: model=replace(model,decision_shifts=np.array(shifts,np.float32))
                if path not in raw_cache: raw_cache[path]=model.prepare_inference().raw(data['selection'][0])
                z=raw_cache[path]
                q=calibrated_quality(data['selection'][1],z,'multiclass',model.decision_shifts)
                cost=max(model.payload_bytes,model.prepare_inference().resident_bytes)
                pool.append(dict(candidate_id=identity,model=model,validation=q,serving_bytes=cost,
                    parameters=model.parameter_count,origin=origin or identity))
            # Reference200 depth5 uses the exact frozen outer-calibration choice.
            reference_rows=[r for r in native['candidates'] if r['candidate_id']=='d5-t200']
            ref=choose(reference_rows)
            ref_artifact=native['artifacts']['d5-t200']
            add('reference200',ref_artifact['artifact'],ref_artifact['sha256'],ref['decision_shifts'])
            reference=pool[0]
            add('ordinary_selected',native['selected_artifact'],native['selected_sha256'])
            ordinary=pool[-1]
            for record in sources['native']['records']:
                if record['case_id']!=case_id: continue
                for identity,item in record['artifacts'].items():
                    for index,row in enumerate(r for r in record['candidates'] if r['candidate_id']==identity):
                        add(record['family']+'-'+identity+f'-s{index}',item['artifact'],item['sha256'],row['decision_shifts'],record['family'])
            for family in ('validation','reserve','reinvest'):
                for old in sources[family]['records']:
                    if old['dataset']!=case['dataset'] or old['seed']!=case['seed']: continue
                    if old['split_sha256']!=case['split_sha256']: raise RuntimeError('compact bank roles differ')
                    for d in old['decisions']:
                        if d.get('artifact'):
                            add(family+'-'+d['method'],d['artifact'],d['sha256'],origin=family+'-'+d['method'])
            low=[p for p in pool if p['model'].base.representation=='low_rank' and quality_guard(p['validation'],reference['validation'])]
            all_eligible=[p for p in pool if quality_guard(p['validation'],reference['validation'])]
            key=lambda p:(p['serving_bytes'],-p['validation']['f1_macro'],p['validation']['loss'],p['candidate_id'])
            compact=min([reference,*low],key=key)
            joint=min(all_eligible,key=key)
            for family,item in (('full_reference',reference),('ordinary_selected',ordinary),('compact_or_full',compact),('joint_selected',joint)):
                model=item['model']
                path=output/'models'/case_id/(family+'.npz')
                path.parent.mkdir(parents=True,exist_ok=True)
                model.save(path)
                records.append(dict(family=family,dataset=case['dataset'],seed=case['seed'],case_id=case_id,
                    path=str(path),sha256=digest(path),data=entry,split_file=case['split_file'],split_sha256=case['split_sha256'],
                    source_candidate=item['candidate_id'],representation=model.base.representation,
                    validation=item['validation'],reference_validation=reference['validation'],
                    serving_bytes=item['serving_bytes'],parameters=model.parameter_count,rank=model.base.coefficients.shape[1],
                    trees=model.base.n_trees,leaves=len(model.base.coefficients),nnz=len(model.values),
                    fallback_full=bool(item is reference),task='multiclass',outputs=case['outputs'],
                    task_id=case_id,development_pass=quality_guard(item['validation'],reference['validation'])))
            options=[p for p in pool if useful_specialist(p['validation'],tfm['chosen']['validation'])]
            eligibility.append(dict(case_id=case_id,eligible_candidates=len(options),candidates=len(pool),
                                    tfm_reference=tfm['chosen']['validation']))
            if options:
                best=min(options,key=lambda p:(-p['validation']['f1_macro'],p['validation']['loss'],p['serving_bytes'],p['candidate_id']))
                path=output/'models'/case_id/'specialist.npz'
                best['model'].save(path)
                q,r=best['validation'],tfm['chosen']['validation']
                specialists.append(dict(case_id=case_id,candidate_id=case_id+'-'+best['candidate_id'],
                    source_candidate=best['candidate_id'],path=str(path),sha256=digest(path),serving_bytes=best['serving_bytes'],
                    validation=q,tfm_reference=r,macro_relative_gain=q['f1_macro']/r['f1_macro']-1,
                    loss_relative_gain=1-q['loss']/r['loss']))
            print(json.dumps(dict(event='bank_development_selected',case_id=case_id,compact=compact['candidate_id'],
                joint=joint['candidate_id'],eligible_specialists=len(options),test_evaluated=False)),flush=True)
    if any(digest(p)!=h for p,h in locks.items()): raise RuntimeError('bank inputs changed')
    subsets=select_subsets(specialists,cfg['specialist_extra_bytes'])
    write(output/'bank_manifest.json',dict(status='development_complete',records=records,specialists=specialists,
        specialist_subsets=subsets,eligibility=eligibility,input_locks=locks,new_fits=0,new_trees=0,
        test_evaluated=False,tfm_contexts_not_free=True,pool_is_exploratory=True,
        main_source_development_sha256=cfg['bank_sources']['native']['sha256']))
    print(json.dumps(dict(event='bank_selection_complete',models=len(records),specialist_tasks=len(specialists),
        manifest_sha256=digest(output/'bank_manifest.json'),new_fits=0)),flush=True)


if __name__=='__main__': main()
