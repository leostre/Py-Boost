"""Independent CPU conditional ensemble pilot; freeze before test join."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.a100_runner import ROOT,sha256,write_report,emit
from experiments.leaf_budget_runtime import load_expanded,load_roles
from experiments.uniform_threshold_core import ThresholdModel
from experiments.history_basis_study import raw
from experiments.final_ensemble_core import Component,feasible_pairs,choose_mixture
from experiments.output_calibration_core import calibrated_quality
from experiments.shared_leaf_basis import quality_guard


def final_ensemble_study(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_final_ensemble_core.py',
                    'tests/test_final_ensemble_shell.py'],cwd=ROOT,check=True,timeout=300)
    source=Path(config['cumulative_development_manifest'])
    roles_path=Path(config['cumulative_roles_manifest'])
    if sha256(source)!=config['cumulative_development_manifest_sha256'] or sha256(roles_path)!=config['cumulative_roles_manifest_sha256']:
        raise RuntimeError('frozen same-role source changed')
    parent=json.loads(source.read_text(encoding='utf-8'))
    manifest=json.loads(roles_path.read_text(encoding='utf-8'))
    sealed,locks=[],{str(source):sha256(source),str(roles_path):sha256(roles_path)}
    with threadpool_limits(limits=1):
        for record in parent['records']:
            case=next(c for c in manifest['cases'] if c['dataset']==record['dataset'] and c['seed']==record['seed'])
            if record['split_sha256']!=case['split_sha256']: raise RuntimeError('ensemble source roles changed')
            x,y=load_expanded(case['dataset'],manifest)
            roles=load_roles(case,len(x))
            locks[manifest['datasets'][case['dataset']]['path']]=manifest['datasets'][case['dataset']]['sha256']
            locks[case['split_file']]=case['split_sha256']
            if sha256(record['full']['artifact'])!=record['full']['sha256']:
                raise RuntimeError('frozen reference changed before seal')
            locks[record['full']['artifact']]=record['full']['sha256']
            components,models=[],{}
            for decision in record['decisions']:
                if decision['candidate'] is None: continue
                if sha256(decision['artifact'])!=decision['sha256']: raise RuntimeError('frozen component changed')
                locks[decision['artifact']]=decision['sha256']
                model=ThresholdModel.load(decision['artifact'])
                components.append(Component(decision['method'],decision['artifact'],decision['sha256'],
                    model.payload_bytes,model.prepare_inference().resident_bytes,model.parameter_count))
                models[decision['sha256']]=model
            pairs=feasible_pairs(components,case['outputs'],case['absolute_budget_bytes'])
            needed={c.sha256 for a,b,_ in pairs for c in (a,b)}
            cal={s:raw(models[s],x[roles['calibration']]) for s in needed}
            sel={s:raw(models[s],x[roles['selection']]) for s in needed}
            choice=choose_mixture(components,cal,y[roles['calibration']],sel,y[roles['selection']],case['task'],
                case['outputs'],case['absolute_budget_bytes'],record['reference_validation'],config)
            sealed.append(dict(dataset=case['dataset'],seed=case['seed'],task=case['task'],
                budget_bytes=case['absolute_budget_bytes'],full=record['full'],decision=choice,
                memory_feasible_pairs=len(pairs),component_count=len(components)))
        seal=dict(cases=sealed,file_locks=locks,test_evaluated=False,new_fits=0,
                  weights=[.25,.5,.75],whole_model_raw_score_average=True,folds_never_combined=True)
        path=output/'ensemble_selection_seal.json'
        write_report(path,seal)
        emit('ensemble_choices_sealed',sha256=sha256(path),cases=len(sealed),test_evaluated=False)
        results=[]
        for item in sealed:
            case=next(c for c in manifest['cases'] if c['dataset']==item['dataset'] and c['seed']==item['seed'])
            x,y=load_expanded(case['dataset'],manifest)
            roles=load_roles(case,len(x))
            full=ThresholdModel.load(item['full']['artifact'])
            if sha256(item['full']['artifact'])!=item['full']['sha256']: raise RuntimeError('full reference changed')
            reference=calibrated_quality(y[roles['test']],raw(full,x[roles['test']]),case['task'],full.decision_shifts)
            chosen=item['decision']['candidate']
            row=dict(dataset=case['dataset'],seed=case['seed'],state=item['decision']['state'],reference=reference,
                     memory_feasible_pairs=item['memory_feasible_pairs'],quality_memory_success=False)
            if chosen is not None:
                a,b=(ThresholdModel.load(chosen[k]['artifact']) for k in ('left','right'))
                scores=chosen['weight']*raw(a,x[roles['test']])+(1-chosen['weight'])*raw(b,x[roles['test']])
                quality=calibrated_quality(y[roles['test']],scores,case['task'],np.array(chosen['decision_shifts'],np.float32))
                row.update(quality=quality,payload_bytes=chosen['payload_bytes'],
                    quality_memory_success=bool(item['decision']['state']=='accepted' and quality_guard(quality,reference)),
                    selected_components=[chosen['left']['method'],chosen['right']['method']],weight=chosen['weight'])
            results.append(row)
            emit('ensemble_frozen_test_result',**row)
        for p,digest in locks.items():
            if sha256(p)!=digest: raise RuntimeError('component changed during evaluation')
    report.update(status='complete',results=results,new_fits=0,new_trees=0,test_is_reused_exploratory=True,
        selection_seal_sha256=sha256(path),serving_measured=False,
        next_serving_cases=[r['dataset'] for r in results if r['quality_memory_success']],
        interpretation='quality and numerical model arrays only; latency/process memory/interpretability not established')
