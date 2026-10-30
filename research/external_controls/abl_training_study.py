"""Pilot/full orchestration with nine sealed development cases and no test yet."""
from collections import Counter
import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from budget_quality_worker import digest,write,locked,event,native_load
from abl_quality_worker import before_test_data,evaluate_candidate,save_logits,check_native_environment
from abl_quality_core import fitted_control
from abl_stability_core import NumericalTolerance,select_stable
from calibration_pilot_core import make_plans,Scheme

SEEDS={'helena':(1204,1214,1224),'mnist':(3104,3114,3124),'rt_iot2022':(3204,3214,3224)}
MODES=('full','fixed','joint')


def make_training_plan(cfg):
    width=locked(cfg['abl_width_completed']['path'],cfg['abl_width_completed']['sha256'])
    audited=locked(cfg['abl_width_audit_completed']['path'],cfg['abl_width_audit_completed']['sha256'])
    if width['status']!='complete' or audited['status']!='passed':raise RuntimeError('width audit required')
    root=Path(cfg['abl_width_completed']['path']).parent
    seal=locked(root/'pretest_choices.json',width['pretest_seal_sha256'])
    roles=locked(cfg['absolute_roles_manifest'],cfg['absolute_roles_manifest_sha256'])
    plans=[]
    for name,seeds in SEEDS.items():
        first=f'{name}_{seeds[0]}';entry=next(c for c in seal['cases'] if c['case_id']==first)
        previous=locked(entry['path'],entry['sha256'])
        chosen=next(c['chosen'] for c in previous['budget_choices'] if c['mode']=='temperature_shifts'
            and c['objective']=='loss' and c['group']=='native' and c['budget_ratio']==4)
        meta=previous['metas'][chosen['uid']]
        if meta['family']!='pyboost':raise RuntimeError('hyperparameter source must be ordinary PyBoost')
        hyper=dict(depth=meta['source']['depth'],minimum_leaf=meta['minimum_leaf'])
        budget=int(np.floor(.75*previous['reference_b0_bytes']))+4
        for seed in seeds:
            case=next(c for c in roles['cases'] if c['dataset']==name and c['seed']==seed)
            record=dict(case=case,dataset=roles['datasets'][name])
            if seed==seeds[0] and case['split_sha256']!=previous['record']['case']['split_sha256']:
                raise RuntimeError('first-seed warmup and current roles differ')
            warmups={}
            if seed==seeds[0]:
                for depth in (3,5):
                    m=next(m for m in previous['metas'].values() if m['family']=='pyboost'
                        and m['source']['depth']==depth and m['source']['rounds']==40
                        and m['minimum_leaf']==hyper['minimum_leaf'])
                    warmups[str(depth)]=m['source']
            plans.append(dict(case_id=f'{name}_{seed}',record=record,hyper=hyper,budget_bytes=budget,
                reference_b0_bytes=previous['reference_b0_bytes'],warmups=warmups,
                hyperparameter_development_source=dict(case=first,chosen_uid=chosen['uid'],
                    objective='loss',mode='temperature_shifts',budget_ratio=4,validation=chosen['validation'])))
    if len(plans)!=9:raise RuntimeError('nine frozen training cases required')
    return plans


def training_choice(rows,metas,group,mode,objective,budget):
    kinds={'full_unbounded':{'full'},'native_budget':{'full'},'posthoc':{'posthoc'},
        'fixed':{'fixed'},'joint':{'joint'},'deployed':{'full','posthoc','fixed','joint'}}[group]
    eligible=[r for r in rows if metas[r['uid']]['family'] in kinds]
    mapped={uid:dict(m,family='pyboost' if m['family']=='full' else 'compact') for uid,m in metas.items()}
    cap=max((m['cost_bytes'] for m in mapped.values()),default=0)+1 if group=='full_unbounded' else budget
    return select_stable(eligible,mapped,'deployed',mode,objective,cap,NumericalTolerance(1e-6,1e-12))


def case_development(cfg,plan,output,ledger_path):
    case_id=plan['case_id'];directory=output/'development'/case_id;directory.mkdir(parents=True,exist_ok=True)
    native_exe=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    worker=Path(__file__).resolve().parent/'abl_training_worker.py'
    training_config={**{k:cfg[k] for k in ('lr','lambda_l2','ntrees','decision_grid','decision_shrinkage','own_pool_limit_bytes')},
        'margin_scale':.5,'bounded_tree_execution':True}
    paths=[]
    for mode in MODES:
        work=output/'trajectories'/case_id/mode;work.mkdir(parents=True,exist_ok=True)
        request=dict(plan,mode=mode,training_config=training_config,ledger_path=str(ledger_path))
        request_path=work/'request.json';write(request_path,request)
        subprocess.run([native_exe,str(worker),str(request_path),str(work)],check=True,timeout=10800)
        path=work/'trajectory.json';trajectory=json.loads(path.read_text())
        if trajectory['status']!='complete' or trajectory['selection_and_test_used_by_controller']:
            raise RuntimeError('native trajectory did not satisfy role contract')
        paths.append(dict(mode=mode,path=str(path),sha256=digest(path)))
        event('abl_training_native_path_saved',case=case_id,mode=mode,models=len(trajectory['models']),
            projection_events=trajectory['projection_events'],compact_gradient_blocks=trajectory['blocks_initialized_from_compact'])
    data=before_test_data(plan['record'],{});bplan=next(p for p in make_plans(data['dev_y'],data['sizes'],
        0,(1.,),cfg['calibration_pilot']['seed']) if p.scheme is Scheme.POOLED)
    metas={};rows=[];timing=[]
    for item in paths:
        trajectory=locked(item['path'],item['sha256'])
        for model in (*trajectory['models'],*trajectory['posthoc']):
            uid=model['uid']
            if uid in metas:raise RuntimeError('duplicate training snapshot identity')
            if digest(model['artifact'])!=model['sha256']:raise RuntimeError('native snapshot changed')
            restored=native_load(model['artifact']);z=restored.prepare_inference().raw(data['dev_x'])
            meta=dict(model,source=dict(artifact=model['artifact'],sha256=model['sha256']),
                training_projection_events_in_prefix=sum(c['action']=='projection' for c in trajectory['cycles'] if c['block']<=model['block']),
                compact_gradient_blocks_in_prefix=sum(c['initial_from_representation']=='low_rank' for c in trajectory['cycles'] if c['block']<=model['block']),
                development_cache=save_logits(directory/(uid.replace('/','-')+'-logits.npz'),z))
            metas[uid]=meta;started=perf_counter()
            rows.extend(evaluate_candidate(uid,z,data['dev_y'],[bplan],cfg))
            timing.append(dict(uid=uid,seconds=perf_counter()-started));del restored,z
    if not any(m['family']=='full' for m in metas.values()):raise RuntimeError('complete full trajectory required')
    choices=[]
    for mode in ('raw','temperature','shifts','temperature_shifts'):
        for objective in ('loss','f1_macro'):
            for group in ('full_unbounded','native_budget','posthoc','fixed','joint','deployed'):
                chosen=training_choice(rows,metas,group,mode,objective,plan['budget_bytes'])
                choices.append(dict(scheme='pooled',repeat=0,plan=0,mode=mode,objective=objective,
                    group=group,budget_bytes=None if group=='full_unbounded' else plan['budget_bytes'],chosen=chosen))
    case=dict(case_id=case_id,record=plan['record'],hyper=plan['hyper'],budget_bytes=plan['budget_bytes'],
        metas=metas,rows=rows,choices=choices,trajectories=paths,output_calibration_timing=timing,
        final_selection_role='original selection only; calibration/control already used by controller',
        calibration_plan=dict(scheme='pooled',repeat=0,fraction=1,
            temperature_indices=list(bplan.refit_temperature),shift_indices=list(bplan.refit_shifts),
            validation_indices=list(bplan.folds[0].validation)),test_accessed=False)
    path=directory/'case.json';write(path,case)
    return dict(case_id=case_id,path=str(path),sha256=digest(path),candidates=len(metas))


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'));output=args.output
    environment=check_native_environment(cfg);plans=make_training_plan(cfg)
    write(output/'training_plan.json',dict(plans=plans,model_trajectories=27,new_tree_cap=21600,
        budget_ratio=.75,final_calibration='pooled',new_packages=False))
    entries=[];pilot=cfg['phase']=='abl_training_pilot'
    if pilot:ledger=dict(reserved_rounds=0,attempts=[],phase_round_limit=2400)
    else:
        previous=locked(cfg['abl_training_pilot_completed']['path'],cfg['abl_training_pilot_completed']['sha256'])
        if previous['status']!='pilot_development_complete':raise RuntimeError('technical pilot required')
        if previous['training_plans']!=plans:raise RuntimeError('frozen pilot plan changed')
        native_exe=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
        subprocess.run([native_exe,str(Path(__file__).resolve().parent/'abl_training_development_audit.py'),
            cfg['abl_training_pilot_completed']['path'],cfg['abl_training_pilot_completed']['sha256'],
            str(output/'pilot_development_audit.json')],check=True,timeout=1800)
        entries=list(previous['cases']);ledger=dict(previous['training_ledger'],phase_round_limit=21600)
    ledger_path=output/'training_ledger.json';write(ledger_path,ledger)
    for plan in plans:
        if pilot and plan['case_id']!='helena_1204':continue
        if any(c['case_id']==plan['case_id'] for c in entries):continue
        entries.append(case_development(cfg,plan,output,ledger_path))
        write(output/'development_progress.json',dict(cases=entries,new_tree_cap=21600,test_accessed=False))
    ledger=json.loads(ledger_path.read_text())
    if ledger['reserved_rounds']>21600 or any(a['state']!='complete' for a in ledger['attempts']):
        raise RuntimeError('incomplete or excessive native training attempts')
    result=dict(status='pilot_development_complete' if pilot else 'training_development_complete',
        cases=entries,training_plans=plans,training_ledger=ledger,native_environment=environment,
        test_accessed=False,model_trajectories=len(entries)*3)
    if not pilot:
        if len(entries)!=9:raise RuntimeError('all nine cases required before test')
        development_report=output/'development_for_audit.json';write(development_report,result)
        subprocess.run([native_exe,str(Path(__file__).resolve().parent/'abl_training_development_audit.py'),
            str(development_report),digest(development_report),str(output/'development_audit.json')],check=True,timeout=6000)
        seal=output/'pretest_choices.json'
        write(seal,dict(status='abl_training_choices_sealed_before_test',cases=entries,
            source_plan_sha256=digest(output/'training_plan.json'),reserved_rounds=ledger['reserved_rounds'],
            development_audit_sha256=digest(output/'development_audit.json')))
        result['pretest_seal_sha256']=digest(seal)
    write(output/'report.json',result)
    event('abl_training_development_complete',phase=cfg['phase'],cases=len(entries),
        reserved_rounds=ledger['reserved_rounds'],path=str(output/'report.json'),sha256=digest(output/'report.json'),test_accessed=False)


if __name__=='__main__':main()
