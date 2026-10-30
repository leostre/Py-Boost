"""Bounded history-of-new-increments experiment; old blocks remain immutable."""
from pathlib import Path
import gc,json,subprocess,sys
import numpy as np
from budget_quality_worker import locked,write,digest,event
from abl_quality_core import fitted_control
from abl_stability_core import select_stable,NumericalTolerance
from abl_transfer_study import data_before_test,evaluated_rows
from abl_quality_worker import save_logits,check_native_environment
from abl_protected_core import load_protected

SEEDS={'yeast':(1304,1314,1324),'mediamill':(1104,1114,1124),'mnist':(3104,3114,3124)}
METHODS=('full','functional','current','mean','history')


def primary_choice(rows,metas,choice,budget):
    allowed={'full_unbounded':{'full'},'native_budget':{'full'},'deployed':set(m['family'] for m in metas.values())}.get(choice['group'],{choice['group']})
    eligible=[r for r in rows if metas[r['uid']]['family'] in allowed]
    mapped={u:dict(m,family='pyboost' if m['family']=='full' else 'compact') for u,m in metas.items()}
    cap=10**12 if choice['group']=='full_unbounded' else budget
    return select_stable(eligible,mapped,'deployed',choice['mode'],choice['objective'],cap,NumericalTolerance(1e-6,1e-12))


def develop_case(cfg,record,out,ledger_path):
    cid=f'{record["case"]["dataset"]}_{record["case"]["seed"]}';native=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    directory=out/'development'/cid;directory.mkdir(parents=True,exist_ok=True)
    config={**{k:cfg[k] for k in ('lr','lambda_l2','ntrees','decision_grid','decision_shrinkage','own_pool_limit_bytes')},'margin_scale':.5}
    worker=str(Path(__file__).resolve().parent/'abl_protected_worker.py')
    def execute(mode,**extra):
        work=out/'trajectories'/cid/mode;work.mkdir(parents=True,exist_ok=True)
        request=dict(case_id=cid,record=record,mode=mode,training_config=config,ledger_path=str(ledger_path),**extra)
        p=work/'request.json';write(p,request)
        subprocess.run([native,worker,str(p),str(work)],check=True,timeout=3600)
        path=work/'trajectory.json';r=json.loads(path.read_text())
        if r['status']!='complete':raise RuntimeError('protected path incomplete')
        return r,dict(mode=mode,path=str(path),sha256=digest(path))
    warm,warm_ref=execute('warmup');full,full_ref=execute('full',warmup=warm_ref,budget_bytes=10**12)
    if full['models'][-1]['trees']!=200:raise RuntimeError('full protected control must reach200trees')
    budget=int(np.floor(.75*(full['models'][-1]['cost_bytes']-4)))+4
    paths=[full_ref]
    for mode in METHODS[1:]:
        _,reference=execute(mode,warmup=warm_ref,budget_bytes=budget);paths.append(reference)
    data=data_before_test(record);metas={};rows=[]
    for reference in paths:
        t=locked(reference['path'],reference['sha256'])
        for item in t['models']:
            uid=item['uid'];m=load_protected(item['artifact']);z=m.raw(data['dev_x'])
            metas[uid]=dict(item,family=reference['mode'],source=dict(artifact=item['artifact'],sha256=item['sha256']),
                development_cache=save_logits(directory/(uid.replace('/','-')+'.npz'),z))
            rows.extend(evaluated_rows(uid,z,data,cfg,record['case']['task']));del m,z
    choices=[];modes=('raw','temperature_shifts') if record['case']['task']=='multiclass' else ('raw','shifts')
    for mode in modes:
        for objective in ('loss','f1_macro'):
            for group in ('full_unbounded','native_budget','functional','current','mean','history','deployed'):
                choice=dict(plan=0,mode=mode,objective=objective,group=group,budget_bytes=None if group=='full_unbounded' else budget)
                choices.append(dict(choice,chosen=primary_choice(rows,metas,choice,budget)))
    result=dict(case_id=cid,record=record,task=record['case']['task'],budget_bytes=budget,metas=metas,rows=rows,
        choices=choices,warmup=warm_ref,trajectories=paths,test_accessed=False,
        budget_scope='75percent actual full200 protected-block control serving envelope, not test-derived',
        hypothesis='history basis for new increments only; fixed depth3/minleaf10/rankhalf; old blocks immutable')
    path=directory/'case.json';write(path,result);event('abl_protected_case_saved',case=cid,path=str(path),sha256=digest(path))
    return dict(case_id=cid,path=str(path),sha256=digest(path))


def run(cfg,out):
    check_native_environment(cfg);source=locked(cfg['absolute_roles_manifest'],cfg['absolute_roles_manifest_sha256'])
    records=[dict(case=next(c for c in source['cases'] if c['dataset']==name and c['seed']==seed),dataset=source['datasets'][name])
        for name,seeds in SEEDS.items() for seed in seeds]
    pilot=cfg['phase']=='abl_protected_pilot';entries=[]
    if pilot:ledger=dict(reserved_rounds=0,attempts=[],phase_limit=2700)
    else:
        previous=locked(cfg['abl_protected_pilot_completed']['path'],cfg['abl_protected_pilot_completed']['sha256'])
        if previous['status']!='protected_pilot_development_complete':raise RuntimeError('technical pilot required')
        from audit_abl_protected import audit_development
        audit_development(previous['cases'],previous['training_ledger'])
        from abl_history_state_audit import audit_history_states
        write(out/'pilot_history_state_audit.json',audit_history_states(previous['cases']))
        entries=list(previous['cases']);ledger=dict(previous['training_ledger'],phase_limit=8000)
    ledger_path=out/'training_ledger.json';write(ledger_path,ledger)
    for record in records:
        cid=f'{record["case"]["dataset"]}_{record["case"]["seed"]}'
        if pilot and record['case']['seed']!=SEEDS[record['case']['dataset']][0]:continue
        if any(e['case_id']==cid for e in entries):continue
        entries.append(develop_case(cfg,record,out,ledger_path))
        write(out/'development_progress.json',dict(cases=entries,test_accessed=False))
    ledger=json.loads(ledger_path.read_text());from audit_abl_protected import audit_development
    audit=out/'development_audit.json';write(audit,audit_development(entries,ledger))
    from abl_history_state_audit import audit_history_states
    history_audit=out/'history_state_audit.json';write(history_audit,audit_history_states(entries))
    result=dict(status='protected_pilot_development_complete' if pilot else 'protected_development_complete',cases=entries,
        training_ledger=ledger,test_accessed=False,development_audit=dict(path=str(audit),sha256=digest(audit)),
        history_state_audit=dict(path=str(history_audit),sha256=digest(history_audit)))
    if not pilot:
        if len(entries)!=9:raise RuntimeError('all nine protected cases required')
        seal=out/'pretest_choices.json';write(seal,dict(status='protected_choices_sealed_before_test',cases=entries,
            development_audit_sha256=digest(audit),history_state_audit_sha256=digest(history_audit),test_used_for_selection=False));result['pretest_seal_sha256']=digest(seal)
    path=out/'report.json';write(path,result);event('abl_protected_development_complete',path=str(path),sha256=digest(path),cases=len(entries),new_rounds=ledger['reserved_rounds'])


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(sys.argv[2]);native=json.loads(Path(cfg['frozen_hasboost_environment']).read_text())['executable']
    if '--native' not in sys.argv:
        subprocess.run([native,__file__,sys.argv[1],sys.argv[2],'--native'],check=True,timeout=21600);return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(cfg,out)


if __name__=='__main__':main()
