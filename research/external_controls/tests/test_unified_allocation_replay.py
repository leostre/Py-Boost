import copy
import budget_quality_worker
import unified_final_evaluation as worker


def test_replay_uses_sealed_ids_even_if_a_different_model_has_better_quality(monkeypatch):
    objects={};options={};inventory=[];refs=[]
    for cid in ['a','b']:
        models=[]
        for uid,cost,loss in [('small',10,3.),('full',20,1.)]:
            e=[dict(objective='loss',quality=dict(loss=loss),refit_control=dict(temperature=1.,shifts=[0.,0.]))]
            models.append(dict(uid=uid,family='ordinary',artifact=cid+uid,sha256=cid+uid,evaluations=e))
        objects[cid]=dict(task='multiclass',models=models,record={'marker':cid})
        refs.append(dict(case=cid,path=cid,sha256='checked-by-locked'))
        options[cid]=[dict(uid='small',cost_bytes=10),dict(uid='full',cost_bytes=20)]
        inventory.append(dict(case=cid,uid='full',bank='full'))
    objects['seal']=dict(options=options,inventory=inventory,bank_frontiers={'all':[dict(budget=30,status='feasible',cost_bytes=30,utility=0.,choices=['small','full'])]})
    report=dict(cases=['a','b'],case_representations=refs,seal=dict(path='seal',sha256='checked-by-locked'))
    monkeypatch.setattr(worker,'locked',lambda path,sha:copy.deepcopy(objects[path]))
    choices,plans=worker.choose_allocations(report,'cohort')
    assert [c['uid'] for c in choices if c['allocation_key'] in plans[0]['member_keys']]==['full','small']
    by={c['allocation_key']:c for c in choices}
    assert [(by[k]['case'],by[k]['uid']) for k in plans[0]['member_keys']]==[('a','small'),('b','full')]
    assert sum(by[k]['cost_bytes'] for k in plans[0]['member_keys'])==30
    assert len(choices)==3  # full references and distinct selected members, no duplicated evaluations
