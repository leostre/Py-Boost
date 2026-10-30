"""Deterministic training-data reclamation; former test indices stay fixed."""
from pathlib import Path
import copy
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from budget_quality_worker import digest


def reclaim_roles(roles,development_target,outputs,task,seed,train_fraction_non_test=.8):
    if not .5<=train_fraction_non_test<1:raise ValueError('registered training fraction outside range')
    old_train=np.asarray(roles['train'],np.int64);test=np.asarray(roles['test'],np.int64)
    old_dev=np.concatenate([roles[k] for k in ('calibration','control','selection')]).astype(np.int64)
    target=np.asarray(development_target)
    if len(target)!=len(old_dev):raise ValueError('development target alignment changed')
    y=np.eye(outputs)[target.astype(int)] if task=='multiclass' else target
    desired_dev=max(3,int(np.ceil((len(old_train)+len(old_dev))*(1-train_fraction_non_test))))
    if desired_dev>=len(old_dev):raise ValueError('reclamation must actually expand train')
    moved,kept=next(MultilabelStratifiedShuffleSplit(n_splits=1,test_size=desired_dev/len(old_dev),random_state=seed+20261028).split(np.zeros((len(old_dev),1)),y))
    dev=old_dev[kept];parts=np.array_split(dev,3)
    new=dict(train=np.r_[old_train,old_dev[moved]],test=test.copy(),calibration=parts[0],control=parts[1],selection=parts[2])
    assert set(old_train)<=set(new['train']) and np.array_equal(new['test'],test)
    joined=np.concatenate(list(new.values()));assert len(joined)==len(set(joined.tolist()))
    assert all(len(v)>0 for v in new.values())
    return new,dict(old_train_rows=len(old_train),new_train_rows=len(new['train']),old_development_rows=len(old_dev),
        new_development_rows=len(dev),moved_to_training=len(moved),train_fraction_non_test_actual=len(new['train'])/(len(old_train)+len(old_dev)),
        note='legacy cal/control/selection keys are storage aliases only; all are pooled into one D')


def expanded_record(record,out,fraction=.8):
    result=copy.deepcopy(record);old=record['case'];dataset=record['dataset']
    assert digest(old['split_file'])==old['split_sha256'] and digest(dataset['path'])==dataset['sha256']
    with np.load(old['split_file'],allow_pickle=False) as f:roles={k:f[k].copy() for k in ('train','calibration','control','selection','test')}
    dev=np.concatenate([roles[k] for k in ('calibration','control','selection')])
    with np.load(dataset['path'],allow_pickle=False) as f:target=f['target'][dev].copy()
    new,info=reclaim_roles(roles,target,old['outputs'],old['task'],old['seed'],fraction)
    path=Path(out)/'expanded_splits.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**new)
    result['case'].update(split_file=str(path),split_sha256=digest(path),sizes={k:len(v) for k,v in new.items()},
        protocol_version=4,existing_case_preserved=False,train_meta={'rows':len(new['train']),'protocol':'expanded-training-v1'})
    result['case'].pop('supports',None)
    result['data_reuse']=dict(info,original_split_file=old['split_file'],original_split_sha256=old['split_sha256'],test_indices_unchanged=True)
    return result
