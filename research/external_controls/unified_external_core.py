"""Frozen data/calibration boundary usable without native training dependencies."""
import numpy as np
from budget_quality_worker import digest
from unified_frontier_core import replay_calibration,choose_policy,variants

def data_before_test(record):
    case,source=record['case'],record['dataset']
    if digest(case['split_file'])!=case['split_sha256'] or digest(source['path'])!=source['sha256']:
        raise RuntimeError('transfer source changed')
    with np.load(case['split_file'],allow_pickle=False) as f:
        roles={k:f[k].copy() for k in ('train','calibration','control','selection','test')}
    joined=np.concatenate(list(roles.values()))
    if not np.array_equal(np.sort(joined),np.arange(len(joined))):raise RuntimeError('invalid transfer roles')
    dev=np.r_[roles['calibration'],roles['control'],roles['selection']]
    with np.load(source['path'],allow_pickle=False) as f:
        x=f['features'];y=f['target']
        result=dict(train_x=x[roles['train']].astype(np.float32),train_y=y[roles['train']].copy(),
            dev_x=x[dev].astype(np.float32),dev_y=y[dev].copy(),roles=roles,
            sizes=[len(roles[k]) for k in ('calibration','control','selection')])
    if case['task']=='multiclass' and set(result['train_y'])!=set(range(case['outputs'])):
        raise RuntimeError('training output domain incomplete')
    return result


def store_replay(replay,path):
    arrays={'validation':replay['validation']}
    for name,parts in replay['arrays'].items():
        for key,value in zip(('prediction','loss','brier'),parts):arrays[name+'__'+key]=value
    np.savez_compressed(path,**arrays)
    return dict(path=str(path),sha256=digest(path),scores=replay['scores'],states=replay['states'])


def evaluate(raw,y,task,folds,cfg,out):
    replay=replay_calibration(raw,y,task,folds,cfg['decision_grid'],cfg['decision_shrinkage'])
    saved=store_replay(replay,out/'evaluation.npz');fitted=variants(raw,y,raw[:1],y[:1],task,cfg['decision_grid'],cfg['decision_shrinkage'])
    return [dict(protocol='crossfit',repeat=0,objective=obj,policy=(name:=choose_policy(replay['scores'],obj)),
        quality=replay['scores'][name],control=replay['scores'],refit_control=fitted[name],evaluation=saved,
        evaluation_scope='adaptive use of D; external evaluation not performed here') for obj in ('f1_macro','loss')]


def quality_row(evaluations,task):
    objective='loss' if task=='multiclass' else 'f1_macro'
    return next(r for r in evaluations if r['objective']==objective)
