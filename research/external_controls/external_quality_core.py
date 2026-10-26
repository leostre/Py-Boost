"""Pure external-class alignment and selection using frozen HASBoost metrics."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'contracts'))
from experiments.uniform_threshold_core import threshold_variants,choice_key


def canonical_raw(values,classes,outputs,*,probabilities=False):
    a=np.asarray(values,dtype=np.float64)
    classes=np.asarray(classes)
    if not np.array_equal(classes,np.arange(outputs)):
        raise ValueError('external output classes not canonical')
    if not probabilities and (a.ndim==1 or a.ndim==2 and a.shape[1]==1):
        if outputs!=2: raise ValueError('scalar margin only valid for binary')
        a=np.column_stack((np.zeros(a.size),a.ravel()))
    if a.ndim!=2 or a.shape[1]!=outputs or not np.isfinite(a).all():
        raise ValueError('invalid external scores')
    if probabilities:
        if np.any(a<0) or np.any(a>1) or not np.allclose(a.sum(axis=1),1,atol=2e-5):
            raise ValueError('invalid external probabilities')
        a=np.log(np.clip(a,1e-12,1))
    return a


def calibrated_rows(raw_cal,ycal,raw_selection,ysel,task,identity,cost,grid,shrinkages):
    return [dict(candidate_id=identity,**cost,**v) for v in threshold_variants(
        raw_cal,ycal,raw_selection,ysel,task,grid,shrinkages)]


def choose(rows):
    if not rows: raise ValueError('empty external candidate family')
    return min(rows,key=choice_key)


def quality_close(actual,expected):
    if (abs(actual['loss']-expected['loss'])>1e-6 or
            any(abs(actual[k]-expected[k])>1e-7 for k in ('f1_macro','f1_weighted'))):
        raise RuntimeError('selected development quality does not replay')


def audit_then_test(records,probe,seal,evaluate):
    evidence=[probe(record) for record in records]
    audit=seal(evidence)
    results=[evaluate(record,audit) for record in records]
    return evidence,results


def tree_counts(dumps):
    """Independent JSON tree traversal, counts actual XGBoost scalar leaves."""
    trees=leaves=internal=0
    def visit(node):
        nonlocal leaves,internal
        if 'leaf' in node:
            leaves+=1
        else:
            children=node.get('children')
            if not isinstance(children,list) or len(children)!=2:
                raise ValueError('nonbinary native tree')
            internal+=1
            for child in children: visit(child)
    for item in dumps:
        trees+=1
        visit(item)
    return dict(trees=trees,leaves=leaves,internal_nodes=internal,leaf_parameters=leaves)
