"""Strict numeric/binary ARFF and feature-group separation, no test metrics."""
import csv,shlex,io
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def integer_domain(values):
    values=np.asarray(values)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not np.array_equal(values,values.astype(np.int64)):
        raise ValueError('finite integer class values required')
    return [int(v) for v in np.unique(values)]


def numeric_arff(text):
    names=[];defaults=[];rows=[];data=False
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith('%'):continue
        if not data:
            if line.lower().startswith('@attribute'):
                parts=shlex.split(line);name=parts[1];kind=''.join(parts[2:]).lower()
                if name in names:raise ValueError('duplicate attribute')
                if kind in ('numeric','real','integer'):default=0.
                elif kind.startswith('{') and kind.endswith('}'):
                    domain=[x.strip().strip("'\"") for x in kind[1:-1].split(',')]
                    if set(domain)!={'0','1'}:raise ValueError('only binary nominal attributes supported')
                    default=float(domain[0])
                else:raise ValueError('unsupported attribute type')
                names.append(name);defaults.append(default)
            elif line.lower()=='@data':data=True
            continue
        if line.startswith('{'):
            if not line.endswith('}'):raise ValueError('unterminated sparse row')
            row=np.asarray(defaults,float).copy();seen=set()
            for entry in line[1:-1].split(','):
                if not entry.strip():continue
                index,value=entry.strip().split(None,1);index=int(index)
                if not 0<=index<len(names) or index in seen:raise ValueError('sparse domain/duplicate')
                seen.add(index);row[index]=float(value.strip().strip("'\""))
        else:
            values=next(csv.reader([line]));row=np.array([float(v.strip().strip("'\"")) for v in values])
            if len(row)!=len(names):raise ValueError('dense width mismatch')
        rows.append(row)
    if not data or not rows or not names:raise ValueError('empty/incomplete ARFF')
    matrix=np.asarray(rows,float)
    if not np.isfinite(matrix).all():raise ValueError('this cohort requires finite source data')
    return names,matrix


def preserve_official_test(train_x,train_y,test_x,test_y):
    tr=np.asarray(train_x,np.float32);te=np.asarray(test_x,np.float32)
    if tr.ndim!=2 or te.shape[1]!=tr.shape[1] or len(tr)!=len(train_y) or len(te)!=len(test_y):raise ValueError('source domain mismatch')
    test_keys={row.tobytes() for row in te};keep=np.array([row.tobytes() not in test_keys for row in tr])
    retained=np.flatnonzero(keep);x=np.r_[tr[keep],te];y=np.concatenate((train_y[keep],test_y))
    source_role=np.r_[np.zeros(len(retained),np.int8),np.ones(len(te),np.int8)]
    source_row=np.r_[retained,np.arange(len(te))].astype(np.int64)
    return x,y,source_role,source_row,dict(original_train_rows=len(tr),official_test_rows=len(te),
        removed_train_rows_crossing_test=int((~keep).sum()),retained_official_test_rows=len(te))


def official_roles(features,source_role,training_target,task,seed):
    x=np.asarray(features);fit=np.flatnonzero(source_role==0);test=np.flatnonzero(source_role==1)
    if len(training_target)!=len(fit):raise ValueError('only training targets may be used')
    _,groups=np.unique(x[fit],axis=0,return_inverse=True)
    proposals=list(GroupShuffleSplit(n_splits=64,test_size=.2,random_state=seed).split(x[fit],groups=groups))
    if task=='multiclass':
        classes=set(np.asarray(training_target).tolist())
        proposals=[(a,b) for a,b in proposals if set(np.asarray(training_target)[a])==classes]
    if not proposals:raise ValueError('no group-separated complete training domain')
    a,b=min(proposals,key=lambda pair:abs(len(pair[1])-.2*len(fit)))
    train,dev=fit[a],fit[b];parts=np.array_split(dev,3)
    result=dict(train=train,calibration=parts[0],control=parts[1],selection=parts[2],test=test)
    assert not set(groups[a])&set(groups[b]);assert np.array_equal(np.sort(np.concatenate(list(result.values()))),np.arange(len(x)))
    assert not {v.tobytes() for v in x[fit]}&{v.tobytes() for v in x[test]}
    return result,dict(train_feature_groups=len(np.unique(groups[a])),development_feature_groups=len(np.unique(groups[b])),
        official_test_preserved=True,split_selection='closest20percent row count among64 fixed group proposals; multiclass train-domain check only')
