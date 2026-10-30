"""Frozen transfer inputs, including a new group-separated GenBase benchmark."""
from pathlib import Path
import json,sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import arff
from sklearn.model_selection import GroupShuffleSplit
from budget_quality_worker import locked,digest,write,event


def read_genbase(arff_path,xml_path):
    rows,metadata=arff.loadarff(arff_path)
    labels=[node.attrib['name'] for node in ET.parse(xml_path).getroot()]
    names=list(metadata.names());assert 'protein' in names and len(labels)==27
    columns=[n for n in names if n!='protein' and n not in labels];assert len(columns)==1185
    def binary(name,positive,negative):
        values=rows[name];assert np.isin(values,[positive,negative]).all(),name
        return (values==positive).astype(np.float32)
    x=np.column_stack([binary(n,b'YES',b'NO') for n in columns])
    y=np.column_stack([binary(n,b'1',b'0') for n in labels]);assert len(x)==662
    return x,y,labels


def grouped_roles(features,seed):
    _,groups=np.unique(features,axis=0,return_inverse=True);ids=np.arange(len(features))
    def split(x,g,seed):
        # Select only by row count, before labels are exposed to this function.
        proposals=GroupShuffleSplit(n_splits=64,test_size=.2,random_state=seed).split(x,groups=g)
        return min(proposals,key=lambda pair:abs(len(pair[1])-.2*len(x)))
    fit,test=split(features,groups,seed)
    a,b=split(features[fit],groups[fit],seed+1)
    train,dev=fit[a],fit[b];parts=np.array_split(dev,3)
    roles=dict(train=train,test=test,calibration=parts[0],control=parts[1],selection=parts[2])
    assert not set(groups[train])&set(groups[dev]) and not set(groups[test])&set(groups[fit])
    assert np.array_equal(np.sort(np.concatenate(list(roles.values()))),ids)
    return roles,dict(unique_feature_groups=int(groups.max()+1),role_groups=dict(train=len(np.unique(groups[train])),development=len(np.unique(groups[dev])),test=len(np.unique(groups[test]))))


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);spec=cfg['unified_transfer_data'];old=locked(spec['g1']['path'],spec['g1']['sha256'])
    cases=[];datasets={}
    for ref in old['cases']:
        case=locked(ref['path'],ref['sha256']);cid=case['case_id']
        if cid not in ('letter_5104','optdigits_5204'):continue
        record=case['record'];c,d=record['case'],record['dataset'];assert digest(c['split_file'])==c['split_sha256'] and digest(d['path'])==d['sha256']
        cases.append(c);datasets[c['dataset']]=d
    raw=Path(__file__).parent/'unified_raw_genbase';receipt=locked(raw/'source.json',spec['source_receipt_sha256'])
    for name,ref in receipt['members'].items():assert digest(raw/name)==ref['sha256']
    assert digest(raw/'genbase.rar')==receipt['archive_sha256']
    x,y,labels=read_genbase(raw/'genbase.arff',raw/'genbase.xml');roles,group_info=grouped_roles(x,5304)
    directory=out/'genbase_5304';directory.mkdir(exist_ok=True);data=directory/'data.npz';splits=directory/'splits.npz'
    np.savez_compressed(data,features=x,target=y);np.savez(splits,**roles)
    dataset=dict(path=str(data),sha256=digest(data),source=receipt,shape=list(x.shape),outputs=len(labels),protein_identifier_removed=True)
    case=dict(dataset='genbase',seed=5304,task='multilabel',outputs=27,split_file=str(splits),split_sha256=digest(splits),
        data_path=str(data),data_sha256=digest(data),sizes={k:len(v) for k,v in roles.items()},protocol_version=4,existing_case_preserved=False,
        train_meta=dict(protocol='grouped80-20-nontest',rows=len(roles['train'])),
        supports_train=y[roles['train']].sum(0).astype(int).tolist(),grouping=group_info,
        split_uses_labels=False,test_labels_used_for_selection=False,interpretation='new task for this series; one seed; no prior hyperparameter adaptation')
    cases.append(case);datasets['genbase']=dataset
    manifest=out/'manifest.json';write(manifest,dict(cases=cases,datasets=datasets,genbase_labels=labels))
    result=dict(status='unified_transfer_data_complete',manifest=dict(path=str(manifest),sha256=digest(manifest)),
        cases=[dict(dataset=c['dataset'],seed=c['seed'],sizes=c['sizes']) for c in cases],genbase_grouping=group_info,
        genbase_train_zero_support_labels=int(np.count_nonzero(y[roles['train']].sum(0)==0)),
        old_cases_test_unchanged=True,new_tree_fits=0,test_evaluated=False,
        limitation='Letter/Optdigits old test previously viewed; GenBase new one-task transfer, no broad independence claim')
    write(out/'report.json',result);event('unified_transfer_data_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'),**result)


if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
