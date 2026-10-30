"""Fresh cohort provenance dependency. No fitting, test metrics or method selection."""
from pathlib import Path
import json,sys,xml.etree.ElementTree as ET
import numpy as np
from budget_quality_worker import digest,write,locked,event
from abl_quality_worker import check_native_environment
from adaptive_data_core import numeric_arff,preserve_official_test,official_roles,integer_domain


def read_pair(directory,name):
    if name in ('emotions','medical'):
        labels=[n.attrib['name'] for n in ET.parse(directory/(name+'.xml')).getroot()]
        names,tr=numeric_arff((directory/(name+'-train.arff')).read_text(encoding='utf-8-sig'))
        other,te=numeric_arff((directory/(name+'-test.arff')).read_text(encoding='utf-8-sig'))
        assert names==other;columns=[i for i,n in enumerate(names) if n not in labels];indices=[names.index(n) for n in labels]
        assert np.isin(tr[:,indices],[0,1]).all() and np.isin(te[:,indices],[0,1]).all()
        return tr[:,columns],tr[:,indices].astype(np.int8),te[:,columns],te[:,indices].astype(np.int8),'multilabel',labels
    tr=np.loadtxt(directory/('pendigits.tra' if name=='pendigits' else 'sat.trn'),delimiter=',' if name=='pendigits' else None)
    te=np.loadtxt(directory/('pendigits.tes' if name=='pendigits' else 'sat.tst'),delimiter=',' if name=='pendigits' else None)
    domain=integer_domain(tr[:,-1]);mapping={v:i for i,v in enumerate(domain)}
    assert set(te[:,-1].astype(int))<=set(mapping)
    return tr[:,:-1],np.array([mapping[int(v)] for v in tr[:,-1]]),te[:,:-1],np.array([mapping[int(v)] for v in te[:,-1]]),'multiclass',domain


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True);spec=cfg['adaptive_data'];environment=check_native_environment(cfg)
    root=Path(__file__).parent/'adaptive_raw';receipt=locked(root/'source.json',spec['source_receipt_sha256']);cases=[];datasets={};audits=[]
    for name,source in receipt.items():
        directory=root/name;assert digest(directory/source['archive'])==source['archive_sha256']
        for file,ref in source['members'].items():assert digest(directory/file)==ref['sha256']
        x,y,tx,ty,task,labels=read_pair(directory,name);expected=spec['expected'][name]
        assert (len(x)+len(tx),x.shape[1],len(labels))==tuple(expected)
        features,target,source_role,source_row,info=preserve_official_test(x,y,tx,ty)
        path=out/(name+'.npz');np.savez_compressed(path,features=features.astype(np.float32),target=target,source_role=source_role,source_row=source_row)
        datasets[name]=dict(path=str(path),sha256=digest(path),shape=list(features.shape),outputs=len(labels),source=source,
            source_url=spec['source_urls'][name],source_label_domain=labels,official_test_rows_preserved=True)
        for seed in spec['seeds'][name]:
            roles,groups=official_roles(features,source_role,target[source_role==0],task,seed)
            cid=f'{name}_{seed}';p=out/cid;p.mkdir(exist_ok=True);split=p/'splits.npz';np.savez(split,**roles)
            case=dict(dataset=name,seed=seed,task=task,outputs=len(labels),split_file=str(split),split_sha256=digest(split),
                sizes={k:len(v) for k,v in roles.items()},protocol_version=5,train_meta=dict(rows=len(roles['train']),protocol='official-test-grouped-train-D'),
                grouping=groups,source_info=info,test_labels_used_for_selection=False)
            if task=='multilabel':case['train_zero_support_labels']=int(np.count_nonzero(target[roles['train']].sum(0)==0))
            cases.append(case)
        audits.append(dict(dataset=name,source_byte_hashes_verified=True,shape_verified=True,**info))
        event('adaptive_fresh_source_verified',dataset=name,task=task,features=features.shape[1],outputs=len(labels),**info)
    manifest=out/'manifest.json';write(manifest,dict(cases=cases,datasets=datasets))
    result=dict(status='adaptive_data_complete',manifest=dict(path=str(manifest),sha256=digest(manifest)),environment=environment,
        audits=audits,cases=[dict(case=f'{c["dataset"]}_{c["seed"]}',sizes=c['sizes']) for c in cases],test_evaluated=False,new_tree_fits=0,
        limitations=['three train/D partitions share each official test, not three independent test sets',
        'new for the recorded project cohort; no universal population-independence claim',
        'Pendigits official writer-independent test retained; Satimage spatial independence not established'])
    write(out/'report.json',result);event('adaptive_data_complete',path=str(out/'report.json'),sha256=digest(out/'report.json'))


if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
