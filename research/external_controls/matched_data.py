"""Public binary controls and immutable existing V3 tasks; no models."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split


PUBLIC={
 'wdbc':dict(url='https://archive.ics.uci.edu/static/public/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic.zip',
     member='wdbc.data',rows=569,features=30,seeds=[4104,4114,4124,4134,4144],
     citation='Wolberg, Mangasarian, Street (1993). UCI WDBC. https://doi.org/10.24432/C5DW2B'),
 'banknote':dict(url='https://archive.ics.uci.edu/static/public/267/banknote%2Bauthentication.zip',
     member='data_banknote_authentication.txt',rows=1372,features=4,seeds=[4204,4214,4224,4234,4244],
     citation='Lohweg (2012). UCI Banknote Authentication. https://doi.org/10.24432/C55P57')}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def binary_roles(target,seed):
    y=np.asarray(target)
    if y.ndim!=1 or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ValueError('invalid public binary target')
    def divide(rows,fraction,state):
        a,b=train_test_split(rows,test_size=fraction,random_state=state,stratify=y[rows])
        return np.asarray(a,np.int64),np.asarray(b,np.int64)
    train,outer=divide(np.arange(len(y)),.4,seed)
    validation,test=divide(outer,.5,seed+1)
    calibration,left=divide(validation,.5,seed+100000)
    control,selection=divide(left,.5,seed+200000)
    v2=dict(train=train,calibration=calibration,control=control,selection=selection,test=test)
    kept,moved=divide(train,.25,seed+310000)
    extra_control,extra_selection=divide(moved,2/3,seed+310001)
    roles=dict(train=kept,calibration=calibration,control=np.r_[control,extra_control],
               selection=np.r_[selection,extra_selection],test=test)
    joined=np.concatenate(list(roles.values()))
    np.testing.assert_array_equal(np.sort(joined),np.arange(len(y)))
    for value in roles.values():
        if len(np.unique(y[value]))!=2: raise RuntimeError('missing binary class in role')
    return v2,roles


def parse_public(name,raw):
    table=np.loadtxt(io.StringIO(raw.decode('ascii')),delimiter=',',dtype=str)
    spec=PUBLIC[name]
    if name=='wdbc':
        if table.shape!=(spec['rows'],32) or set(table[:,1])!={'B','M'}:
            raise RuntimeError('WDBC raw schema changed')
        if len(np.unique(table[:,0]))!=len(table): raise RuntimeError('duplicate WDBC ID')
        x=table[:,2:].astype(np.float32)
        mapping={'B':0,'M':1}
        y=np.array([mapping[v] for v in table[:,1]],np.int32)
        ids=table[:,0]
    else:
        if table.shape!=(spec['rows'],5) or set(table[:,-1])!={'0','1'}:
            raise RuntimeError('Banknote raw schema changed')
        x=table[:,:4].astype(np.float32)
        y=table[:,-1].astype(np.int32)
        mapping={'0':0,'1':1}
        ids=np.arange(len(y)).astype(str)
    if x.shape!=(spec['rows'],spec['features']) or not np.isfinite(x).all():
        raise RuntimeError('public feature shape/nonfinite mismatch')
    return x,y,ids,mapping


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    output=Path(sys.argv[2])
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_matched_data.py'],check=True,timeout=300)
    source_path=Path(cfg['absolute_roles_manifest'])
    if digest(source_path)!=cfg['absolute_roles_manifest_sha256']: raise RuntimeError('existing V3 source changed')
    old=json.loads(source_path.read_text(encoding='utf-8'))
    datasets={k:v for k,v in old['datasets'].items() if k in ('helena','mnist','rt_iot2022')}
    cases=[c for c in old['cases'] if c['dataset'] in datasets]
    if len(cases)!=15: raise RuntimeError('expected five fixed splits per existing multiclass task')
    for d in datasets.values():
        if digest(d['path'])!=d['sha256']: raise RuntimeError('fixed multiclass data changed')
    for c in cases:
        if digest(c['split_file'])!=c['split_sha256']: raise RuntimeError('fixed multiclass roles changed')
    for name,spec in PUBLIC.items():
        directory=output/name
        directory.mkdir()
        with urllib.request.urlopen(spec['url'],timeout=120) as response:
            archive=response.read(2*1024**2+1)
        if len(archive)>2*1024**2: raise RuntimeError('unexpected public archive size')
        (directory/'source.zip').write_bytes(archive)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            names=[p for p in z.namelist() if Path(p).name==spec['member']]
            if len(names)!=1: raise RuntimeError('ambiguous public archive member')
            raw=z.read(names[0])
        (directory/spec['member']).write_bytes(raw)
        x,y,ids,mapping=parse_public(name,raw)
        data=directory/'data.npz'
        np.savez_compressed(data,features=x,target=y,source_indices=np.arange(len(x)),source_ids=ids)
        datasets[name]=dict(path=str(data),sha256=digest(data),task='multiclass',features_shape=list(x.shape),
            target_shape=list(y.shape),outputs=2,source=spec['url'],source_zip_sha256=digest(directory/'source.zip'),
            source_member_sha256=digest(directory/spec['member']),target_mapping=mapping,
            feature_policy='original numeric columns; WDBC identifier excluded; no scaling or learned encoding',
            license='CC BY4.0; attribution preserved',citation=spec['citation'])
        (directory/'ATTRIBUTION.txt').write_text(spec['citation']+'\nCC BY 4.0\n'+spec['url']+'\n',encoding='utf-8')
        for seed in spec['seeds']:
            v2,roles=binary_roles(y,seed)
            before=directory/f'v2-{seed}.npz'
            after=directory/f'v3-{seed}.npz'
            np.savez(before,**v2)
            np.savez(after,**roles)
            cases.append(dict(dataset=name,seed=seed,task='multiclass',outputs=2,protocol_version=3,
                split_file=str(after),split_sha256=digest(after),source_split_file=str(before),
                source_split_sha256=digest(before),absolute_budget_bytes=131072,
                sizes={k:len(v) for k,v in roles.items()},
                role_policy='binary stratified V2 then train-only redistribution to approximately45/10/10/15/20',
                test_metrics_evaluated=False))
    result=dict(status='complete',datasets=datasets,cases=cases,new_fits=0,test_metrics_evaluated=False,
        primary_case_ids=['helena_1204','mnist_3104','rt_iot2022_3204','wdbc_4104','banknote_4204'],
        source_roles_manifest=str(source_path),source_roles_sha256=digest(source_path),
        interpretation='five distinct tasks; existing public subsets and new binary controls, not100tasks')
    path=output/'matched_tasks_manifest.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='complete',tasks=len(datasets),cases=len(cases),new_fits=0,
                         manifest_sha256=digest(path))),flush=True)


if __name__=='__main__': main()
