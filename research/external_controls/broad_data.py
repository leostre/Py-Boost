"""F0: source-verified multilabel cohorts, feature-grouped roles, no learner."""
from pathlib import Path
import json, sys, hashlib
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from budget_quality_worker import locked, digest, write, event
from adaptive_data_core import numeric_arff, preserve_official_test, official_roles


def label_columns(names, labels, aliases=None):
    resolved = [(aliases or {}).get(n,n) for n in labels]
    if len(set(resolved))!=len(resolved) or not set(resolved)<=set(names):
        raise ValueError('invalid XML label domain')
    return [names.index(n) for n in resolved]


def read_mulan(path, xml, aliases=None):
    names, matrix = numeric_arff(Path(path).read_text(encoding='utf-8-sig'))
    labels = [n.attrib['name'] for n in ET.parse(xml).getroot()]
    outputs = label_columns(names,labels,aliases)
    features = [i for i in range(len(names)) if i not in outputs]
    x, y = matrix[:, features].astype(np.float32), matrix[:, outputs].astype(np.float32)
    if not np.isin(y, [0, 1]).all(): raise ValueError('nonbinary labels')
    return x, y, labels


def blind_test(x, seed):
    _, groups = np.unique(x, axis=0, return_inverse=True)
    proposals = GroupShuffleSplit(n_splits=64, test_size=.2, random_state=seed).split(x, groups=groups)
    return min(proposals, key=lambda pair: abs(len(pair[1])-.2*len(x)))


def run(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    spec = cfg['broad']; sources = {}
    for kind, ids, key in [('fresh', ['emotions_6104','medical_6204'], 'cases'),
                           ('legacy', ['scene_2104','yeast_1304','birds_2204','mediamill_1104','bibtex_2304'], 'case_representations')]:
        ref = cfg['saved_banks'][kind]; report = locked(ref['path'], ref['sha256'])
        for r in report[key]:
            if r['case'] in ids:
                c = locked(r['path'], r['sha256']); sources[c['record']['case']['dataset']] = c['record']
    ref = spec['genbase']; old = locked(ref['path'], ref['sha256'])
    sources['genbase'] = dict(case=next(c for c in old['cases'] if c['dataset']=='genbase'), dataset=old['datasets']['genbase'])
    datasets, cases = {}, []
    for index, name in enumerate(spec['datasets']):
        directory = out/name; directory.mkdir(exist_ok=True)
        if name in sources:
            record = sources[name]; c, d = record['case'], record['dataset']
            assert digest(c['split_file']) == c['split_sha256'] and digest(d['path']) == d['sha256']
            with np.load(d['path'], allow_pickle=False) as f: x, y = f['features'].copy(), f['target'].copy()
            with np.load(c['split_file'], allow_pickle=False) as f: test = f['test'].copy()
            fit = np.setdiff1d(np.arange(len(x)), test)
            old_test_sha = hashlib.sha256(x[test].tobytes()+y[test].tobytes()).hexdigest()
            x, y, role, source_row, info = preserve_official_test(x[fit], y[fit], x[test], y[test])
            assert hashlib.sha256(x[role==1].tobytes()+y[role==1].tobytes()).hexdigest() == old_test_sha
            provenance = dict(kind='historical_test_exploratory', original=record, test_content_sha256=old_test_sha, **info)
            labels = [str(j) for j in range(y.shape[1])]
        else:
            raw = Path(__file__).parent/'broad_raw'/spec['source_dirs'][name]
            receipt_name = 'full_source.json' if name=='enron' else 'source.json'
            receipt = locked(raw/receipt_name, spec['receipts'][name])
            for f in receipt['files']: assert digest(raw/f['name']) == f['sha256']
            xml = next(raw/f['name'] for f in receipt['files'] if f['name'].lower().endswith('.xml'))
            files = {f['name'].lower(): raw/f['name'] for f in receipt['files']}
            train = next((p for n,p in files.items() if n.endswith('-train.arff')), None)
            test = next((p for n,p in files.items() if n.endswith('-test.arff')), None)
            if train is not None and test is not None:
                aliases=spec.get('label_aliases',{}).get(name,{})
                tx, ty, labels = read_mulan(train, xml, aliases); vx, vy, other = read_mulan(test, xml, aliases); assert labels==other
                x, y, role, source_row, info = preserve_official_test(tx,ty,vx,vy)
                provenance = dict(kind='repository_original_train_test', receipt=receipt, label_aliases=aliases, **info)
            else:
                full = next(p for n,p in files.items() if n.endswith('.arff'))
                x, y, labels = read_mulan(full, xml); a,b = blind_test(x, 202610290+index)
                x,y,role,source_row,info = preserve_official_test(x[a],y[a],x[b],y[b])
                provenance = dict(kind='new_blind_grouped_test_not_official', seed=202610290+index, receipt=receipt, **info)
        assert y.ndim==2 and np.isin(y,[0,1]).all() and len(x)==len(y)
        data = directory/'data.npz'; np.savez_compressed(data, features=x, target=y, source_role=role, source_row=source_row)
        datasets[name] = dict(path=str(data),sha256=digest(data),rows=len(x),features=x.shape[1],outputs=y.shape[1],labels=labels,provenance=provenance)
        for repeat in range(3):
            seed = 71004+index*100+repeat*10
            roles, groups = official_roles(x, role, y[role==0], 'multilabel', seed)
            split = directory/f'splits-{seed}.npz'; np.savez(split, **roles)
            tr = y[roles['train']]; supp=tr.sum(0)
            case = dict(dataset=name,seed=seed,task='multilabel',outputs=y.shape[1],split_file=str(split),split_sha256=digest(split),
                        sizes={k:len(v) for k,v in roles.items()},protocol_version=5,train_meta=dict(protocol='grouped80-20-nontest',rows=len(tr)),
                        train_support=supp.astype(int).tolist(),train_zero_support_labels=int((supp==0).sum()),
                        train_meta_profile=dict(label_density=float(tr.mean()),mean_cardinality=float(tr.sum(1).mean()),
                                                support_quantiles=np.quantile(supp,[0,.25,.5,.75,1]).tolist()),
                        grouping=groups,test_labels_used_for_selection=False,split_uses_labels=False)
            cases.append(case)
        event('broad_data_task',dataset=name,shape=list(x.shape),outputs=y.shape[1],source=provenance['kind'])
    manifest=out/'manifest.json';write(manifest,dict(cases=cases,datasets=datasets,protocol=spec['protocol']))
    report=dict(status='broad_data_complete',manifest=dict(path=str(manifest),sha256=digest(manifest)),tasks=len(datasets),cases=len(cases),
                new_trees=0,test_evaluated=False,source_config_sha256=digest(Path(__file__).parent/'external_config.json'))
    write(out/'report.json',report);event('broad_data_complete',**report)


if __name__=='__main__': run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
