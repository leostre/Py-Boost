"""CPU-only synthetic task isolation and shared-weight ownership preflight."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import numpy as np
from storage_graph import inventory,numerical_spans,union_bytes


def rss():
    fields=Path('/proc/self/status').read_text().splitlines()
    line=next(s for s in fields if s.startswith('VmRSS:'))
    return int(line.split()[1])*1024


def child(cfg,mode,bank,output):
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.base import ModelSpecs,initialize_tabpfn_model
    torch.set_num_threads(2)
    weights=Path(cfg['environment_root'])/'weights'/cfg['weights_revision']/cfg['weights_filename']
    if not weights.is_file() or weights.stat().st_size!=cfg['weights_bytes']:
        raise RuntimeError('licensed checkpoint absent or wrong size')
    # Parent verifies the full SHA once; each child requires that exact file.
    models,configs,_,inference=initialize_tabpfn_model(weights,'classifier',
        devices=[torch.device('cpu')],force_inference_dtype=torch.float32,n_estimators_override=1)
    if len(models)!=1: raise RuntimeError('unexpected multiple checkpoint weights')
    spec=ModelSpecs(models[0],configs[0],inference)
    specs=[spec,spec if bank=='shared' else copy.deepcopy(spec)]
    classifiers=[]
    rng=np.random.default_rng(20261021)
    datasets=[]
    for c,f in ((3,6),(5,9)):
        x=rng.normal(size=(72,f)).astype(np.float32)
        y=np.arange(64,dtype=np.int32)%c
        rng.shuffle(y)
        x[:64,1]+=y
        x[::17,0]=np.nan
        datasets.append((x[:64],y,x[64:]))
    probabilities=[]
    fit_seconds=[]
    for i,((x,y,q),item) in enumerate(zip(datasets,specs)):
        model=TabPFNClassifier(model_path=item,n_estimators=1,auto_scale_n_estimators=False,
            device='cpu',inference_precision=torch.float32,fit_mode=mode,memory_saving_mode=True,
            enable_torch_compile=False,random_state=420+i,n_preprocessing_jobs=1,show_progress_bar=False)
        started=perf_counter()
        model.fit(x,y)
        fit_seconds.append(perf_counter()-started)
        probabilities.append(model.predict_proba(q))
        classifiers.append(model)
    errors=[]
    for model,(_,_,q),expected in zip(classifiers,datasets,probabilities):
        actual=model.predict_proba(q)
        errors.append(float(np.max(np.abs(actual-expected))))
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
    if len({id(m.models_[0]) for m in classifiers})!=(1 if bank=='shared' else 2):
        raise RuntimeError('public model references not shared as declared')
    a=numerical_spans(classifiers[0].models_)
    b=numerical_spans(classifiers[1].models_)
    overlap=union_bytes(a)+union_bytes(b)-union_bytes(a+b)
    if (bank=='shared' and overlap!=union_bytes(a)) or (bank=='independent' and overlap!=0):
        raise RuntimeError('weight storage aliasing contract failed')
    info=dict(mode=mode,bank=bank,status='verified',device='cpu',synthetic_tasks=2,
        classifier_fit_calls=2,nn_weight_training=False,probabilities=[p.tolist() for p in probabilities],
        interference_max_abs_error=max(errors),fit_seconds=fit_seconds,
        reachable_state=inventory(classifiers),weight_union_bytes=union_bytes(a+b),
        weight_sum_bytes=union_bytes(a)+union_bytes(b),shared_weight_bytes=overlap,
        rss_bytes=rss(),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        separate_process_for_each_bank_mode=True,task_datasets_loaded=False,
        interpretation='synthetic correctness/ownership only; not quality or inference speed benchmark')
    output.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k!='probabilities'}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=['low_memory','fit_preprocessors','fit_with_cache'])
    parser.add_argument('--bank',choices=['shared','independent'])
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if args.mode:
        child(cfg,args.mode,args.bank,args.output)
        return
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_storage_graph.py'],check=True,timeout=300)
    weights=Path(cfg['environment_root'])/'weights'/cfg['weights_revision']/cfg['weights_filename']
    h=hashlib.sha256()
    with weights.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    if h.hexdigest()!=cfg['weights_sha256']: raise RuntimeError('licensed weight SHA changed')
    records=[]
    for mode in ('low_memory','fit_preprocessors','fit_with_cache'):
        for bank in ('shared','independent'):
            path=args.output/(mode+'_'+bank+'.json')
            subprocess.run([sys.executable,__file__,str(args.config),str(path),'--mode',mode,'--bank',bank],
                           check=True,timeout=900)
            records.append(json.loads(path.read_text(encoding='utf-8')))
        first,second=records[-2:]
        for a,b in zip(first['probabilities'],second['probabilities']):
            np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
    (args.output/'report.json').write_text(json.dumps(dict(status='complete',records=records,
        scientific_fits=0,synthetic_classifier_fit_calls=12,weight_updates=0,gpu_used=False,
        tabpfn_task_context_not_free=True),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='complete',workers=6,synthetic_fit_calls=12,gpu_used=False)),flush=True)


if __name__=='__main__': main()
