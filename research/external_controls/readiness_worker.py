"""GPU runtime and tiny synthetic controls, with HF explicitly offline."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    if os.environ.get('HF_HUB_OFFLINE')!='1': raise RuntimeError('weights gate missing')
    import torch
    import tabpfn  # Import only: no estimator or ModelSpecs is instantiated.
    from catboost import CatBoostClassifier
    import xgboost as xgb
    rng=np.random.default_rng(20261021)
    x=rng.normal(size=(256,8)).astype(np.float32)
    y=(x[:,0]+x[:,1]>.1).astype(np.int32)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in separate runtime')
    free,total=torch.cuda.mem_get_info()
    if free<4*1024**3: raise RuntimeError('insufficient shared A100 free memory for readiness')
    torch.cuda.set_per_process_memory_fraction(.1)
    t=torch.tensor(x,device='cuda')
    np.testing.assert_allclose((t.T@t).cpu().numpy(),x.T@x,atol=2e-4,rtol=2e-5)
    cat=CatBoostClassifier(iterations=4,depth=3,task_type='GPU',devices='0',gpu_ram_part=.1,
                          thread_count=2,random_seed=20261021,verbose=False,allow_writing_files=False)
    cat.fit(x,y)
    xp=xgb.XGBClassifier(n_estimators=4,max_depth=3,device='cuda',tree_method='hist',n_jobs=2,
                         random_state=20261021)
    xp.fit(x,y)
    predictions=[cat.predict_proba(x),xp.predict_proba(x)]
    if not all(np.isfinite(p).all() and p.shape==(256,2) for p in predictions):
        raise RuntimeError('external GPU synthetic control invalid')
    info=dict(python=sys.version,versions={p:metadata.version(p) for p in
        ('torch','tabpfn','catboost','xgboost-cu12','numpy','scipy','scikit-learn')},
        torch_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),free_bytes=free,total_bytes=total,
        torch_peak_allocated=torch.cuda.max_memory_allocated(),synthetic_fits=2,synthetic_trees=8,
        gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,memory.free',
                                           '--format=csv,noheader'],text=True).strip(),
        tabpfn_imported=True,tabpfn_instantiated=False,weights_downloaded=False,
        hf_cache_files=[str(p) for p in Path(os.environ['HF_HOME']).rglob('*') if p.is_file()])
    if info['hf_cache_files']: raise RuntimeError('unexpected Hugging Face cache materialization')
    Path(sys.argv[1]).write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info),flush=True)


if __name__=='__main__': main()
