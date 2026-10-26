"""Isolated bootstrap supervisor; never installs into the HASBoost environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, **kwargs):
    print(json.dumps(dict(command=[str(a) for a in args])), flush=True)
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    source=Path(__file__).resolve().parent
    cfg=json.loads((source/'external_config.json').read_text(encoding='utf-8'))
    if cfg.get('phase')=='publication':
        print(json.dumps(dict(status='publication_only', experiments_started=False,
            note='Restore an explicitly authorized local configuration before a research run.')), flush=True)
        return
    old=Path(cfg['frozen_hasboost_environment'])
    if digest(old)!=cfg['frozen_hasboost_sha256']:
        raise RuntimeError('HASBoost environment changed before separate setup')
    root=Path(cfg['environment_root']).resolve()
    if root!=Path('/root/hasboost_external_20261021'):
        raise RuntimeError('unexpected isolated environment location')
    root.mkdir(exist_ok=True)
    output=source/cfg['output_dir']
    output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,UV_PYTHON_INSTALL_DIR=str(root/'python'),
             HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',DO_NOT_TRACK='1',
             HF_HOME=str(root/'hf-cache'),MPLCONFIGDIR=str(root/'mpl'),
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    lock=root/'environment.json'
    if cfg['phase'] in ('weights_readiness','sharing_readiness','matched_data','quality_development','quality_final','bank_selection','tfm_bank_serving'):
        if not cfg['tabpfn_weights_authorized'] or digest(lock)!=cfg['external_environment_sha256']:
            raise RuntimeError('licensed weights or frozen runtime gate missing')
        frozen=json.loads(lock.read_text(encoding='utf-8'))
        installed=run([root/'bootstrap/bin/uv','pip','freeze','--python',root/'venv/bin/python'],
                      env=env,text=True,stdout=subprocess.PIPE,timeout=120).stdout
        if installed!=frozen['freeze']: raise RuntimeError('separate package versions changed')
        env.update(HF_HUB_OFFLINE='0' if cfg['phase']=='weights_readiness' else '1',
                   HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TABPFN_NO_BROWSER='1')
        worker={'weights_readiness':'weights_worker.py','sharing_readiness':'sharing_worker.py',
                'matched_data':'matched_data.py','quality_development':'quality_worker.py',
                'quality_final':'quality_audit.py','bank_selection':'bank_selection.py',
                'tfm_bank_serving':'tfm_bank_worker.py'}[cfg['phase']]
        run([root/'venv/bin/python',source/worker,source/'external_config.json',output],
            env=env,timeout=21600 if cfg['phase'] in ('quality_development','quality_final','tfm_bank_serving') else (1200 if cfg['phase']=='weights_readiness' else 6000))
        if digest(old)!=cfg['frozen_hasboost_sha256'] or digest(lock)!=cfg['external_environment_sha256']:
            raise RuntimeError('environment descriptor changed during weight preparation')
        return
    if cfg['phase']!='environment_readiness' or cfg['tabpfn_weights_authorized']:
        raise RuntimeError('bootstrap permits readiness only; no weights')
    if lock.exists():
        raise RuntimeError('isolated environment already sealed; bootstrap cannot alter it')
    bootstrap=root/'bootstrap'
    if not bootstrap.exists(): venv.EnvBuilder(with_pip=True).create(bootstrap)
    run([bootstrap/'bin/python','-m','pip','install','--disable-pip-version-check',
         '--only-binary=:all:','uv=='+cfg['uv_version']],env=env,timeout=600)
    uv=bootstrap/'bin/uv'
    run([uv,'python','install',cfg['python_version']],env=env,timeout=900)
    python=run([uv,'python','find',cfg['python_version']],env=env,text=True,
               stdout=subprocess.PIPE,timeout=60).stdout.strip()
    if not Path(python).resolve().is_relative_to(root):
        raise RuntimeError('managed Python escaped isolated root')
    runtime=root/'venv'
    if not runtime.exists(): run([uv,'venv','--python',python,runtime],env=env,timeout=120)
    exe=runtime/'bin/python'
    run([uv,'pip','install','--python',exe,'--only-binary=:all:',
         '--index-url','https://download.pytorch.org/whl/cu126',
         'torch==2.13.0+cu126'],env=env,timeout=1800)
    run([uv,'pip','install','--python',exe,'--only-binary=:all:',
         '--index-url','https://pypi.org/simple','-r',source/'requirements.txt'],env=env,timeout=1800)
    run([uv,'pip','check','--python',exe],env=env,timeout=120)
    freeze=run([uv,'pip','freeze','--python',exe],env=env,text=True,stdout=subprocess.PIPE,timeout=120).stdout
    (output/'requirements.lock.txt').write_text(freeze,encoding='utf-8')
    run([exe,source/'readiness_worker.py',output/'readiness.json'],env=env,timeout=300)
    if digest(old)!=cfg['frozen_hasboost_sha256']:
        raise RuntimeError('HASBoost environment changed during separate setup')
    info=dict(python=python,runtime=str(exe),requirements_sha256=digest(source/'requirements.txt'),
              lock_sha256=digest(output/'requirements.lock.txt'),freeze=freeze,
              readiness_sha256=digest(output/'readiness.json'),hasboost_sha256=digest(old),
              tabpfn_weights_downloaded=False,source_config_sha256=digest(source/'external_config.json'))
    lock.write_text(json.dumps(info,indent=2),encoding='utf-8')
    (output/'environment.json').write_bytes(lock.read_bytes())
    print(json.dumps(dict(status='ready',environment_sha256=digest(lock),weights_downloaded=False)),flush=True)


if __name__=='__main__': main()
