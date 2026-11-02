"""Isolated fixed nvForest environment. Never changes either HASBoost runtime."""
from pathlib import Path
import hashlib,json,subprocess,sys,os

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(args,**kwargs):
    print(json.dumps(dict(command=list(map(str,args)))),flush=True)
    return subprocess.run(list(map(str,args)),check=True,**kwargs)

root=Path(__file__).resolve().parent;cfg=json.loads((root/'config.json').read_text());base=Path(cfg['environment_root'])
if cfg.get('phase')=='publication':
    print(json.dumps(dict(status='publication_only',experiments_started=False)),flush=True)
    sys.exit(0)
for p,digest in cfg['unchanged_environments'].items():assert sha(p)==digest,p
base.mkdir(exist_ok=True);python=base/'venv/bin/python';lock=base/'environment.json';uv=Path('/root/hasboost_external_20261021/bootstrap/bin/uv')
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1')
if cfg['phase']=='readiness':
    if lock.exists():raise RuntimeError('Readiness is frozen; use its child and do not reinstall')
    if not python.exists():run([uv,'venv',base/'venv','--python','/root/hasboost_external_20261021/venv/bin/python'])
    run([uv,'pip','install','--python',python,*cfg['requirements']],env=env,timeout=1800)
    frozen=run([uv,'pip','freeze','--python',python],text=True,stdout=subprocess.PIPE).stdout
    probe=run([python,'-c',"import nvforest,cupy,treelite,json,sys; print(json.dumps(dict(python=sys.version,device=cupy.cuda.runtime.getDeviceCount(),nvforest=nvforest.__version__,cupy=cupy.__version__,treelite=treelite.__version__)))"],env=env,text=True,stdout=subprocess.PIPE).stdout
    record=dict(requirements=cfg['requirements'],freeze=frozen,probe=probe,unchanged_environments=cfg['unchanged_environments']);lock.write_text(json.dumps(record,indent=2)+'\n')
    result=dict(status='nvforest_environment_ready',environment_sha256=sha(lock),record=record,new_trees=0)
else:
    assert sha(lock)==cfg['environment_sha256']
    frozen=run([uv,'pip','freeze','--python',python],text=True,stdout=subprocess.PIPE).stdout
    assert frozen==json.loads(lock.read_text())['freeze']
    run([python,root/'nvforest_measure.py',root/'config.json'],env=env,timeout=21600)
    result=dict(status='nvforest_phase_complete',environment_sha256=sha(lock),phase=cfg['phase'])
for p,digest in cfg['unchanged_environments'].items():assert sha(p)==digest,p
(root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
