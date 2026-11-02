"""Finite stage dispatch; input configs retain the frozen outer run contract."""
from pathlib import Path
import json,subprocess,sys

RUNNERS={
    'R1':'six_r1.py','R1A':'six_r1a.py','R2':'six_r2.py','R2C':'six_r2c.py',
    'R3':'six_r3.py','R3C':'six_r3c.py','R4':'six_r4.py',
    'R5':'six_r5.py','R5A':'six_r5a.py','R5Q':'six_r5q.py','R5QM':'six_r5qm.py','R5S':'six_r5s.py','R5O':'six_r5o.py',
    'R6D':'six_r6d.py','R6E':'six_r6e.py','R6B':'six_r6b.py','R6T':'six_r6t.py','R7':'six_analysis.py'}

def runner_for(stage):
    if not isinstance(stage,str) or stage not in RUNNERS:raise ValueError('unknown six-direction research stage')
    return RUNNERS[stage]

if __name__=='__main__':
    config=Path(sys.argv[1]);cfg=json.loads(config.read_text(encoding='utf8'));stage=cfg['six']['stage']
    subprocess.run([sys.executable,str(Path(__file__).parent/runner_for(stage)),str(config),*sys.argv[2:]],check=True)
