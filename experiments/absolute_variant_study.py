"""Subset wrapper reuses sealed controls without training full models again."""
import json
import subprocess
import sys
from pathlib import Path
from experiments.a100_runner import sha256,write_report
from experiments.absolute_history_study import absolute_history_training


def absolute_variant_training(config,report,output):
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_horizon_reserve.py'],check=True,timeout=300)
    rp=Path(config['variant_source_roles_manifest'])
    if sha256(rp)!=config['variant_source_roles_manifest_sha256']:
        raise RuntimeError('variant source roles changed')
    roles=json.loads(rp.read_text(encoding='utf-8'))
    wanted=set(config['variant_cases'])
    cases=[c for c in roles['cases'] if f'{c["dataset"]}_{c["seed"]}' in wanted]
    if len(cases)!=len(wanted): raise RuntimeError('variant cohort incomplete or duplicate')
    path=output/'variant_roles_manifest.json'
    write_report(path,dict(roles,cases=cases,planned_cases=len(cases)))
    cfg=dict(config,cumulative_roles_manifest=str(path),cumulative_roles_manifest_sha256=sha256(path))
    absolute_history_training(cfg,report,output)
    report.update(roles_manifest=str(path),roles_manifest_sha256=sha256(path))
