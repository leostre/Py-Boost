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
            note='Restore the documented frozen inputs and an explicitly authorized configuration before running research.')), flush=True)
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
    if cfg['phase'] in ('weights_readiness','sharing_readiness','matched_data','quality_development','quality_final','bank_selection','tfm_bank_serving','budget_quality','budget_quality_analysis','budget_quality_matched_policy','calibration_pilot','calibration_pilot_audit','abl_quality_preflight','abl_quality_study','abl_quality_audit','abl_serving','abl_numerical_diagnostic','abl_width','abl_width_audit','abl_training_pilot','abl_training_development','abl_training_final','abl_protected_pilot','abl_protected_development','abl_protected_final','abl_increment_diagnostic','abl_stop_rule_replay','abl_stop_rule_continuation','abl_budget_revision','unified_recalibration','unified_training','unified_history','unified_history_diagnostic','unified_transfer_data','unified_serving','unified_external_serving','unified_final','unified_product','adaptive_saved','adaptive_training','direct_preflight','direct_training','adaptive_fresh','adaptive_final','adaptive_external','adaptive_matched','saved_banks','window_reserve','reserve_precision','residual_gate','broad_data','broad_training','wide_serving','broad_final','broad_audit','broad_analysis'):
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
                'tfm_bank_serving':'tfm_bank_worker.py','budget_quality':'budget_quality_worker.py',
                'budget_quality_analysis':'audit_stage1.py','budget_quality_matched_policy':'matched_policy_worker.py',
                'calibration_pilot':'calibration_pilot_worker.py',
                'calibration_pilot_audit':'audit_calibration_pilot.py',
                'abl_quality_preflight':'abl_quality_worker.py',
                'abl_quality_study':'abl_quality_worker.py',
                'abl_quality_audit':'audit_abl_quality.py',
                'abl_serving':'abl_serving_study.py',
                'abl_numerical_diagnostic':'abl_stability_study.py',
                'abl_width':'abl_width_study.py',
                'abl_width_audit':'audit_abl_width.py',
                'abl_training_pilot':'abl_training_study.py',
                'abl_training_development':'abl_training_study.py',
                'abl_training_final':'abl_training_final.py',
                'abl_protected_pilot':'abl_protected_study.py',
                'abl_protected_development':'abl_protected_study.py',
                'abl_protected_final':'abl_protected_final.py',
                'abl_increment_diagnostic':'abl_increment_diagnostic.py',
                'abl_stop_rule_replay':'abl_stop_rule_replay.py',
                'abl_stop_rule_continuation':'abl_stop_continuation.py',
                'abl_budget_revision':'abl_budget_revision.py','unified_recalibration':'unified_recalibration.py','unified_training':'unified_training.py','unified_history':'unified_history_training.py','unified_history_diagnostic':'unified_history_diagnostic.py','unified_transfer_data':'unified_transfer_data.py','unified_serving':'unified_serving.py','unified_external_serving':'unified_external_serving.py','unified_final':'unified_final_evaluation.py','unified_product':'unified_product_extensions.py','adaptive_saved':'adaptive_saved.py','adaptive_training':'adaptive_training.py','direct_preflight':'direct_leaf_worker.py','direct_training':'direct_training.py','adaptive_fresh':'adaptive_fresh_training.py','adaptive_final':'adaptive_fresh_evaluate.py','adaptive_external':'adaptive_external.py','adaptive_matched':'adaptive_matched_serving.py','saved_banks':'saved_bank_study.py','window_reserve':'window_reserve_study.py','reserve_precision':'reserve_precision_study.py','residual_gate':'residual_gate_study.py','broad_data':'broad_data.py','broad_training':'broad_training.py','wide_serving':'wide_serving.py','broad_final':'broad_final.py','broad_audit':'broad_development_audit.py','broad_analysis':'two_direction_analysis.py'}[cfg['phase']]
        if cfg['phase']=='abl_quality_preflight':
            run([root/'venv/bin/python','-m','pytest','-q',source/'tests/test_abl_quality_core.py',
                 source/'tests/test_abl_quality_properties.py',source/'tests/test_abl_quality_boundary.py',
                 '--basetemp',output/'pytest-tmp'],env=env,cwd=source,timeout=300)
        if cfg['phase']=='calibration_pilot':
            run([root/'venv/bin/python','-m','pytest','-q',source/'tests/test_budget_quality_core.py',
                 source/'tests/test_calibration_pilot_core.py',source/'tests/test_calibration_pilot_properties.py',
                 '--basetemp',output/'pytest-tmp'],env=env,cwd=source,timeout=180)
        native_phase=cfg['phase'] in ('abl_training_final','abl_protected_pilot','abl_protected_development','abl_protected_final','abl_increment_diagnostic','abl_stop_rule_replay','abl_stop_rule_continuation','abl_budget_revision','unified_recalibration','unified_training','unified_history','unified_history_diagnostic','unified_transfer_data','unified_serving','unified_final','unified_product','adaptive_saved','adaptive_training','direct_preflight','direct_training','adaptive_fresh','adaptive_final','adaptive_matched','saved_banks','window_reserve','reserve_precision','residual_gate','broad_data','broad_training','wide_serving','broad_final','broad_audit','broad_analysis')
        worker_python=Path(json.loads(old.read_text())['executable']) if native_phase else root/'venv/bin/python'
        run([worker_python,source/worker,source/'external_config.json',output,*(['--native'] if native_phase else [])],
            env=env,timeout=21600 if cfg['phase'] in ('quality_development','quality_final','tfm_bank_serving','budget_quality','calibration_pilot','abl_quality_study','abl_serving','abl_width','abl_training_pilot','abl_training_development','abl_stop_rule_continuation','abl_budget_revision','unified_recalibration','unified_training','unified_history','unified_history_diagnostic','unified_transfer_data','unified_serving','unified_external_serving','unified_final','unified_product','adaptive_saved','adaptive_training','direct_preflight','direct_training','adaptive_fresh','adaptive_final','adaptive_external','adaptive_matched','saved_banks','window_reserve','reserve_precision','residual_gate','broad_data','broad_training','wide_serving','broad_final','broad_audit','broad_analysis') else (1200 if cfg['phase']=='weights_readiness' else 6000))
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
