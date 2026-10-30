"""Keep external worker out of native-only imports and preserve exact math."""
import ast,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def test_resource_free_external_import():
    code="import sys,types; sys.modules.setdefault('resource',types.ModuleType('resource')); import unified_external_serving; assert 'experiments.a100_runner' not in sys.modules; assert 'abl_transfer_study' not in sys.modules; assert 'unified_training' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)

def test_exact_copied_boundary():
    def functions(name):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse((ROOT/name).read_text()).body if isinstance(n,ast.FunctionDef)}
    actual=functions('unified_external_core.py')
    for file,names in [('abl_transfer_study.py',['data_before_test']),('unified_recalibration.py',['store_replay']),('unified_training.py',['evaluate','quality_row'])]:
        source=functions(file)
        for name in names:assert actual[name]==source[name]


def test_identical_policy_math():
    def body(file,name):
        return ast.dump(next(n for n in ast.parse((ROOT/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    assert body('unified_calibration_core.py','calibrated_local_variants')==body('abl_training_core.py','calibrated_local_variants')
