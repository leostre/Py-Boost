"""Fixed orx entry point: vary only committed revision_config.json on child nodes."""

from __future__ import annotations

import json
import os
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The same environment contract applies to every phase and experiment node.
os.environ['CUPY_CACHE_DIR'] = str(Path(tempfile.gettempdir()) / 'cupy_cache_hasboost')
os.environ['HASBOOST_GPU_ACCEPTANCE'] = '1'


def main():
    config = json.loads((ROOT / 'experiments/revision_config.json').read_text(encoding='utf-8'))
    import py_boost
    from experiments.reproducibility import source_fingerprint
    imported = Path(py_boost.__file__).resolve()
    if ROOT not in imported.parents:
        raise RuntimeError('run imported editable worktree instead of the immutable snapshot')
    print(json.dumps({'phase': config['phase'], 'effective_config': config,
                      'source_fingerprint': source_fingerprint(ROOT),
                      'source_root': str(ROOT), 'imported_source': str(imported),
                      'python': sys.executable, 'cupy_cache': os.environ['CUPY_CACHE_DIR']},
                     ensure_ascii=False), flush=True)
    if config['phase'] == 'acceptance':
        import pytest
        code = pytest.main(['tests', '-q'])
        print(json.dumps({'FINAL_SUMMARY': {'phase': 'acceptance', 'pytest_exit_code': int(code)}}), flush=True)
        raise SystemExit(int(code))
    if config['phase'] == 'causal_forecast_analysis':
        from experiments.revision_forecast_replay import replay_forecast_analysis
        replay_forecast_analysis(config)
        return
    if config['phase'] == 'rmt_followup':
        import pytest
        code = pytest.main(['tests/test_rmt_followup.py', 'tests/test_rmt_validation.py', '-q', '-s'])
        if code:
            raise SystemExit(int(code))
        from experiments.rmt_followup import run_followup
        run_followup(config)
        return
    if config['phase'] == 'rmt_validation':
        import pytest
        code = pytest.main(['tests/test_rmt_validation.py', '-q', '-s'])
        if code:
            raise SystemExit(int(code))
        from experiments.revision_rmt_validation import run_validation
        run_validation(config)
        return
    if config['phase'] in ('calibration', 'budget_comparison', 'fixed_memory', 'causal_forecast'):
        if config['phase'] == 'causal_forecast':
            import pytest
            code = pytest.main(['tests/test_gpu_measurements.py', '-q', '-s'])
            if code:
                raise SystemExit(int(code))
        from experiments.revision_comparison import run_comparison
        run_comparison(config)
        return
    raise ValueError('unknown committed experiment phase')


if __name__ == '__main__':
    main()
