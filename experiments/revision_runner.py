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
RUNTIME_TEMP = Path(tempfile.gettempdir()) / 'hasboost_runtime'
RUNTIME_TEMP.mkdir(parents=True, exist_ok=True)
os.environ['TEMP'] = str(RUNTIME_TEMP)
os.environ['TMP'] = str(RUNTIME_TEMP)
os.environ['HASBOOST_GPU_ACCEPTANCE'] = '1'


def main():
    config = json.loads((ROOT / 'experiments/revision_config.json').read_text(encoding='utf-8'))
    runtime_site_packages = config.get('runtime_site_packages')
    if runtime_site_packages:
        runtime_site_packages = str(runtime_site_packages)
        if runtime_site_packages not in sys.path:
            sys.path.insert(0, runtime_site_packages)
    import py_boost
    from experiments.reproducibility import source_fingerprint
    imported = Path(py_boost.__file__).resolve()
    if ROOT not in imported.parents:
        raise RuntimeError('run imported editable worktree instead of the immutable snapshot')
    print(json.dumps({'phase': config['phase'], 'effective_config': config,
                      'source_fingerprint': source_fingerprint(ROOT),
                      'source_root': str(ROOT), 'imported_source': str(imported),
                      'python': sys.executable,
                      'runtime_site_packages': runtime_site_packages,
                      'cupy_cache': os.environ['CUPY_CACHE_DIR'],
                      'runtime_temp': str(RUNTIME_TEMP)},
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
    if config['phase'] == 'decision_aware_feasibility':
        import pytest
        code = pytest.main(['tests/test_decision_aware_feasibility.py', '-q', '-s'])
        if code:
            raise SystemExit(int(code))
        from experiments.decision_aware_feasibility import run_feasibility_audit
        run_feasibility_audit(config)
        return
    if config['phase'] == 'decision_aware_collection':
        import pytest
        code = pytest.main([
            'tests/test_decision_aware_feasibility.py',
            'tests/test_split_diagnostic_artifact.py',
            'tests/test_gpu_measurements.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.decision_aware_collection import run_collection
        run_collection(config)
        return
    if config['phase'] == 'decision_aware_collection_replay':
        import pytest
        code = pytest.main([
            'tests/test_decision_aware_feasibility.py',
            'tests/test_split_diagnostic_artifact.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.decision_aware_collection_replay import replay_collection
        replay_collection(config)
        return
    if config['phase'] == 'decision_aware_refinement':
        import pytest
        code = pytest.main([
            'tests/test_decision_aware_refinement.py',
            'tests/test_split_diagnostics.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.decision_aware_refinement import run_refinement
        run_refinement(config)
        return
    if config['phase'] == 'decision_aware_refinement_shadow':
        import pytest
        code = pytest.main([
            'tests/test_exact_leaf_counts.py',
            'tests/test_decision_aware_refinement.py',
            'tests/test_decision_aware_refinement_shadow.py',
            'tests/test_top_k_extraction.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_integration_benchmark.py',
            'tests/test_refinement_numeric_diagnostic.py',
            'tests/test_gpu_measurements.py::test_exact_shortlist_refinement_matches_full_gain_and_stays_shadow_only',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.decision_aware_refinement_shadow import run_refinement_shadow
        run_refinement_shadow(config)
        return
    if config['phase'] == 'decision_aware_refinement_mismatch_diagnostic':
        import pytest
        code = pytest.main([
            'tests/test_exact_leaf_counts.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_mismatch_diagnostic.py',
            'tests/test_gpu_measurements.py::test_exact_shortlist_refinement_matches_full_gain_and_stays_shadow_only',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.refinement_mismatch_diagnostic import (
            run_refinement_mismatch_diagnostic,
        )
        run_refinement_mismatch_diagnostic(config)
        return
    if config['phase'] == 'decision_aware_refinement_diagnostics':
        import pytest
        code = pytest.main([
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_gpu_measurements.py::test_exact_shortlist_refinement_matches_full_gain_and_stays_shadow_only',
            'tests/test_decision_aware_refinement_shadow.py',
            '-q',
            '-s',
        ])
        summary = {
            'phase': 'decision_aware_refinement_diagnostics',
            'pytest_exit_code': int(code),
            'new_boosting_fits': 0,
            'media_mill_fits': 0,
            'inactive_rows_filtered_before_histogram': True,
        }
        print(json.dumps({'FINAL_SUMMARY': summary}, ensure_ascii=False), flush=True)
        raise SystemExit(int(code))
    if config['phase'] == 'decision_aware_top_k_benchmark':
        import pytest
        code = pytest.main(['tests/test_top_k_extraction.py', '-q', '-s'])
        if code:
            raise SystemExit(int(code))
        from experiments.top_k_benchmark import run_top_k_benchmark
        run_top_k_benchmark(config)
        return
    if config['phase'] == 'decision_aware_fused_top_k_benchmark':
        import pytest
        code = pytest.main(['tests/test_top_k_extraction.py', '-q', '-s'])
        if code:
            raise SystemExit(int(code))
        from experiments.top_k_benchmark import run_top_k_benchmark
        run_top_k_benchmark(config)
        return
    if config['phase'] == 'decision_aware_fused_merge_top_k_benchmark':
        import pytest
        code = pytest.main(['tests/test_top_k_extraction.py', '-q', '-s'])
        if code:
            raise SystemExit(int(code))
        from experiments.top_k_benchmark import run_top_k_benchmark
        run_top_k_benchmark(config)
        return
    if config['phase'] == 'decision_aware_refinement_integration_benchmark':
        import pytest
        code = pytest.main([
            'tests/test_exact_leaf_counts.py',
            'tests/test_top_k_extraction.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_integration_benchmark.py',
            'tests/test_gpu_measurements.py::test_exact_shortlist_refinement_matches_full_gain_and_stays_shadow_only',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.refinement_integration_benchmark import (
            run_refinement_integration_benchmark,
        )
        run_refinement_integration_benchmark(config)
        return
    if config['phase'] == 'refinement_numeric_diagnostic':
        import pytest
        code = pytest.main([
            'tests/test_exact_leaf_counts.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_numeric_diagnostic.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.refinement_numeric_diagnostic import (
            run_refinement_numeric_diagnostic,
        )
        run_refinement_numeric_diagnostic(config)
        return
    if config['phase'] == 'decision_aware_refinement_apply':
        import pytest
        code = pytest.main([
            'tests/test_exact_leaf_counts.py',
            'tests/test_top_k_extraction.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_application.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.decision_aware_refinement_apply import run_refinement_apply
        run_refinement_apply(config)
        return
    if config['phase'] == 'refinement_determinism_forensics':
        import pytest
        code = pytest.main([
            'tests/test_refinement_determinism_forensics.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.refinement_determinism_forensics import (
            run_refinement_determinism_forensics,
        )
        run_refinement_determinism_forensics(config)
        return
    if config['phase'] == 'decision_aware_refinement_determinism':
        import pytest
        code = pytest.main([
            'tests/test_refinement_determinism_frontier.py',
            'tests/test_refinement_application.py',
            'tests/test_top_k_extraction.py',
            'tests/test_candidate_gpu_statistics.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.decision_aware_refinement_determinism import (
            run_refinement_determinism,
        )
        run_refinement_determinism(config)
        return
    if config['phase'] == 'deterministic_leaf_reduction':
        import pytest
        code = pytest.main([
            'tests/test_deterministic_leaf_values.py',
            'tests/test_deterministic_leaf_reduction_core.py',
            'tests/test_refinement_determinism_frontier.py',
            'tests/test_refinement_application.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.deterministic_leaf_reduction import (
            run_deterministic_leaf_reduction,
        )
        run_deterministic_leaf_reduction(config)
        return
    if config['phase'] in {
        'deterministic_histogram',
        'deterministic_histogram_performance',
    }:
        import pytest
        code = pytest.main([
            'tests/test_deterministic_histogram.py',
            'tests/test_deterministic_histogram_core.py',
            'tests/test_deterministic_leaf_values.py',
            'tests/test_exact_leaf_counts.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_top_k_extraction.py',
            'tests/test_refinement_determinism_frontier.py',
            'tests/test_refinement_application.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.deterministic_histogram import (
            run_deterministic_histogram,
        )
        run_deterministic_histogram(config)
        return
    if config['phase'] == 'deterministic_histogram_profile':
        import pytest
        code = pytest.main([
            'tests/test_deterministic_histogram_profile_core.py',
            'tests/test_deterministic_histogram.py',
            'tests/test_deterministic_histogram_core.py',
            'tests/test_deterministic_leaf_values.py',
            'tests/test_exact_leaf_counts.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_determinism_frontier.py',
            'tests/test_refinement_application.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.deterministic_histogram_profile import (
            run_deterministic_histogram_profile,
        )
        run_deterministic_histogram_profile(config)
        return
    if config['phase'] == 'deterministic_histogram_performance_2':
        import pytest
        code = pytest.main([
            'tests/test_deterministic_histogram.py',
            'tests/test_deterministic_histogram_core.py',
            'tests/test_deterministic_leaf_values.py',
            'tests/test_exact_leaf_counts.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_determinism_frontier.py',
            'tests/test_refinement_application.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.deterministic_histogram_perf2 import (
            run_deterministic_histogram_perf2,
        )
        run_deterministic_histogram_perf2(config)
        return
    if config['phase'] == 'deterministic_histogram_performance_3':
        import pytest
        code = pytest.main([
            'tests/test_deterministic_histogram.py',
            'tests/test_deterministic_histogram_core.py',
            'tests/test_deterministic_leaf_values.py',
            'tests/test_exact_leaf_counts.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_determinism_frontier.py',
            'tests/test_refinement_application.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.deterministic_histogram_perf3 import (
            run_deterministic_histogram_perf3,
        )
        run_deterministic_histogram_perf3(config)
        return
    if config['phase'] == 'deterministic_histogram_cross_dataset':
        import pytest
        code = pytest.main([
            'tests/test_deterministic_histogram.py',
            'tests/test_deterministic_histogram_core.py',
            'tests/test_deterministic_leaf_values.py',
            'tests/test_exact_leaf_counts.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_determinism_frontier.py',
            'tests/test_refinement_application.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.deterministic_histogram_cross_dataset import (
            run_deterministic_histogram_cross_dataset,
        )
        run_deterministic_histogram_cross_dataset(config)
        return
    if config['phase'] == 'deterministic_histogram_article_suite':
        import pytest
        code = pytest.main([
            'tests/test_benchmark_datasets.py',
            'tests/test_deterministic_histogram_article_suite_core.py',
            'tests/test_deterministic_histogram.py',
            'tests/test_deterministic_histogram_core.py',
            'tests/test_deterministic_leaf_values.py',
            'tests/test_exact_leaf_counts.py',
            'tests/test_candidate_gpu_statistics.py',
            'tests/test_refinement_determinism_frontier.py',
            'tests/test_refinement_application.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.deterministic_histogram_article_suite import (
            run_deterministic_histogram_article_suite,
        )
        run_deterministic_histogram_article_suite(config)
        return
    if config['phase'] == 'deterministic_histogram_timing_repeats':
        import pytest
        code = pytest.main([
            'tests/test_deterministic_histogram_timing_repeats_core.py',
            'tests/test_deterministic_histogram_article_suite_core.py',
            'tests/test_deterministic_histogram.py',
            'tests/test_deterministic_histogram_core.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.deterministic_histogram_timing_repeats import (
            run_deterministic_histogram_timing_repeats,
        )
        run_deterministic_histogram_timing_repeats(config)
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
    if config['phase'] == 'rmt_subspace_reproducibility':
        import pytest
        code = pytest.main([
            'tests/test_rmt_subspace_reproducibility_core.py',
            'tests/test_revision_forecast_replay.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.rmt_subspace_reproducibility import run_reproducibility
        run_reproducibility(config)
        return
    if config['phase'] == 'rmt_scalar_reproducibility':
        import pytest
        code = pytest.main([
            'tests/test_rmt_scalar_reproducibility_core.py',
            'tests/test_rmt_subspace_reproducibility_core.py',
            'tests/test_revision_forecast_replay.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.rmt_scalar_reproducibility import run_scalar_reproducibility
        run_scalar_reproducibility(config)
        return
    if config['phase'] == 'rmt_residual_signal':
        import pytest
        code = pytest.main([
            'tests/test_rmt_residual_signal.py',
            'tests/test_rmt_residual_signal_core.py',
            'tests/test_rmt_scalar_reproducibility_core.py',
            'tests/test_rmt_subspace_reproducibility_core.py',
            'tests/test_revision_forecast_replay.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.rmt_residual_signal import run_residual_signal
        run_residual_signal(config)
        return
    if config['phase'] == 'rmt_residual_signal_expanded':
        import pytest
        code = pytest.main([
            'tests/test_benchmark_datasets.py',
            'tests/test_rmt_residual_expanded_core.py',
            'tests/test_rmt_residual_signal_core.py',
            'tests/test_rmt_scalar_reproducibility_core.py',
            'tests/test_rmt_subspace_reproducibility_core.py',
            'tests/test_revision_forecast_replay.py',
            '-q',
            '-s',
        ])
        if code:
            raise SystemExit(int(code))
        from experiments.rmt_residual_expanded import run_expanded_residual_signal
        run_expanded_residual_signal(config)
        return
    if config['phase'] == 'h5_2_event_scheduler':
        import pytest
        code = pytest.main(['tests/test_h5_2_event_scheduler.py', '-q', '-s'])
        if code:
            raise SystemExit(int(code))
        from experiments.h5_2_event_scheduler import run_h5_2_event_scheduler
        run_h5_2_event_scheduler(config)
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
