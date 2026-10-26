import json
from pathlib import Path

import pytest

from experiments.model_bank_core import traffic_schedule
from tfm_serving_core import atomic_write, gpu_owner, interrupted_result, request_plan, worker_plan


@pytest.mark.parametrize('kind', ['uniform', 'skew'])
@pytest.mark.parametrize('count', [5, 6, 16, 32, 127, 128])
def test_short_request_plan_preserves_original_request_identity(kind, count):
    # Independent frozen protocol is the OLD full trace, not a fresh short RNG.
    old = traffic_schedule(5, 128, kind, 20261021)
    actual = request_plan(5, count, kind)
    assert actual == old[:count]
    assert actual[:5] == [0, 1, 2, 3, 4]
    assert len(actual) == count


def test_uniform_fresh_short_trace_would_not_be_a_fair_comparison():
    assert request_plan(5, 32, 'uniform') != traffic_schedule(5, 32, 'uniform', 20261021)


@pytest.mark.parametrize('count', [0, 4, 129, True, 32.5])
def test_short_request_count_cannot_hide_tasks_or_expand_old_protocol(count):
    with pytest.raises(ValueError): request_plan(5, count, 'uniform')


def test_cached_strong_model_is_first_without_dropping_other_modes():
    assert worker_plan() == [('fit_with_cache', 'tabpfn_auto'), ('fit_with_cache', 'tabpfn_1'),
                             ('fit_preprocessors', 'tabpfn_auto'), ('fit_preprocessors', 'tabpfn_1')]


@pytest.mark.parametrize('after, identities, expected', [
    ({100: 123, 200: 456}, {200}, dict(pid=200, verified=True, method='namespace_pid_match')),
    ({100: 123}, {200}, dict(pid=None, verified=False, method='ambiguous_or_no_observed_pid')),
    ({100: 123, 300: 456}, {200}, dict(pid=300, verified=False,
        method='unique_new_pid_after_actual_inference_not_namespace_verified')),
    ({100: 123, 300: 456, 400: 789}, {200}, dict(pid=None, verified=False, method='ambiguous_or_no_observed_pid')),
])
def test_gpu_ownership_does_not_silently_certify_a_foreign_process(after, identities, expected):
    assert gpu_owner({100: 123}, after, identities) == expected


def test_late_cuda_initialization_can_resolve_pid_only_after_actual_inference():
    before = {100: 123}
    assert gpu_owner(before, before, {200})['pid'] is None
    assert gpu_owner(before, {100: 123, 200: 456}, {200})['verified']


def test_checkpoint_roundtrip_and_timeout_preserve_every_completed_request(tmp_path):
    path = tmp_path/'worker.json'
    completed = [dict(batch=1, requests=32)]
    pending = dict(batch=32, request_samples_ms=[])
    for i in range(4):
        pending['request_samples_ms'].append(dict(task_index=i, elapsed_ms=float(i)))
        atomic_write(path, dict(status='running', metrics=completed, pending_trace=pending,
                               fits=[dict(actual_n_estimators=8)]))
        saved = json.loads(path.read_text())
        assert len(saved['pending_trace']['request_samples_ms']) == i+1
        assert not path.with_suffix('.partial.json').exists()
    result = interrupted_result(path, 'fit_with_cache', 'tabpfn_auto', 'timeout', timeout_seconds=3600)
    assert result['status'] == 'timeout'
    assert result['metrics'] == completed
    assert len(result['pending_trace']['request_samples_ms']) == 4
    assert result['fits'][0]['actual_n_estimators'] == 8


def test_invalid_snapshot_cannot_destroy_the_last_valid_checkpoint(tmp_path):
    path = tmp_path/'worker.json'
    atomic_write(path, dict(status='running', requests=3))
    with pytest.raises(ValueError): atomic_write(path, dict(timing=float('nan')))
    assert json.loads(path.read_text()) == dict(status='running', requests=3)


def test_timeout_before_first_checkpoint_is_explicit_not_fabricated(tmp_path):
    result = interrupted_result(tmp_path/'absent.json', 'fit_with_cache', 'tabpfn_auto', 'timeout')
    assert result == dict(status='timeout', mode='fit_with_cache', family='tabpfn_auto')


def test_approved_configuration_preserves_scientific_and_environment_contracts():
    cfg = json.loads((Path(__file__).resolve().parents[1]/'external_config.json').read_text())
    assert cfg['bank_requests'] == 32 and cfg['original_bank_requests'] == 128
    assert cfg['worker_timeout_seconds'] == 3600
    assert cfg['torch_memory_fraction'] == .4
    assert cfg['quality_cases'] == ['helena_1204', 'mnist_3104', 'rt_iot2022_3204', 'wdbc_4104', 'banknote_4204']
    assert cfg['external_environment_sha256'] == '34588579bac49b4f49adf00f22459b84040c799c9bcd2d21b78c4821c139cbb7'
    assert cfg['frozen_hasboost_sha256'] == '634d173ce93429e267a57122d1594073b48b9ef25cd410d10798b144c32a6f42'
    assert cfg['weights_sha256'] == 'ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3'
