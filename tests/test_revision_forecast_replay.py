import hashlib
import json

import pytest

from experiments.revision_forecast_replay import verified_saved_records, replay_forecast_analysis


def _receipt(tmp_path, windows_hash_paths=False):
    directory = tmp_path / 'causal_forecast'
    artifact = directory / 'artifacts' / 'snapshot' / 'saved.bin'
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'unchanged G/H receipt fixture')
    source = 'C:/frozen-run/repo/results/revision/artifacts/snapshot/saved.bin'
    artifact_key = artifact.relative_to(tmp_path).as_posix()
    if windows_hash_paths:
        artifact_key = artifact_key.replace('/', '\\')
    receipt = {'run_id': 'original-run', 'artifact_sha256': {
        artifact_key: hashlib.sha256(artifact.read_bytes()).hexdigest()},
        'results': [{'dataset': 'toy', 'seed': 1, 'model_seed': 1, 'split_seed': 2,
                     'trees_built': 100, 'source_fingerprint': 'original-source',
                     **{key: source for key in ('trajectory_path', 'split_diagnostics_path',
                                               'split_indices_path', 'effective_config_path')}}]}
    path = directory / 'receipt.json'
    path.write_text(json.dumps(receipt), encoding='utf-8')
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), artifact


@pytest.mark.parametrize('windows_hash_paths', [False, True])
def test_verified_replay_resolves_saved_copy_and_keeps_source_identity(tmp_path, windows_hash_paths):
    path, checksum, artifact = _receipt(tmp_path, windows_hash_paths)
    records, provenance = verified_saved_records(path, checksum)
    assert records[0]['trajectory_path'] == str(artifact.resolve())
    assert provenance['source_run_id'] == 'original-run'
    assert provenance['source_fingerprints'] == ['original-source']


def test_changed_receipt_or_saved_data_fails_closed(tmp_path):
    path, checksum, artifact = _receipt(tmp_path)
    with pytest.raises(ValueError, match='receipt checksum'):
        verified_saved_records(path, 'different')
    artifact.write_bytes(b'mutated saved data')
    with pytest.raises(ValueError, match='artifact checksum'):
        verified_saved_records(path, checksum)


def test_analysis_only_replay_never_imports_training_driver(tmp_path, monkeypatch):
    path, checksum, _ = _receipt(tmp_path)
    import experiments.revision_forecast as forecast
    import experiments.revision_comparison as training
    def forbidden(*args, **kwargs):
        raise AssertionError('boosting training must not run during replay')
    monkeypatch.setattr(training, 'run_comparison', forbidden)
    def analysis(records, output):
        output.write_text(json.dumps({'evaluation': [], 'rows': [], 'protocol': {}}), encoding='utf-8')
        return {'forecast_report': str(output), 'summary': []}
    monkeypatch.setattr(forecast, 'analyze_forecasts', analysis)
    monkeypatch.chdir(tmp_path)
    replay_forecast_analysis({'source_receipt': str(path), 'source_receipt_sha256': checksum,
                             'datasets': ['toy'], 'model_seeds': [1], 'split_seeds': [2], 'ntrees': 100})
    result = json.loads(next((tmp_path / 'results/revision').glob('*.json')).read_text())
    assert result['protocol']['new_boosting_trainings'] == 0
