"""Replay only the analysis of immutable, checksum-verified saved trajectories."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_saved_records(receipt_path, expected_sha256):
    """Resolve copies by their receipt hash, never silently fall back to live code/data."""
    receipt_path = Path(receipt_path).resolve()
    if file_sha256(receipt_path) != expected_sha256:
        raise ValueError('source receipt checksum mismatch')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    artifact_root = receipt_path.parent.parent
    hashes = {relative.replace('\\', '/'): expected
              for relative, expected in receipt['artifact_sha256'].items()}
    for relative, expected in hashes.items():
        path = (artifact_root / relative).resolve()
        if not path.is_relative_to(artifact_root):
            raise ValueError('saved artifact escapes receipt root')
        if file_sha256(path) != expected:
            raise ValueError(f'saved artifact checksum mismatch: {relative}')
    records = []
    for original in receipt['results']:
        record = dict(original)
        for key in ('trajectory_path', 'split_diagnostics_path', 'split_indices_path', 'effective_config_path'):
            suffix = str(record[key]).replace('\\', '/').split('/artifacts/', 1)[1]
            local = (receipt_path.parent / 'artifacts' / suffix).resolve()
            if not local.is_relative_to(artifact_root):
                raise ValueError('source record escapes receipt root')
            relative = local.relative_to(artifact_root).as_posix()
            if relative not in hashes:
                raise ValueError('source record is missing an artifact checksum')
            record[key] = str(local)
        records.append(record)
    provenance = {'source_run_id': receipt['run_id'], 'receipt_sha256': expected_sha256,
                  'verified_artifacts': len(hashes), 'source_fingerprints': sorted({r['source_fingerprint'] for r in records}),
                  'source_receipt': str(receipt_path)}
    return records, provenance


def replay_forecast_analysis(config):
    from experiments.revision_forecast import analyze_forecasts
    records, provenance = verified_saved_records(config['source_receipt'], config['source_receipt_sha256'])
    expected = {(dataset, seed) for dataset in config['datasets'] for seed in config['model_seeds']}
    actual = [(r['dataset'], r['model_seed']) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('saved trajectories do not match the frozen comparison blocks')
    split_seeds = dict(zip(config['model_seeds'], config['split_seeds']))
    for record in records:
        if record['split_seed'] != split_seeds[record['model_seed']] or record['trees_built'] != config['ntrees']:
            raise ValueError('saved source split/length differs from the accepted protocol')
    print(json.dumps({'VERIFIED_INPUTS': provenance, 'new_boosting_trainings': 0}), flush=True)
    directory = Path('results/revision')
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = directory / f'forecast_analysis_{stamp}.json'
    decision = analyze_forecasts(records, path)
    report = json.loads(path.read_text(encoding='utf-8'))
    report['provenance'] = provenance
    report['decision'] = decision
    report['protocol']['new_boosting_trainings'] = 0
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'FINAL_SUMMARY': {**decision, 'new_boosting_trainings': 0,
                     'forecast_evaluations': len(report['evaluation']), 'forecast_rows': len(report['rows']),
                     'verified_artifacts': provenance['verified_artifacts']}}), flush=True)
