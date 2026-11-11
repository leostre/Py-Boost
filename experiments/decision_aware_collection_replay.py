"""Replay the candidate-profile audit without another boosting fit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.decision_aware_feasibility_core import (
    assess_shortlist_recall_gate,
    evaluate_shortlist_collection,
)
from experiments.split_diagnostic_artifact import load_split_diagnostic_artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_compatible(value):
    """Normalize mapping keys exactly as persisted JSON represents them."""

    return json.loads(json.dumps(value, ensure_ascii=False))


def replay_collection(config: dict[str, object]) -> dict[str, object]:
    """Verify the saved receipt and recompute the frozen shortlist metrics."""

    protocol = config['decision_aware_collection_replay']
    artifact_root = Path(str(protocol['artifact_root']))
    receipt_path = artifact_root / 'receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    verified_files = {}
    for name, expected in receipt['files'].items():
        path = artifact_root / name
        actual_hash = _sha256(path)
        actual_bytes = path.stat().st_size
        if actual_hash != expected['sha256'] or actual_bytes != expected['bytes']:
            raise ValueError(f'artifact receipt mismatch: {name}')
        verified_files[name] = {
            'bytes': actual_bytes,
            'sha256': actual_hash,
        }

    records = load_split_diagnostic_artifact(
        artifact_root / 'split_diagnostics.json',
        artifact_root / 'candidate_profiles.npz',
    )
    evaluation = evaluate_shortlist_collection(
        records,
        tuple(int(value) for value in protocol['shortlist_sizes']),
        tuple(int(value) for value in protocol['budgets']),
        float(protocol['material_gain_epsilon']),
        float(protocol['contribution_absolute_tolerance']),
        float(protocol['contribution_relative_tolerance']),
    )
    recall_gate = assess_shortlist_recall_gate(
        evaluation,
        tuple(int(value) for value in protocol['shortlist_sizes']),
        tuple(int(value) for value in protocol['budgets']),
        float(protocol['minimum_recall_at_k']),
    )
    source_report = json.loads(
        (artifact_root / 'collection_report.json').read_text(encoding='utf-8')
    )
    if _json_compatible(evaluation) != source_report['evaluation']:
        raise ValueError('replayed evaluation differs from the source report')
    if recall_gate != source_report['shortlist_recall_gate']:
        raise ValueError('replayed shortlist gate differs from the source report')
    report = {
        'phase': 'decision_aware_collection_replay',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_experiment_id': receipt['source_experiment_id'],
        'source_run_id': receipt['source_run_id'],
        'source_commit': receipt['source_commit'],
        'verified_files': verified_files,
        'protocol': protocol,
        'evaluation': evaluation,
        'shortlist_recall_gate': recall_gate,
        'source_result_match': True,
        'scientific_conclusion': (
            'The collection contract is valid and the pre-registered recall threshold '
            'is met with joint K=64. A separately pre-registered selector experiment '
            'may proceed; this replay does not train or evaluate a selector.'
        ),
        'new_boosting_fits': 0,
    }
    output = Path('results/revision/decision_aware_collection_replay.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'DECISION_AWARE_COLLECTION_REPLAY': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'new_boosting_fits': 0,
        'records': len(records),
        'contract_valid': evaluation['contract_valid'],
        'recall_at_k': {
            budget: values['recall_at_k']
            for budget, values in evaluation['per_budget'].items()
        },
        'joint_minimum_viable_k': recall_gate['joint_minimum_viable_k'],
        'shortlist_recall_gate_passed': recall_gate['passed'],
        'source_result_match': report['source_result_match'],
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
