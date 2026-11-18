"""Audit saved artifacts before any decision-aware HASBoost training."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.decision_aware_feasibility_core import (
    ArtifactCoverage,
    assess_sufficiency,
    audit_records,
    minimal_logging_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _merge(coverages: list[ArtifactCoverage], sizes: tuple[int, ...]) -> ArtifactCoverage:
    return ArtifactCoverage(
        records=sum(value.records for value in coverages),
        material_nodes=sum(value.material_nodes for value in coverages),
        nonempty_diagnostic_shortlists=sum(value.nonempty_diagnostic_shortlists for value in coverages),
        nonempty_full_shortlists=sum(value.nonempty_full_shortlists for value in coverages),
        complete_shortlists={
            size: sum(value.complete_shortlists[size] for value in coverages)
            for size in sizes
        },
        output_profile_records=sum(value.output_profile_records for value in coverages),
        selected_output_records=sum(value.selected_output_records for value in coverages),
    )


def run_feasibility_audit(config: dict[str, object]) -> dict[str, object]:
    protocol = config['decision_aware_protocol']
    sizes = tuple(int(value) for value in protocol['shortlist_sizes'])
    budgets = tuple(int(value) for value in protocol['budgets'])
    source_receipt = Path(str(config['source_receipt']))
    expected_receipt_hash = str(config['source_receipt_sha256'])
    actual_receipt_hash = _sha256(source_receipt)
    if actual_receipt_hash != expected_receipt_hash:
        raise ValueError('source receipt checksum mismatch')
    receipt = json.loads(source_receipt.read_text(encoding='utf-8'))
    artifact_root = Path(str(config['source_artifact_root']))

    artifacts: list[dict[str, object]] = []
    coverages: list[ArtifactCoverage] = []
    seen: set[Path] = set()
    for run in receipt['results']:
        original = Path(str(run['split_diagnostics_path']))
        relative = Path('causal_forecast/artifacts/20260918T135052Z/split_diagnostics') / original.name
        path = artifact_root / relative
        if path in seen:
            raise ValueError(f'duplicate diagnostic artifact: {path.name}')
        seen.add(path)
        expected_hash = receipt['artifact_sha256'][str(relative).replace('/', '\\')]
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f'diagnostic checksum mismatch: {path.name}')
        records = json.loads(path.read_text(encoding='utf-8'))
        coverage = audit_records(records, sizes, float(protocol['material_gain_epsilon']))
        coverages.append(coverage)
        artifacts.append({
            'dataset': run['dataset'],
            'model_seed': run['model_seed'],
            'split_seed': run['split_seed'],
            'path': str(path),
            'sha256': actual_hash,
            'source_split_diagnostics_top_k': 0,
            'source_split_output_attribution': bool(run['split_output_attribution']),
            'coverage': coverage.__dict__,
        })

    total = _merge(coverages, sizes)
    verdict = assess_sufficiency(total, sizes, budgets)
    report = {
        'phase': 'decision_aware_feasibility',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': 'insufficient_saved_evidence' if not verdict['sufficient_for_stage_a_metrics'] else 'ready_for_metrics',
        'source_receipt': str(source_receipt),
        'source_receipt_sha256': actual_receipt_hash,
        'artifacts': artifacts,
        'verdict': verdict,
        'pre_registered_gate': {
            'minimum_recall_at_k': protocol['minimum_recall_at_k'],
            'minimum_weighted_p95_regret_reduction': protocol['minimum_weighted_p95_regret_reduction'],
            'require_estimated_cost_below_full': protocol['require_estimated_cost_below_full'],
            'evaluated': bool(verdict['sufficient_for_stage_a_metrics']),
            'passed': False,
        },
        'why_existing_fields_are_insufficient': {
            'single_winner_identity': 'supports recall@1 only, not recall@2/4/8',
            'empty_top_k': 'candidate ranks beyond the winner were not persisted',
            'leave_one_output_regret': 'not recorded and would not recover per-candidate contribution profiles',
            'trajectory_snapshots': 'lack node assignments, candidate identities, and candidate histograms',
        },
        'minimal_logging_contract': minimal_logging_contract(max(sizes)),
        'next_action': (
            'Stop Stage A. Do not fit a decision-aware selector. If authorized later, collect one '
            'new full shadow trajectory with the frozen logging contract, then repeat Stage A.'
        ),
    }
    output = Path('results/revision') / 'decision_aware_feasibility.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'DECISION_AWARE_FEASIBILITY': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': report['scientific_status'],
        'artifacts_verified': len(artifacts),
        'records': total.records,
        'material_nodes': total.material_nodes,
        'nonempty_shortlists': total.nonempty_diagnostic_shortlists,
        'output_profile_records': total.output_profile_records,
        'gate_evaluated': report['pre_registered_gate']['evaluated'],
        'gate_passed': False,
        'new_boosting_fits': 0,
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
