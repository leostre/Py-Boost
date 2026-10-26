"""Offline exact-shortlist refinement experiment on a frozen artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.decision_aware_refinement_core import (
    assess_refinement_gate,
    evaluate_exact_refinement,
)
from experiments.split_diagnostic_artifact import load_split_diagnostic_artifact


def run_refinement(config: dict[str, object]) -> dict[str, object]:
    protocol = config['decision_aware_refinement']
    artifact_root = Path(str(protocol['artifact_root']))
    records = load_split_diagnostic_artifact(
        artifact_root / 'split_diagnostics.json',
        artifact_root / 'candidate_profiles.npz',
    )
    policies = {
        name: {int(budget): int(size) for budget, size in mapping.items()}
        for name, mapping in protocol['policies'].items()
    }
    budgets = tuple(int(value) for value in protocol['budgets'])
    evaluation = evaluate_exact_refinement(
        records,
        policies,
        budgets,
        int(protocol['candidate_count']),
        float(protocol['material_gain_epsilon']),
    )
    gate = assess_refinement_gate(
        evaluation,
        str(protocol['primary_policy']),
        budgets,
        float(protocol['minimum_weighted_p95_regret_reduction']),
    )
    report = {
        'phase': 'decision_aware_refinement',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_status': 'offline_gate_passed' if gate['passed'] else 'offline_gate_failed',
        'protocol': protocol,
        'evaluation': evaluation,
        'gate': gate,
        'new_boosting_fits': 0,
        'scope_limit': (
            'This run reranks saved candidates exactly by all-output contributions. '
            'The work ratio is analytical and excludes top-k extraction overhead; '
            'it is not a measured GPU runtime.'
        ),
    }
    output = Path('results/revision/decision_aware_refinement.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'DECISION_AWARE_REFINEMENT': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'new_boosting_fits': 0,
        'scientific_status': report['scientific_status'],
        'primary_policy': gate['primary_policy'],
        'gate_passed': gate['passed'],
        'per_budget': evaluation['per_policy'][gate['primary_policy']],
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
