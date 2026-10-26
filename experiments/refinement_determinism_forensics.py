"""Run the preregistered H-DETERMINISM-1 artifact-only forensic analysis."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.refinement_determinism_forensics_core import (
    analyze_refinement_determinism,
)


def _load_verified_json(spec: dict[str, object]) -> object:
    path = Path(str(spec['path']))
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = str(spec['sha256']).lower()
    if digest != expected:
        raise ValueError(f'artifact hash mismatch for {path}: {digest} != {expected}')
    return json.loads(payload.decode('utf-8'))


def run_refinement_determinism_forensics(config: dict[str, object]) -> dict[str, object]:
    """Compare the frozen H-APPLY-1 treatment and repeat artifacts."""

    protocol = config['refinement_determinism_forensics']
    sources = protocol['sources']
    treatment_trace = _load_verified_json(sources['treatment_refinement_trace'])
    repeat_trace = _load_verified_json(sources['repeat_refinement_trace'])
    treatment_diagnostics = _load_verified_json(sources['treatment_split_diagnostics'])
    repeat_diagnostics = _load_verified_json(sources['repeat_split_diagnostics'])
    analysis = analyze_refinement_determinism(
        treatment_trace,
        repeat_trace,
        treatment_diagnostics,
        repeat_diagnostics,
    )
    report = {
        'phase': 'refinement_determinism_forensics',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_experiment_id': protocol['source_experiment_id'],
        'source_run_id': protocol['source_run_id'],
        'new_boosting_fits': 0,
        'sources': sources,
        'analysis': analysis,
    }
    output = Path('results/revision/refinement_determinism_forensics.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(json.dumps({'REFINEMENT_DETERMINISM_FORENSICS': report}, ensure_ascii=False), flush=True)
    print(json.dumps({'FINAL_SUMMARY': {
        'phase': report['phase'],
        'scientific_status': analysis['scientific_status'],
        'classification': analysis['classification'],
        'new_boosting_fits': 0,
        'aligned_nodes': analysis['aligned_nodes'],
        'first_refined_identity_mismatch': analysis['first_refined_identity_mismatch'],
        'report': str(output.resolve()),
    }}, ensure_ascii=False), flush=True)
    return report
