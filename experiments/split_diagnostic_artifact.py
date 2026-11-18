"""Compact persistence for split diagnostics with dense output profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


PROFILE_KEY = 'candidate_output_contributions'


def write_split_diagnostic_artifact(
    records: Iterable[Mapping[str, object]],
    metadata_path: Path,
) -> Path | None:
    """Write JSON metadata and move dense candidate profiles to compressed NPZ."""

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    compact_records: list[dict[str, object]] = []
    profile_rows: list[list[float]] = []
    output_count: int | None = None
    for source in records:
        record = dict(source)
        profiles = record.pop(PROFILE_KEY, None)
        if profiles:
            current_output_count = int(record['n_outputs'])
            if output_count is None:
                output_count = current_output_count
            elif current_output_count != output_count:
                raise ValueError('all candidate profiles must use the same output count')
            start = len(profile_rows)
            profile_rows.extend([list(values) for values in profiles])
            record['candidate_profile_offset'] = start
            record['candidate_profile_count'] = len(profiles)
        compact_records.append(record)

    metadata_path.write_text(
        json.dumps(compact_records, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    if not profile_rows:
        return None

    profile_path = metadata_path.with_name(f'{metadata_path.stem}_profiles.npz')
    values = np.asarray(profile_rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != output_count:
        raise ValueError('candidate profiles must form a dense candidate-by-output matrix')
    np.savez_compressed(profile_path, **{PROFILE_KEY: values})
    return profile_path


def load_split_diagnostic_artifact(
    metadata_path: Path,
    profile_path: Path | None = None,
) -> list[dict[str, object]]:
    """Load compact diagnostics and restore profile rows into each record."""

    records = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError('split diagnostic metadata must contain a record list')
    if profile_path is None:
        inferred = metadata_path.with_name(f'{metadata_path.stem}_profiles.npz')
        profile_path = inferred if inferred.exists() else None
    values = None
    if profile_path is not None:
        with np.load(profile_path, allow_pickle=False) as payload:
            values = np.asarray(payload[PROFILE_KEY])

    restored: list[dict[str, object]] = []
    for source in records:
        record = dict(source)
        count = int(record.pop('candidate_profile_count', 0))
        offset = int(record.pop('candidate_profile_offset', 0))
        if count:
            if values is None or offset < 0 or offset + count > len(values):
                raise ValueError('candidate profile reference is outside the profile array')
            record[PROFILE_KEY] = values[offset:offset + count].tolist()
        elif PROFILE_KEY not in record:
            record[PROFILE_KEY] = None
        restored.append(record)
    return restored
