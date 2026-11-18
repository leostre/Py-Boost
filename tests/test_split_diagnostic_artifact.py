from __future__ import annotations

import json

import numpy as np

from experiments.split_diagnostic_artifact import (
    load_split_diagnostic_artifact,
    write_split_diagnostic_artifact,
)
from experiments.decision_aware_collection_replay import _json_compatible


def test_dense_candidate_profiles_round_trip_through_compressed_sidecar(tmp_path):
    records = [
        {
            'node_id': 7,
            'n_outputs': 3,
            'candidate_output_contributions': [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        },
        {
            'node_id': 8,
            'n_outputs': 3,
            'candidate_output_contributions': [[7.0, 8.0, 9.0]],
        },
    ]
    metadata_path = tmp_path / 'diagnostics.json'
    profile_path = write_split_diagnostic_artifact(records, metadata_path)

    assert profile_path is not None
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    assert 'candidate_output_contributions' not in metadata[0]
    with np.load(profile_path, allow_pickle=False) as payload:
        assert payload['candidate_output_contributions'].shape == (3, 3)
    restored = load_split_diagnostic_artifact(metadata_path, profile_path)
    assert restored == records


def test_artifact_without_profiles_needs_no_sidecar(tmp_path):
    path = tmp_path / 'diagnostics.json'
    profile_path = write_split_diagnostic_artifact([{'node_id': 1}], path)
    assert profile_path is None
    assert load_split_diagnostic_artifact(path) == [
        {'node_id': 1, 'candidate_output_contributions': None}
    ]


def test_json_compatibility_normalizes_integer_mapping_keys():
    assert _json_compatible({'complete_shortlists': {8: 12, 64: 12}}) == {
        'complete_shortlists': {'8': 12, '64': 12}
    }
