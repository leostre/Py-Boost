"""Tests for the Age Prediction provenance audit."""

from __future__ import annotations

import csv

from experiments.age_prediction_provenance import summarize_folds


def _write_fold(path, train_ids, test_ids, feature_count=64):
    path.mkdir()
    columns = ["client_id", "bins"] + [f"emb_{index:04d}" for index in range(feature_count)]
    for name, ids in (("emb_train.csv", train_ids), ("emb_test.csv", test_ids)):
        with (path / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for client_id in ids:
                row = {column: 0 for column in columns}
                row["client_id"] = client_id
                writer.writerow(row)


def test_summarize_folds_distinguishes_row_instances_from_unique_clients(tmp_path):
    _write_fold(tmp_path / "fold_0", ["a", "b"], ["c"])
    _write_fold(tmp_path / "fold_1", ["a", "c"], ["b"])

    summary = summarize_folds(tmp_path, expected_folds=2)

    assert summary["total_row_instances"] == 6
    assert summary["global_unique_clients"] == 3
    assert summary["all_fold_unions_equal"] is True
    assert summary["all_feature_sets_equal"] is True
    assert summary["embedding_features"] == 64
