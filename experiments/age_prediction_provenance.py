"""Reproducible provenance audit for the article's Age Prediction row."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _read_ids_and_features(path: Path) -> tuple[set[str], tuple[str, ...], int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        if "client_id" not in columns:
            raise ValueError(f"client_id is absent from {path}")
        features = tuple(column for column in columns if column.startswith("emb_"))
        ids = set()
        rows = 0
        for row in reader:
            ids.add(row["client_id"])
            rows += 1
    return ids, features, rows


def summarize_folds(folds_root: Path, expected_folds: int = 5) -> dict[str, object]:
    fold_dirs = sorted(path for path in folds_root.glob("fold_*") if path.is_dir())
    if len(fold_dirs) != expected_folds:
        raise ValueError(f"expected {expected_folds} folds, found {len(fold_dirs)}")

    fold_reports = []
    unions = []
    feature_sets = []
    total_row_instances = 0
    for fold_dir in fold_dirs:
        train_ids, train_features, train_rows = _read_ids_and_features(
            fold_dir / "emb_train.csv"
        )
        test_ids, test_features, test_rows = _read_ids_and_features(
            fold_dir / "emb_test.csv"
        )
        if train_features != test_features:
            raise ValueError(f"feature columns differ inside {fold_dir.name}")
        union = train_ids | test_ids
        unions.append(union)
        feature_sets.append(train_features)
        total_row_instances += train_rows + test_rows
        fold_reports.append({
            "fold": fold_dir.name,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "unique_clients": len(union),
            "train_test_overlap": len(train_ids & test_ids),
            "embedding_features": len(train_features),
        })

    all_feature_sets_equal = all(features == feature_sets[0] for features in feature_sets[1:])
    all_fold_unions_equal = all(union == unions[0] for union in unions[1:])
    global_clients = set().union(*unions)
    return {
        "folds": fold_reports,
        "fold_count": len(fold_reports),
        "total_row_instances": total_row_instances,
        "global_unique_clients": len(global_clients),
        "all_fold_unions_equal": all_fold_unions_equal,
        "all_feature_sets_equal": all_feature_sets_equal,
        "embedding_features": len(feature_sets[0]),
    }


def _git_import_metadata(repository: Path, relative_path: str) -> dict[str, str]:
    output = subprocess.check_output(
        [
            "git", "-C", str(repository), "log", "-1",
            "--format=%H%n%aI%n%an%n%s", "--", relative_path,
        ],
        text=True,
        encoding="utf-8",
    ).splitlines()
    if len(output) != 4:
        raise RuntimeError("could not recover the data import commit")
    return dict(zip(("commit", "authored_at", "author", "subject"), output))


def _article_claims(tex_paths: list[Path]) -> list[dict[str, object]]:
    reports = []
    pattern = re.compile(
        r"(?:Age Prediction|\\datasetAgePrediction\{\})[^\n]*"
        r"\(CoLES-(\d+)\).*?&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)"
    )
    for path in tex_paths:
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Age Prediction table row was not found in {path}")
        representation, samples, outputs, features = map(int, match.groups())
        reports.append({
            "path": str(path),
            "representation_dimensions": representation,
            "reported_samples": samples,
            "reported_outputs": outputs,
            "reported_features": features,
        })
    return reports


def _official_configurations(coles_repository: Path, ptls_repository: Path) -> dict[str, object]:
    original_config_path = (
        coles_repository / "experiments/scenario_age_pred/conf/mles_proj_head_params.json"
    )
    original_config = json.loads(original_config_path.read_text(encoding="utf-8"))
    original_default = int(original_config["params"]["projection_head"]["output_size"])
    original_sweep_path = (
        coles_repository / "experiments/scenario_age_pred/bin/scenario_projection_head.sh"
    )
    original_sweep = original_sweep_path.read_text(encoding="utf-8")
    sweep_match = re.search(r"for PRJ_SIZE in ([0-9 ]+)", original_sweep)
    if not sweep_match:
        raise ValueError("projection-size sweep was not found in the CoLES repository")
    sweep_dimensions = [int(value) for value in sweep_match.group(1).split()]

    ptls_config_path = ptls_repository / "scenario_age_pred/conf/mles_proj_head_params.yaml"
    ptls_sweep_path = ptls_repository / "scenario_age_pred/bin/scenario_projection_head.sh"
    ptls_sweep = ptls_sweep_path.read_text(encoding="utf-8")
    ptls_match = re.search(r"for PRJ_SIZE in ([0-9 ]+)", ptls_sweep)
    if not ptls_match:
        raise ValueError("projection-size sweep was not found in ptls-experiments")
    return {
        "coles_original": {
            "default_projection_dimensions": original_default,
            "swept_projection_dimensions": sweep_dimensions,
            "config_path": str(original_config_path),
            "sweep_path": str(original_sweep_path),
        },
        "ptls_experiments": {
            "swept_projection_dimensions": [
                int(value) for value in ptls_match.group(1).split()
            ],
            "config_path": str(ptls_config_path),
            "sweep_path": str(ptls_sweep_path),
        },
    }


def run_age_prediction_provenance(config: dict[str, object]) -> None:
    protocol = config["age_prediction_provenance"]
    folds_root = Path(protocol["folds_root"])
    source_repository = Path(protocol["source_repository"])
    article_paths = [Path(value) for value in protocol["article_tex_paths"]]
    coles_repository = Path(protocol["coles_repository"])
    ptls_repository = Path(protocol["ptls_repository"])

    folds = summarize_folds(folds_root, int(protocol["expected_folds"]))
    claims = _article_claims(article_paths)
    official = _official_configurations(coles_repository, ptls_repository)
    import_metadata = _git_import_metadata(
        source_repository, "experiments/data/age_pred"
    )

    article_claims_consistent = len({
        (
            claim["representation_dimensions"],
            claim["reported_samples"],
            claim["reported_outputs"],
            claim["reported_features"],
        )
        for claim in claims
    }) == 1
    exact_article_artifact_recovered = (
        article_claims_consistent
        and folds["global_unique_clients"] == claims[0]["reported_samples"]
        and folds["embedding_features"] == claims[0]["reported_features"]
    )
    row_count_explained_by_repeated_folds = (
        folds["all_fold_unions_equal"]
        and folds["total_row_instances"]
        == folds["global_unique_clients"] * folds["fold_count"]
        == claims[0]["reported_samples"]
    )
    gate_passed = (
        article_claims_consistent
        and folds["fold_count"] == int(protocol["expected_folds"])
        and folds["all_fold_unions_equal"]
        and folds["all_feature_sets_equal"]
        and folds["embedding_features"] == 64
        and row_count_explained_by_repeated_folds
        and 64 in official["coles_original"]["swept_projection_dimensions"]
        and 128 in official["coles_original"]["swept_projection_dimensions"]
        and not exact_article_artifact_recovered
    )
    report = {
        "phase": "age_prediction_provenance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "article_artifact_not_recovered" if gate_passed else "audit_gate_failed",
        "protocol": protocol,
        "article_claims": claims,
        "stored_folds": folds,
        "source_import": import_metadata,
        "official_configurations": official,
        "article_claims_consistent": article_claims_consistent,
        "row_count_explained_by_repeated_folds": row_count_explained_by_repeated_folds,
        "exact_article_artifact_recovered": exact_article_artifact_recovered,
        "audit_gate_passed": gate_passed,
        "conclusion": (
            "The stored HASBoost data contain 30000 unique clients repeated across five folds "
            "and 64 embedding features. Official CoLES scenarios support both 64 and 128 "
            "dimensions, but the stored CSV files provide no checkpoint or variant identity. "
            "Therefore the article's exact CoLES-128 artifact is not recoverable from the "
            "available provenance chain."
        ),
    }
    output_path = Path(protocol["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"FINAL_SUMMARY": report}, ensure_ascii=False), flush=True)
    if not gate_passed:
        raise SystemExit(1)
