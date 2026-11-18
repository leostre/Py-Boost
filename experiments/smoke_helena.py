"""Small end-to-end diagnostic run for the corrected HASBoost axes."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from py_boost.gpu.history_boosting import HistoryBasedBoostingModel


OPENML_DATA_ID = 41169


def load_helena(max_samples: int, seed: int):
    dataset = fetch_openml(data_id=OPENML_DATA_ID, as_frame=False)
    features = np.asarray(dataset.data, dtype=np.float32)
    target = LabelEncoder().fit_transform(np.asarray(dataset.target))

    if max_samples and max_samples < features.shape[0]:
        features, _, target, _ = train_test_split(
            features,
            target,
            train_size=max_samples,
            random_state=seed,
            stratify=target,
        )

    return features, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=12_000)
    parser.add_argument("--ntrees", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features, target = load_helena(args.max_samples, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=args.seed,
        stratify=target,
    )

    model = HistoryBasedBoostingModel(
        loss="crossentropy",
        ntrees=args.ntrees,
        lr=0.05,
        max_depth=5,
        min_data_in_leaf=20,
        subsample=0.5,
        sketch_outputs=25,
        sketch_method="rand",
        smoothing_alpha=0.9,
        # The smoke run forces the corrected branch after history is initialized.
        stabilization_threshold=100.0,
        es=0,
        seed=args.seed,
        verbose=1,
    )

    started = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_sets=[{"X": X_test, "y": y_test}],
    )
    elapsed = time.perf_counter() - started

    probabilities = model.predict(X_test)
    predictions = probabilities.argmax(axis=1)
    sketch = model.multioutput_sketch
    result = {
        "dataset": "helena",
        "openml_data_id": OPENML_DATA_ID,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "n_classes": int(np.unique(target).size),
        "trees_built": len(model.models),
        "history_decisions_applied": sketch.decisions_applied,
        "last_structure_sketch_shape": sketch.last_decision_shape,
        "last_unique_structure_sketch_shape": sketch.last_unique_decision_shape,
        "base_row_subsample": model.params["subsample"],
        "history_row_subsample": model.params["history_subsample"],
        "full_leaf_values": model.params["full_leaf_values"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities)),
        "fit_seconds": elapsed,
    }
    print(json.dumps(result, indent=2))

    if sketch.decisions_applied == 0:
        raise RuntimeError("The smoke run did not exercise history-based sketching")
    if sketch.last_unique_decision_shape != sketch.last_decision_shape:
        raise RuntimeError("History sketching sampled duplicate rows or outputs")


if __name__ == "__main__":
    main()
