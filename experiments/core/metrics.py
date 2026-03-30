from functools import partial

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


AVERAGE = "weighted"
ZERO_DIVISION = np.nan


def exact_match_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Exact match ratio:
    (number of samples with fully correct label vector) / N.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_arr.shape}, y_pred={y_pred_arr.shape}")

    # Single-label: exact match is standard accuracy.
    if y_true_arr.ndim == 1:
        return float(np.mean(y_true_arr == y_pred_arr))

    # Multilabel: all labels in a sample must match.
    return float(np.mean(np.all(y_true_arr == y_pred_arr, axis=1)))


METRICS = {
    "f1": partial(f1_score, average=AVERAGE, zero_division=ZERO_DIVISION),
    "f1_micro": partial(
        f1_score, average="micro", zero_division=ZERO_DIVISION
    ),
    "f1_macro": partial(
        f1_score, average="macro", zero_division=ZERO_DIVISION
    ),
    "accuracy": accuracy_score,
    "exact_match": exact_match_score,
    "precision": partial(
        precision_score, average=AVERAGE, zero_division=ZERO_DIVISION
    ),
    "recall": partial(recall_score, average=AVERAGE, zero_division=ZERO_DIVISION),
}

ROCAUC_SCORE = partial(roc_auc_score, average=AVERAGE, multi_class="ovr")


def onelabel_postproc(pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """
    One-hot predictions aligned to ``n_classes`` columns.

    Fitting ``LabelBinarizer`` only on predicted labels can yield fewer columns
    than ``y_true`` (e.g. one-hot from training with all classes). Pass
    ``n_classes`` (typically ``y_test.shape[1]`` or ``context.n_classes``).
    """
    from sklearn.preprocessing import LabelBinarizer

    pred = np.asarray(pred)
    y_pred_labels = np.argmax(pred, axis=-1)
    if n_classes is None:
        n_classes = int(pred.shape[-1])
    lb = LabelBinarizer()
    lb.fit(np.arange(n_classes, dtype=int))
    return lb.transform(y_pred_labels)


def optimize_threshold_per_label(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> np.ndarray:
    """
    Optimize per-label decision thresholds for multilabel problems.

    Matches the requested logic: for each label, compute precision/recall
    over thresholds and pick the threshold maximizing F1.
    """
    if metric != "f1":
        raise ValueError(
            "Only metric='f1' is supported in optimize_threshold_per_label"
        )
    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError(
            "Expected y_true and y_prob to be 2D arrays (n_samples, n_labels)"
        )
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")

    n_labels = y_true.shape[1]
    thresholds: list[float] = []

    from sklearn.metrics import precision_recall_curve

    for i in range(n_labels):
        precision, recall, thresh = precision_recall_curve(
            y_true[:, i], y_prob[:, i]
        )

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = int(np.argmax(f1_scores))

        # precision_recall_curve last threshold corresponds to 1.0
        if best_idx < len(thresh):
            best_thresh = float(thresh[best_idx])
        else:
            best_thresh = 1.0
        thresholds.append(best_thresh)

    return np.array(thresholds, dtype=float)


def calculate_aggregated_metrics(fold_scores):
    """Aggregate per-fold metric scores into summary statistics."""
    aggregated = {}
    for metric_name, scores in fold_scores.items():
        if scores:
            scores_array = np.array(scores)
            aggregated.update(
                {
                    f"mean_{metric_name}": np.mean(scores_array),
                    f"std_{metric_name}": np.std(scores_array),
                    f"min_{metric_name}": np.min(scores_array),
                    f"max_{metric_name}": np.max(scores_array),
                    f"median_{metric_name}": np.median(scores_array),
                }
            )
    return aggregated
