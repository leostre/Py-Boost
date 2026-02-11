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

METRICS = {
    "f1": partial(f1_score, average=AVERAGE, zero_division=ZERO_DIVISION),
    "accuracy": accuracy_score,
    "precision": partial(
        precision_score, average=AVERAGE, zero_division=ZERO_DIVISION
    ),
    "recall": partial(recall_score, average=AVERAGE, zero_division=ZERO_DIVISION),
}

ROCAUC_SCORE = partial(roc_auc_score, average=AVERAGE, multi_class="ovr")

PRED_THR = 0.5


def multilabel_postprocess(pred: np.ndarray) -> np.ndarray:
    return (pred > PRED_THR).astype(int)


def onelabel_postproc(pred: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()
    labels = lb.fit_transform(np.argmax(pred, -1))
    return labels


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

