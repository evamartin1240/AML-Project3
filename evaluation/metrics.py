# src/evaluation/metrics.py

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    f1_score,
    balanced_accuracy_score
)


def evaluate_model(y_true, y_prob, threshold=0.5):
    """
    Compute evaluation metrics from true labels and predicted probabilities.
    """

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
    }

    return metrics

