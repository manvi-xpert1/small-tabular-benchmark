"""
Scores for judging predictions.

All functions take:
  y_true - true labels (0, 1, 2, ...)
  proba  - predicted probabilities, one row per example, one column per class
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss


def expected_calibration_error(y_true, proba, n_bins=10):
    """
    Expected calibration error (ECE): checks whether confidence matches reality.

    For each prediction, take the model's confidence (its top probability).
    Group predictions into bins by confidence (0-10%, 10-20%, ...).
    In each bin compare average confidence with actual accuracy.
    ECE = weighted average gap. 0 = perfectly calibrated.
    """
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
            ece += in_bin.mean() * gap
    return ece


def all_metrics(y_true, proba, n_classes, ordinal, n_bins=10):
    """Return a dict of every score used in the results table."""
    y_pred = proba.argmax(axis=1)

    # Log loss breaks on exact zeros (log(0) = minus infinity), so clip slightly.
    proba_safe = np.clip(proba, 1e-15, 1)
    proba_safe = proba_safe / proba_safe.sum(axis=1, keepdims=True)

    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, proba_safe, labels=list(range(n_classes))),
        "ece": expected_calibration_error(y_true, proba, n_bins),
    }
    # Mean absolute error in "class steps": predicting 6 when truth is 5 costs 1.
    # Only meaningful when classes are ordered; otherwise left empty (NaN) so every
    # results row has the same columns.
    scores["mae"] = float(np.mean(np.abs(y_pred - y_true))) if ordinal else np.nan
    return scores
