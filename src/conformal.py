"""
Split conformal prediction for classification.

Idea in plain words:
  1. Train a model on one part of the data.
  2. On a separate "calibration" part, measure how surprised the model is by the
     true answer: score = 1 - (probability the model gave the true class).
  3. Pick a cut-off q so that about (1 - alpha) of calibration scores fall below it.
  4. For a new example, the prediction SET is every class whose score would be
     below q, i.e. every class with probability >= 1 - q.

Guarantee: if calibration and test data come from the same distribution
(exchangeable = order doesn't matter, same source), the set contains the true
class at least (1 - alpha) of the time ON AVERAGE over many calibration draws.
It does NOT promise that every single run hits 90%; with tiny calibration sets,
individual runs can swing a lot. Measuring that swing is part of this project.
"""

import numpy as np


def conformal_threshold(proba_cal, y_cal, alpha):
    """
    Compute the cut-off q from the calibration set.

    Uses the standard finite-sample correction: the ceil((n+1)(1-alpha))-th
    smallest score. If the calibration set is too small for that rank to exist
    (n < 1/alpha - 1), q = 1, which means "include every class".
    """
    n = len(y_cal)
    scores = 1.0 - proba_cal[np.arange(n), y_cal]
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return 1.0
    return float(np.sort(scores)[rank - 1])


def prediction_sets(proba, q):
    """True/False table: row i, column k is True if class k is in example i's set."""
    return (1.0 - proba) <= q


def coverage_and_size(proba_test, y_test, q):
    """
    coverage - share of test examples whose set contains the true class
    set_size - average number of classes per set (smaller = more useful)
    """
    sets = prediction_sets(proba_test, q)
    covered = sets[np.arange(len(y_test)), y_test]
    return float(covered.mean()), float(sets.sum(axis=1).mean())
