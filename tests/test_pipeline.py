"""
Quick checks that the code works.

!!! TEST-ONLY: every dataset in this file is SYNTHETIC (randomly generated). !!!
It is used only to check the code runs and behaves sensibly. It is never used
for results.

Run from the project folder:   python -m pytest tests -q
"""

import numpy as np
import pandas as pd

from src.conformal import conformal_threshold, coverage_and_size
from src.data import Dataset
from src.metrics import all_metrics, expected_calibration_error
from src.models import make_model, predict_proba_all_classes
from src.noise import add_noise, feature_stds


def make_synthetic_test_only_dataset(n=400, seed=0):
    """SYNTHETIC, TEST-ONLY: 2 numeric features, 1 categorical, 3 classes."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": pd.Categorical(rng.choice(["x", "y", "z"], size=n)),
    })
    y = np.digitize(X["a"] + 0.5 * X["b"], [-0.5, 0.5])  # classes 0, 1, 2
    return Dataset("synthetic_test_only", X, y, ["a", "b"], ["c"],
                   ["0", "1", "2"], ordinal=True)


def test_zero_noise_changes_nothing():
    ds = make_synthetic_test_only_dataset()
    rng = np.random.default_rng(1)
    noisy = add_noise(ds.X, ds.numeric, ds.categorical, 0.0,
                      feature_stds(ds.X, ds.numeric), rng)
    pd.testing.assert_frame_equal(noisy, ds.X)


def test_noise_size_roughly_matches_level():
    ds = make_synthetic_test_only_dataset(n=20000)
    rng = np.random.default_rng(2)
    stds = feature_stds(ds.X, ds.numeric)
    noisy = add_noise(ds.X, ds.numeric, ds.categorical, 0.5, stds, rng)
    added = (noisy["a"] - ds.X["a"]).std()
    assert abs(added - 0.5 * stds["a"]) < 0.02
    # categorical: about 50% chosen for replacement, ~1/3 of those land on the same value
    changed = (noisy["c"].astype(str) != ds.X["c"].astype(str)).mean()
    assert 0.28 < changed < 0.38


def test_ece_zero_when_perfectly_calibrated():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y, proba) == 0.0


def test_conformal_coverage_close_to_target():
    """With exchangeable data and a big calibration set, coverage ~ 90%."""
    rng = np.random.default_rng(3)
    n, k = 20000, 4
    logits = rng.normal(size=(n, k))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=p) for p in proba])  # labels drawn from proba
    q = conformal_threshold(proba[:10000], y[:10000], alpha=0.1)
    coverage, _ = coverage_and_size(proba[10000:], y[10000:], q)
    assert 0.88 < coverage < 0.92


def test_tiny_calibration_set_gives_full_sets():
    """With fewer than 9 calibration points at alpha=0.1, sets must be all classes."""
    proba = np.full((5, 3), 1 / 3)
    q = conformal_threshold(proba, np.array([0, 1, 2, 0, 1]), alpha=0.1)
    _, size = coverage_and_size(proba, np.array([0, 1, 2, 0, 1]), q)
    assert q == 1.0 and size == 3


def test_missing_class_gets_zero_probability():
    """Train without class 2: the probability table must still have 3 columns."""
    ds = make_synthetic_test_only_dataset()
    keep = ds.y != 2
    model = make_model("logistic_regression", ds.numeric, ds.categorical, seed=0)
    model.fit(ds.X[keep], ds.y[keep])
    proba = predict_proba_all_classes(model, ds.X, 3)
    assert proba.shape == (len(ds.y), 3)
    assert np.allclose(proba[:, 2], 0)
    assert np.allclose(proba.sum(axis=1), 1)


def test_classic_models_train_and_score():
    ds = make_synthetic_test_only_dataset()
    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        model = make_model(name, ds.numeric, ds.categorical, seed=0)
        model.fit(ds.X[:300], ds.y[:300])
        proba = predict_proba_all_classes(model, ds.X[300:], 3)
        scores = all_metrics(ds.y[300:], proba, 3, ordinal=True)
        assert scores["accuracy"] > 0.6, name  # easy synthetic task
