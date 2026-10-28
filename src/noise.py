"""
Add controlled noise to inputs, to mimic unreliable measurements.

- Numeric columns: add Gaussian noise (random jitter from a bell curve) whose size
  is `level` times that column's standard deviation (its natural spread).
- Categorical columns: with probability `level`, replace a value with a category
  picked at random from that column's categories.

Noise only touches the inputs (X), never the labels (y).
"""

import numpy as np
import pandas as pd


def add_noise(X, numeric, categorical, level, feature_std, rng):
    """
    Return a noisy copy of X (the original is not changed).

    X           - DataFrame of features
    numeric     - list of numeric column names
    categorical - list of categorical column names
    level       - noise strength (0 = no noise)
    feature_std - dict {column: standard deviation}, measured once on the full
                  dataset so the same noise scale is used in every sample
    rng         - numpy random generator (makes the noise repeatable)
    """
    X_noisy = X.copy()
    if level == 0:
        return X_noisy

    for col in numeric:
        jitter = rng.normal(loc=0.0, scale=level * feature_std[col], size=len(X))
        X_noisy[col] = X[col].to_numpy(dtype=float) + jitter

    for col in categorical:
        categories = X[col].cat.categories
        flip = rng.random(len(X)) < level            # which rows get changed
        random_values = rng.choice(categories, size=len(X))
        new_values = np.where(flip, random_values, X[col].astype(str).to_numpy())
        X_noisy[col] = pd.Categorical(new_values, categories=categories)

    return X_noisy


def feature_stds(X, numeric):
    """Standard deviation of each numeric column (used to scale the noise)."""
    return {col: float(X[col].std()) for col in numeric}
