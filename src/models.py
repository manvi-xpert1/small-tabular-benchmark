"""
The four models, built the same way so they can be swapped in the experiment loop.

All use default-ish settings (no tuning), so the comparison is about the models,
not about how much effort went into tuning each one.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def _one_hot_preprocessor(numeric, categorical):
    """
    Numeric columns -> standardised (mean 0, spread 1).
    Categorical columns -> one-hot encoded (one 0/1 column per category).
    'handle_unknown=ignore' means a category unseen in training becomes all zeros
    instead of crashing.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


class TabPFNWrapper:
    """
    Small wrapper so TabPFN takes the same pandas input as the other models.

    TabPFN wants numbers only, so categories are turned into integer codes
    (ordinal encoding), and TabPFN is told which columns are categorical.
    """

    def __init__(self, numeric, categorical, seed):
        self.numeric = numeric
        self.categorical = categorical
        self.seed = seed

    def fit(self, X, y):
        # Imported here so the other models still work if TabPFN is not installed.
        from tabpfn import TabPFNClassifier

        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1)
        X_num = X[self.numeric].to_numpy(dtype=float)
        X_cat = self.encoder.fit_transform(X[self.categorical]) if self.categorical else None
        X_all = X_num if X_cat is None else np.hstack([X_num, X_cat])

        cat_idx = list(range(len(self.numeric), X_all.shape[1]))
        self.model = TabPFNClassifier(categorical_features_indices=cat_idx or None,
                                      random_state=self.seed, device="auto")
        self.model.fit(X_all, y)
        self.classes_ = self.model.classes_
        return self

    def _to_array(self, X):
        X_num = X[self.numeric].to_numpy(dtype=float)
        if not self.categorical:
            return X_num
        return np.hstack([X_num, self.encoder.transform(X[self.categorical])])

    def predict_proba(self, X):
        return self.model.predict_proba(self._to_array(X))


def make_model(name, numeric, categorical, seed):
    """Build an untrained model by name."""
    if name == "logistic_regression":
        return Pipeline([
            ("prep", _one_hot_preprocessor(numeric, categorical)),
            ("model", LogisticRegression(max_iter=2000)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("prep", _one_hot_preprocessor(numeric, categorical)),
            ("model", RandomForestClassifier(n_estimators=300, random_state=seed,
                                             n_jobs=-1)),
        ])
    if name == "gradient_boosting":
        return Pipeline([
            ("prep", _one_hot_preprocessor(numeric, categorical)),
            ("model", HistGradientBoostingClassifier(random_state=seed)),
        ])
    if name == "tabpfn":
        return TabPFNWrapper(numeric, categorical, seed)
    raise ValueError(f"Unknown model: {name}")


def predict_proba_all_classes(model, X, n_classes):
    """
    Predicted probabilities with one column for EVERY class.

    Why this is needed: a small training sample may miss a class entirely.
    The model then only outputs columns for the classes it saw. We put those
    columns in the right place and give unseen classes probability 0.
    """
    proba_seen = model.predict_proba(X)
    proba = np.zeros((len(X), n_classes))
    proba[:, np.asarray(model.classes_, dtype=int)] = proba_seen
    return proba
