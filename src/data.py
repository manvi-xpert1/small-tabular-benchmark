"""
Load and clean the two real datasets.

Each loader returns a `Dataset` with:
  X            - table of input features (pandas DataFrame)
  y            - class labels as whole numbers 0, 1, 2, ... (numpy array)
  numeric      - names of numeric columns
  categorical  - names of categorical (text-category) columns
  class_names  - what each label number means
  ordinal      - True if classes have a natural order (wine quality does)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DATA_DIR


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    numeric: list
    categorical: list
    class_names: list
    ordinal: bool


def _check_file(path):
    """Stop with a clear message if a data file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"See data/README.md for download instructions."
        )


# ---------------------------------------------------------------------------
# Wine Quality
# ---------------------------------------------------------------------------
def load_wine(data_dir=DATA_DIR):
    """
    Red + white wine combined into one table, with a 'colour' column.

    Quality scores run 3-9, but 3 and 9 are very rare (about 30 and 5 wines).
    Small samples often contain none of them, so we merge 3 into 4 and 9 into 8.
    That leaves 5 ordered classes: 4, 5, 6, 7, 8  ->  labels 0, 1, 2, 3, 4.
    """
    red_path = data_dir / "winequality-red.csv"
    white_path = data_dir / "winequality-white.csv"
    _check_file(red_path)
    _check_file(white_path)

    # The files use ';' between columns.
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")
    red["colour"] = "red"
    white["colour"] = "white"
    df = pd.concat([red, white], ignore_index=True)

    # Merge rare quality scores.
    quality = df.pop("quality").clip(lower=4, upper=8)
    class_values = [4, 5, 6, 7, 8]
    y = quality.map({q: i for i, q in enumerate(class_values)}).to_numpy()

    categorical = ["colour"]
    numeric = [c for c in df.columns if c not in categorical]
    df["colour"] = df["colour"].astype("category")

    class_names = ["<=4", "5", "6", "7", ">=8"]
    return Dataset("wine", df, y, numeric, categorical, class_names, ordinal=True)


# ---------------------------------------------------------------------------
# Adult Income
# ---------------------------------------------------------------------------
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]


def load_adult(data_dir=DATA_DIR):
    """
    adult.data and adult.test combined into one pool (about 48,800 people).

    Cleaning choices:
      - 'adult.test' starts with a comment line and its labels end in '.', e.g. '>50K.'
      - '?' means unknown; kept as its own category 'Unknown' rather than dropping rows.
      - 'fnlwgt' is a census sampling weight, not a property of the person -> dropped.
      - 'education' (text) duplicates 'education_num' (number) -> text version dropped.
    Label: 1 = income >50K, 0 = income <=50K.
    """
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    _check_file(train_path)
    _check_file(test_path)

    read_options = dict(names=ADULT_COLUMNS, skipinitialspace=True, na_values="?")
    part1 = pd.read_csv(train_path, **read_options)
    part2 = pd.read_csv(test_path, skiprows=1, **read_options)
    df = pd.concat([part1, part2], ignore_index=True)
    df = df.dropna(subset=["income"])  # drops blank trailing lines, if any

    income = df.pop("income").str.rstrip(".")
    y = (income == ">50K").astype(int).to_numpy()

    df = df.drop(columns=["fnlwgt", "education"])

    categorical = ["workclass", "marital_status", "occupation", "relationship",
                   "race", "sex", "native_country"]
    numeric = [c for c in df.columns if c not in categorical]
    for col in categorical:
        df[col] = df[col].fillna("Unknown").astype("category")

    return Dataset("adult", df.reset_index(drop=True), y, numeric, categorical,
                   ["<=50K", ">50K"], ordinal=False)


LOADERS = {"wine": load_wine, "adult": load_adult}


def load_dataset(name, data_dir=DATA_DIR):
    return LOADERS[name](data_dir)
