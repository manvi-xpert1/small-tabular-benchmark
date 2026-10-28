"""
Main experiment loop.

For each dataset:
  1. Set aside a fixed reference test set (same for every run).
  2. For each sample size n and each repeat:
       - draw n rows at random from the rest of the data (the "pool")
       - split them into a training part and a calibration part
       - for each model:
           a. train on CLEAN data  -> evaluate on clean test data
                                   -> evaluate on NOISY test data ("test_only":
                                      noise appears only after training, a shift)
           b. for each noise level, train on NOISY data -> evaluate on NOISY test
              data ("both": noise everywhere, the same distribution)
  3. Save one row of results per (model, n, repeat, noise setting).

Results are appended to a CSV after every model fit group, and finished work is
skipped on restart, so a Colab disconnect loses only a few minutes.

Run from the project folder:
    python -m src.experiment --quick          (small check, a few minutes)
    python -m src.experiment                  (full experiment)
    python -m src.experiment --models logistic_regression random_forest
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.conformal import conformal_threshold, coverage_and_size
from src.data import load_dataset
from src.metrics import all_metrics
from src.models import make_model, predict_proba_all_classes
from src.noise import add_noise, feature_stds

DATASET_IDS = {"wine": 1, "adult": 2}  # used only to build distinct random seeds

# Fixed column order for the results CSV. Every row is written with exactly these
# columns, so rows from different datasets always line up when appended.
RESULT_COLUMNS = ["dataset", "model", "n", "repeat", "n_train", "n_cal",
                  "noise_where", "noise_level", "accuracy", "balanced_accuracy",
                  "log_loss", "ece", "mae", "conformal_q", "coverage", "set_size",
                  "seconds"]


def split_reference(ds, rng):
    """Split full data into a fixed test set and a pool to draw samples from."""
    order = rng.permutation(len(ds.y))
    test_idx, pool_idx = order[:config.TEST_SIZE], order[config.TEST_SIZE:]
    return test_idx, pool_idx


def draw_sample(pool_idx, y, n, cal_share, rng):
    """
    Draw n rows from the pool and split into train / calibration parts.
    Redraws if the training part has fewer than 2 classes (a model can't learn
    from one class). This is rare except at n = 50.
    """
    n_cal = int(round(n * cal_share))
    for _ in range(100):
        chosen = rng.choice(pool_idx, size=n, replace=False)
        train_idx, cal_idx = chosen[n_cal:], chosen[:n_cal]
        if len(np.unique(y[train_idx])) >= 2:
            return train_idx, cal_idx
    raise RuntimeError("Could not draw a training sample with at least 2 classes.")


def evaluate(model, X_cal, y_cal, X_test, y_test, ds, n_classes):
    """Point-prediction scores plus conformal coverage and set size."""
    proba_cal = predict_proba_all_classes(model, X_cal, n_classes)
    proba_test = predict_proba_all_classes(model, X_test, n_classes)
    scores = all_metrics(y_test, proba_test, n_classes, ds.ordinal, config.ECE_BINS)
    q = conformal_threshold(proba_cal, y_cal, config.ALPHA)
    coverage, set_size = coverage_and_size(proba_test, y_test, q)
    scores.update(conformal_q=q, coverage=coverage, set_size=set_size)
    return scores


def already_done(out_path):
    """Set of (dataset, model, n, repeat) groups already saved in the CSV."""
    if not out_path.exists():
        return set()
    old = pd.read_csv(out_path)
    return set(zip(old["dataset"], old["model"], old["n"], old["repeat"]))


def run(datasets, models, sizes, repeats, noise_levels, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)
    if done:
        print(f"Resuming: {len(done)} finished groups found in {out_path.name}")

    for ds_name in datasets:
        ds = load_dataset(ds_name)
        n_classes = len(ds.class_names)
        stds = feature_stds(ds.X, ds.numeric)  # noise scale, from the full data

        test_idx, pool_idx = split_reference(
            ds, np.random.default_rng([config.BASE_SEED, DATASET_IDS[ds_name]]))
        X_test, y_test = ds.X.iloc[test_idx], ds.y[test_idx]
        print(f"\n=== {ds_name}: {len(ds.y)} rows, test={len(test_idx)}, "
              f"pool={len(pool_idx)}, classes={n_classes}")

        for n in sizes:
            for rep in range(repeats):
                # One random generator per (dataset, n, repeat): every model sees
                # the SAME sample and the SAME noise, so the comparison is fair.
                seed = [config.BASE_SEED, DATASET_IDS[ds_name], n, rep]
                rng = np.random.default_rng(seed)
                train_idx, cal_idx = draw_sample(pool_idx, ds.y, n,
                                                 config.CALIBRATION_SHARE, rng)
                X_tr, y_tr = ds.X.iloc[train_idx], ds.y[train_idx]
                X_cal, y_cal = ds.X.iloc[cal_idx], ds.y[cal_idx]

                # Pre-make noisy copies for every level (shared by all models).
                noisy = {}
                for level in noise_levels:
                    if level == 0:
                        continue
                    noise_rng = np.random.default_rng(seed + [int(level * 1000)])
                    noisy[level] = tuple(
                        add_noise(X, ds.numeric, ds.categorical, level, stds, noise_rng)
                        for X in (X_tr, X_cal, X_test))

                for model_name in models:
                    if (ds_name, model_name, n, rep) in done:
                        continue
                    rows, start = [], time.time()
                    base = dict(dataset=ds_name, model=model_name, n=n, repeat=rep,
                                n_train=len(y_tr), n_cal=len(y_cal))

                    # (a) train on clean data
                    model = make_model(model_name, ds.numeric, ds.categorical, rep)
                    model.fit(X_tr, y_tr)
                    rows.append({**base, "noise_where": "none", "noise_level": 0.0,
                                 **evaluate(model, X_cal, y_cal, X_test, y_test,
                                            ds, n_classes)})
                    for level, (_, _, X_test_noisy) in noisy.items():
                        # calibration stays clean: noise only appears at test time
                        rows.append({**base, "noise_where": "test_only",
                                     "noise_level": level,
                                     **evaluate(model, X_cal, y_cal, X_test_noisy,
                                                y_test, ds, n_classes)})

                    # (b) train on noisy data, test on noisy data
                    for level, (X_tr_n, X_cal_n, X_test_n) in noisy.items():
                        model = make_model(model_name, ds.numeric, ds.categorical, rep)
                        model.fit(X_tr_n, y_tr)
                        rows.append({**base, "noise_where": "both",
                                     "noise_level": level,
                                     **evaluate(model, X_cal_n, y_cal, X_test_n,
                                                y_test, ds, n_classes)})

                    seconds = time.time() - start
                    for r in rows:
                        r["seconds"] = round(seconds, 2)
                    table = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
                    table.to_csv(out_path, mode="a", index=False,
                                              header=not out_path.exists())
                    print(f"  {ds_name} n={n:<5} rep={rep:<3} {model_name:<20} "
                          f"acc(clean)={rows[0]['accuracy']:.3f}  {seconds:.1f}s",
                          flush=True)  # flush = show the line immediately


def main():
    parser = argparse.ArgumentParser(description="Run the small-data benchmark.")
    parser.add_argument("--datasets", nargs="+", default=["wine", "adult"])
    parser.add_argument("--models", nargs="+", default=config.MODEL_NAMES)
    parser.add_argument("--quick", action="store_true",
                        help="tiny run to check everything works")
    parser.add_argument("--out", default=None, help="output CSV path")
    # Optional overrides, e.g. to run a smaller TabPFN design when time is short.
    # Seeds depend only on (dataset, n, repeat, noise level), so a smaller run gives
    # exactly the same rows as the matching part of the full run.
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="sample sizes to run, e.g. --sizes 50 100 250")
    parser.add_argument("--repeats", type=int, default=None,
                        help="number of repeats per sample size, e.g. --repeats 10")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=None,
                        help="noise levels, must include 0, e.g. --noise-levels 0 0.25")
    args = parser.parse_args()

    if args.quick:
        sizes, repeats, levels = (config.QUICK_SAMPLE_SIZES, config.QUICK_N_REPEATS,
                                  config.QUICK_NOISE_LEVELS)
        default_name = "raw_results_quick.csv"
    else:
        sizes, repeats, levels = (config.SAMPLE_SIZES, config.N_REPEATS,
                                  config.NOISE_LEVELS)
        default_name = "raw_results.csv"

    if args.sizes is not None:
        sizes = args.sizes
    if args.repeats is not None:
        repeats = args.repeats
    if args.noise_levels is not None:
        levels = args.noise_levels

    out_path = config.TABLES_DIR / default_name if args.out is None else args.out
    run(args.datasets, args.models, sizes, repeats, levels, Path(out_path))


if __name__ == "__main__":
    main()
