"""
Compare measured conformal coverage with what theory predicts.

Theory (see notes/predictions.md): with c calibration rows and k = ceil((c + 1) * 0.9),
the coverage of ONE run follows a Beta(k, c + 1 - k) distribution (a curve for a random
percentage). This holds for any model, as long as calibration and test data are
exchangeable (drawn the same way).

This script:
  1. computes the predicted mean, standard deviation (typical run-to-run distance) and
     5%-95% range for each sample size n
  2. measures the same things from the experiment results, per dataset / model / n /
     noise setting
  3. saves a comparison table and two plots

Run from the project folder:
    python -m src.coverage_theory
(It reads results/tables/raw_results.csv and, if present, raw_results_tabpfn.csv.)
"""

import math

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

from src import config
from src.report import STYLE, load_results

TARGET = 1 - config.ALPHA


def theory_for(n):
    """Predicted coverage distribution for sample size n (our 25% calibration split)."""
    c = int(round(n * config.CALIBRATION_SHARE))       # calibration rows, as in experiment.py
    k = math.ceil((c + 1) * (1 - config.ALPHA))         # rank of the cut-off score
    if k > c:                                           # too few rows: sets = all classes
        return dict(n=n, n_cal=c, k=k, mean=1.0, sd=0.0, low=1.0, high=1.0, below_85=0.0)
    curve = beta(k, c + 1 - k)
    return dict(n=n, n_cal=c, k=k, mean=curve.mean(), sd=curve.std(),
                low=curve.ppf(0.05), high=curve.ppf(0.95), below_85=curve.cdf(0.85))


def compare(raw):
    """One row per dataset / model / n / noise setting: measured vs predicted."""
    theory = pd.DataFrame([theory_for(n) for n in sorted(raw["n"].unique())])
    keys = ["dataset", "model", "n", "noise_where", "noise_level"]
    measured = raw.groupby(keys)["coverage"].agg(
        measured_mean="mean", measured_sd="std", repeats="size",
        measured_below_85=lambda x: (x < 0.85).mean()).reset_index()
    table = measured.merge(theory, on="n")
    table = table.rename(columns={"mean": "theory_mean", "sd": "theory_sd",
                                  "below_85": "theory_below_85"})
    # Ratio > 1 means runs swing MORE than theory predicts.
    table["sd_ratio"] = table["measured_sd"] / table["theory_sd"]
    # Is the measured mean within 2 standard errors of the theory mean?
    # (standard error = how precisely 20 repeats pin down an average = sd / sqrt(repeats))
    se = table["theory_sd"] / np.sqrt(table["repeats"])
    table["mean_ok"] = (table["measured_mean"] - table["theory_mean"]).abs() <= 2 * se
    return table


def plot_runs_vs_theory(raw, ds, out):
    """Clean data: every run's coverage (dots) on top of the predicted 5%-95% band."""
    data = raw[(raw["dataset"] == ds) & (raw["noise_where"] == "none")]
    sizes = sorted(data["n"].unique())
    th = pd.DataFrame([theory_for(n) for n in sizes])
    x = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.fill_between(x, th["low"], th["high"], color="#c3c2b7", alpha=0.5,
                    label="theory: middle 90% of runs")
    ax.plot(x, th["mean"], color="#52514e", linewidth=1.5, label="theory: average")
    ax.axhline(TARGET, color="#52514e", linestyle="--", linewidth=1, label="target 90%")

    models = [m for m in STYLE if m in data["model"].unique()]
    width = 0.6 / max(len(models), 1)
    for i, model in enumerate(models):
        part = data[data["model"] == model]
        offset = (i - (len(models) - 1) / 2) * width     # side-by-side per model
        xs = part["n"].map({n: j for j, n in enumerate(sizes)}) + offset
        ax.scatter(xs, part["coverage"], s=12, alpha=0.6, color=STYLE[model]["color"],
                   marker=STYLE[model]["marker"], label=STYLE[model]["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("sample size n")
    ax.set_ylabel("coverage of one run")
    ax.set_title(f"{ds}: conformal coverage per run vs theory (clean data)", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_spread_vs_theory(table, ds, out):
    """Run-to-run spread (standard deviation) measured per model vs the theory line."""
    data = table[(table["dataset"] == ds) & (table["noise_where"] == "none")]
    sizes = sorted(data["n"].unique())
    th = pd.DataFrame([theory_for(n) for n in sizes])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(sizes, th["sd"], color="#52514e", linewidth=2.5, label="theory (any model)")
    for model, s in STYLE.items():
        part = data[data["model"] == model].sort_values("n")
        if part.empty:
            continue
        ax.plot(part["n"], part["measured_sd"], color=s["color"], marker=s["marker"],
                linewidth=1.5, markersize=6, label=s["label"])
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("sample size n (log scale)")
    ax.set_ylabel("standard deviation of coverage")
    ax.set_title(f"{ds}: how much coverage swings between runs", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    raw = load_results()
    table = compare(raw)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLES_DIR / "coverage_vs_theory.csv", index=False)

    for ds in raw["dataset"].unique():
        plot_runs_vs_theory(raw, ds, config.FIGURES_DIR / f"{ds}_coverage_runs_vs_theory.png")
        plot_spread_vs_theory(table, ds, config.FIGURES_DIR / f"{ds}_coverage_spread_vs_theory.png")

    clean = table[table["noise_where"] == "none"]
    cols = ["dataset", "model", "n", "measured_mean", "theory_mean", "measured_sd",
            "theory_sd", "sd_ratio", "mean_ok"]
    print(clean[cols].round(3).to_string(index=False))
    print(f"\nSaved {config.TABLES_DIR / 'coverage_vs_theory.csv'} and plots in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
