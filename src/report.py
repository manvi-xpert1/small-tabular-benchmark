"""
Turn the raw results CSV into summary tables and plots.

Run from the project folder:
    python -m src.report                       (uses results/tables/raw_results.csv)
    python -m src.report --raw results/tables/raw_results_quick.csv --tag quick

Outputs:
  results/tables/summary[_tag].csv  - mean / std / min / max over repeats
  results/tables/summary[_tag].md   - readable table of the main numbers
  results/figures/*[_tag].png       - plots (one set per dataset)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src import config

METRICS = ["accuracy", "balanced_accuracy", "log_loss", "ece", "mae",
           "coverage", "set_size"]

# Fixed colour + marker per model (marker shapes help colour-blind readers).
STYLE = {
    "logistic_regression": dict(color="#2a78d6", marker="o", label="Logistic regression"),
    "random_forest":       dict(color="#eb6834", marker="s", label="Random forest"),
    "gradient_boosting":   dict(color="#1baf7a", marker="^", label="Gradient boosting"),
    "tabpfn":              dict(color="#eda100", marker="D", label="TabPFN"),
}


def summarise(raw):
    """Mean, standard deviation, min and max of each metric over the repeats."""
    keys = ["dataset", "model", "n", "noise_where", "noise_level"]
    metrics = [m for m in METRICS if m in raw.columns]
    summary = raw.groupby(keys)[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary["repeats"] = raw.groupby(keys).size()
    return summary.reset_index()


def markdown_table(summary):
    """Clean-data results as 'mean ± std' text, one table per dataset."""
    lines = []
    clean = summary[summary["noise_where"] == "none"]
    for ds, part in clean.groupby("dataset"):
        cols = ["accuracy", "log_loss", "ece", "coverage", "set_size"]
        if part["mae_mean"].notna().any():
            cols.insert(1, "mae")
        lines += [f"\n### {ds} (clean data, mean ± std over repeats)\n",
                  "| model | n | " + " | ".join(cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        for _, r in part.sort_values(["n", "model"]).iterrows():
            cells = [f"{r[c + '_mean']:.3f} ± {r[c + '_std']:.3f}" for c in cols]
            lines.append(f"| {r['model']} | {r['n']} | " + " | ".join(cells) + " |")
    lines.append(f"\nConformal target coverage = {1 - config.ALPHA:.0%}.")
    return "\n".join(lines)


def _style_axes(ax, title, ylabel, xlabel):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_metric_vs_n(raw, ds, metric, ylabel, out, target=None):
    """Clean data: metric vs sample size. Line = mean, band = ±1 std, dots = repeats."""
    data = raw[(raw["dataset"] == ds) & (raw["noise_where"] == "none")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for model, s in STYLE.items():
        part = data[data["model"] == model]
        if part.empty:
            continue
        stats = part.groupby("n")[metric].agg(["mean", "std"]).reset_index()
        ax.scatter(part["n"], part[metric], color=s["color"], alpha=0.18, s=10)
        ax.plot(stats["n"], stats["mean"], color=s["color"], marker=s["marker"],
                linewidth=2, markersize=6, label=s["label"])
        ax.fill_between(stats["n"], stats["mean"] - stats["std"].fillna(0),
                        stats["mean"] + stats["std"].fillna(0),
                        color=s["color"], alpha=0.12)
    if target is not None:
        ax.axhline(target, color="#52514e", linestyle="--", linewidth=1,
                   label=f"target {target:.0%}")
    ax.set_xscale("log")
    ax.set_xticks(sorted(data["n"].unique()))
    ax.set_xticklabels(sorted(data["n"].unique()))
    _style_axes(ax, f"{ds}: {ylabel} vs training-sample size (clean data)",
                ylabel, "sample size n (log scale)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_metric_vs_noise(raw, ds, metric, ylabel, out, target=None):
    """
    Metric vs noise level at one middle sample size, two panels:
      left  - noise in train AND test ('both')
      right - noise only in test ('test_only', a distribution shift)
    """
    data = raw[raw["dataset"] == ds]
    sizes = sorted(data["n"].unique())
    n_mid = sizes[len(sizes) // 2]
    data = data[data["n"] == n_mid]
    clean = data[data["noise_where"] == "none"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, where, title in zip(axes, ["both", "test_only"],
                                ["noise in training + test", "noise in test only"]):
        for model, s in STYLE.items():
            part = pd.concat([clean[clean["model"] == model],
                              data[(data["model"] == model) &
                                   (data["noise_where"] == where)]])
            if part.empty:
                continue
            stats = part.groupby("noise_level")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(stats["noise_level"], stats["mean"], yerr=stats["std"],
                        color=s["color"], marker=s["marker"], linewidth=2,
                        markersize=6, capsize=3, label=s["label"])
        if target is not None:
            ax.axhline(target, color="#52514e", linestyle="--", linewidth=1)
        _style_axes(ax, title, ylabel, "noise level")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{ds}: {ylabel} vs noise (n = {n_mid}, bars = ±1 std)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarise results and draw plots.")
    parser.add_argument("--raw", default=str(config.TABLES_DIR / "raw_results.csv"))
    parser.add_argument("--tag", default="", help="suffix for output file names")
    args = parser.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    raw = pd.read_csv(args.raw)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = summarise(raw)
    summary.to_csv(config.TABLES_DIR / f"summary{tag}.csv", index=False)
    (config.TABLES_DIR / f"summary{tag}.md").write_text(markdown_table(summary),
                                                        encoding="utf-8")

    target = 1 - config.ALPHA
    fig_dir = config.FIGURES_DIR
    for ds in raw["dataset"].unique():
        plot_metric_vs_n(raw, ds, "accuracy", "accuracy", fig_dir / f"{ds}_accuracy_vs_n{tag}.png")
        plot_metric_vs_n(raw, ds, "ece", "calibration error (ECE)", fig_dir / f"{ds}_ece_vs_n{tag}.png")
        plot_metric_vs_n(raw, ds, "coverage", "conformal coverage",
                         fig_dir / f"{ds}_coverage_vs_n{tag}.png", target=target)
        plot_metric_vs_n(raw, ds, "set_size", "average set size", fig_dir / f"{ds}_set_size_vs_n{tag}.png")
        plot_metric_vs_noise(raw, ds, "accuracy", "accuracy", fig_dir / f"{ds}_accuracy_vs_noise{tag}.png")
        plot_metric_vs_noise(raw, ds, "coverage", "conformal coverage",
                             fig_dir / f"{ds}_coverage_vs_noise{tag}.png", target=target)

    print(f"Wrote summary tables to {config.TABLES_DIR}")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
