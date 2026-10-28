# Small, Noisy Tabular Data: A Benchmark of TabPFN vs Classic Models

**Status:** code written and tested on the classic models; full experiment not yet run. No results yet.

## What this project does

This is a **benchmark** (a fair, repeatable comparison), not a new method.
It asks two questions:

1. How well do four models work when the training data is **small** and **noisy**?
   - Logistic regression (a simple linear model)
   - Random forest (many decision trees that vote)
   - Gradient boosting (trees built one after another, each fixing the last one's mistakes)
   - TabPFN (a neural network pre-trained on millions of synthetic tables, built for small tables)
2. Can their **uncertainty estimates** be trusted? We test this with
   **conformal prediction** (a method that turns any model's output into a *set*
   of possible answers that should contain the true answer a chosen share of the
   time, e.g. 90%).

## Datasets (real, public)

| Dataset | Rows | Task |
|---|---|---|
| Wine Quality (UCI) | ~6,500 | Predict quality score (ordered classes) — main dataset |
| Adult Income (UCI) | ~48,800 | Predict income above/below $50K — second dataset |

Download instructions are in [`data/README.md`](data/README.md). Raw data is **not**
stored in this repo.

## Folder layout

```
data/              raw datasets (you download these; ignored by git)
src/               project code
tests/             quick checks that code runs (synthetic data, TEST-ONLY)
results/tables/    result tables (CSV)
results/figures/   plots
notes/             decisions, setup log, write-up drafts
colab/             notebook for running the full experiment on a Colab GPU
```

## Setup on Windows (laptop)

This project uses **Python 3.12** (TabPFN's dependency PyTorch does not yet have
stable Windows builds for Python 3.15).

Open Command Prompt in the project folder and run:

```
py -V:3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Line 1 creates a **virtual environment** (a private copy of Python just for this project) in `.venv`.
- Line 2 **activates** it (your prompt will start with `(.venv)`). Do this every time you open a new terminal.
- Lines 3–4 install the packages listed in `requirements.txt`.

## TabPFN access (one-time)

TabPFN's model weights are hosted on Hugging Face (a site for sharing AI models) behind
a terms-of-use page:

1. Make a free account at https://huggingface.co and accept the terms at
   https://huggingface.co/Prior-Labs/tabpfn_3_5
2. Create a **read** token (a personal access key) under Settings → Access Tokens.
3. Laptop: with the virtual environment active, run `hf auth login` and paste the token.
   Colab: store it as a Colab secret named `HF_TOKEN` (see the notebook).

Never paste the token into code or commit it to git.

## How to run

All commands from the project folder, with the virtual environment active.

```
python -m pytest tests -q                      # quick checks (synthetic TEST-ONLY data)
python -m src.experiment --quick               # tiny real-data run (~1 min without TabPFN)
python -m src.experiment --quick --models logistic_regression random_forest gradient_boosting
python -m src.report --raw results/tables/raw_results_quick.csv --tag quick
```

The **full** experiment (5 sample sizes x 20 repeats x 4 noise levels x 4 models x
2 datasets) should be run on Colab: open `colab/run_experiments.ipynb` in Colab and
run the cells in order.

## Code map

| File | What it does |
|---|---|
| `src/config.py` | every experiment setting (sizes, repeats, noise levels, alpha) |
| `src/data.py` | loads and cleans Wine Quality and Adult Income |
| `src/noise.py` | adds controlled noise to inputs |
| `src/models.py` | the four models, built the same way |
| `src/metrics.py` | accuracy, balanced accuracy, log loss, calibration error, MAE |
| `src/conformal.py` | split conformal prediction sets, coverage, set size |
| `src/experiment.py` | main loop; saves results after every model, resumes after a crash |
| `src/report.py` | summary tables (CSV + Markdown) and plots |
| `tests/test_pipeline.py` | checks the pieces work (synthetic, TEST-ONLY data) |

## TabPFN licence note

TabPFN's **code** is Apache-2.0, but the default **model weights** (TabPFN-3.5) are under a
**non-commercial licence**. This project is non-commercial research, which the licence
allows. Details: `notes/decisions.md`.
