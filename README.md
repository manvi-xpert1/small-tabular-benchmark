# Small, Noisy Tabular Data: A Benchmark of TabPFN vs Classic Models

**Status:** experiment runs in progress (classic models locally, TabPFN on Colab). Results and write-up to follow.

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

Needs: Python 3.12 (`py install 3.12`), git, and the Microsoft Visual C++
Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe), a system component
PyTorch needs on Windows.

In PowerShell or Command Prompt, from the project folder:

```
py -V:3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

- Line 1 creates a **virtual environment** (a private copy of Python just for this project) in `.venv`.
- Line 2 installs the packages listed in `requirements.txt` into it (about 1 GB; the last
  "Installing collected packages" step can sit silent for 10+ minutes on Windows).
- We call `.venv\Scripts\python` directly instead of "activating" the environment,
  because PowerShell blocks the activation script by default.

## TabPFN access (one-time)

TabPFN needs **two** permissions before it can download its model weights:

1. **Hugging Face** (the site hosting the files): make a free account, accept the terms at
   https://huggingface.co/Prior-Labs/tabpfn_3_5, and create a **read** token under
   Settings → Access Tokens. Laptop: run `.venv\Scripts\hf auth login`.
2. **Prior Labs** (TabPFN's makers): accept the TabPFN-3.5 licence at
   https://platform.priorlabs.ai/account/licenses and create an **API key** at
   https://platform.priorlabs.ai/account/api-keys.

On Windows, TabPFN's automatic browser login crashes (`WinError 10038`), so give it the
key directly. In PowerShell, for the current window only:

```
$env:TABPFN_TOKEN = "your-api-key"
$env:TABPFN_NO_BROWSER = "1"
```

After one successful run, TabPFN caches the key in `%USERPROFILE%\.cache\tabpfn\`
(outside this project), so later runs don't need it.
Colab: store both keys as Colab secrets `HF_TOKEN` and `TABPFN_TOKEN` (see the notebook).

Never paste keys into code or commit them to git.

## How to run

All commands from the project folder.

```
.venv\Scripts\python -m pytest tests -q        # quick checks (synthetic TEST-ONLY data)
.venv\Scripts\python -m src.experiment --models logistic_regression random_forest gradient_boosting
.venv\Scripts\python -m src.report               # summary tables + plots
.venv\Scripts\python -m src.coverage_theory      # measured coverage vs theory
```

- The three **classic models** run on a normal computer (the full run takes under an hour).
  Results: `results/tables/raw_results.csv`.
- **TabPFN** needs a GPU (graphics card), so it runs on Google Colab:
  open `colab/run_experiments.ipynb` in Colab and run the cells top to bottom.
  Results: `results/tables/raw_results_tabpfn.csv` (download it into `results/tables/`).
- `src.coverage_theory` combines both files automatically.
- Optional, to run a smaller design: `--sizes 50 100 250 --repeats 10 --noise-levels 0 0.25`.
  Seeds depend only on (dataset, n, repeat, noise level), so a smaller run gives exactly
  the same numbers as the matching part of the full run.

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
| `src/coverage_theory.py` | compares measured conformal coverage with the Beta-distribution theory (see `notes/predictions.md`) |
| `tests/test_pipeline.py` | checks the pieces work (synthetic, TEST-ONLY data) |

## TabPFN licence note

TabPFN's **code** is Apache-2.0, but the default **model weights** (TabPFN-3.5) are under a
**non-commercial licence**. This project is non-commercial research, which the licence
allows. Details: `notes/decisions.md`.
