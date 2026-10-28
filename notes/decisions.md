# Decisions log

A running record of choices made and why, so they can be explained later.

## 2026-10-28 — Phase 0 (setup)

- **Python 3.12.10** for this project, in a virtual environment (`.venv`).
  The laptop also has Python 3.15.2, but PyTorch's last full release (2.13) had no
  Windows builds for 3.15, so 3.12 is the safe choice. Both versions coexist fine.
- **Git 2.56.0** installed on the laptop.
- **Package versions** pinned in `requirements.txt`; checked that they install together
  on Windows/Python 3.12 and on Linux (Colab).
- **Where things run:**
  - GitHub repo = home of the project (code, results, write-up).
  - Laptop = write code, quick small checks.
  - Google Colab (free GPU) = full experiment runs, because TabPFN is a neural network
    and is slow on a CPU. TabPFN's docs say CPU is only practical up to ~5,000 training rows.
- **TabPFN licence and access:** code is Apache-2.0; default model weights (TabPFN-3.5)
  are non-commercial and **gated**: you need Hugging Face access (read token) AND a
  Prior Labs API key with the TabPFN-3.5 licence accepted. Fine for this non-commercial research. The token must never be committed.

## 2026-10-28 — Data decisions

- **Wine: red + white combined** (6,497 wines) with a `colour` column, for a bigger
  reference pool.
- **Wine: rare scores merged**: 3 -> 4 and 9 -> 8, leaving 5 ordered classes
  (<=4, 5, 6, 7, >=8). Counts: 246 / 2,138 / 2,836 / 1,079 / 198. Small samples would
  otherwise often miss classes 3 and 9 entirely. Cost: "very bad" and "bad" are no
  longer separated.
- **Adult: adult.data + adult.test combined** into one pool (48,842 people).
  `?` kept as its own "Unknown" category (not dropped). `fnlwgt` dropped (a census
  sampling weight, not a property of the person). `education` text dropped because it
  duplicates `education_num`. Classes: 37,155 <=50K vs 11,687 >50K (about 76% / 24%).

## 2026-10-28 — Experiment design (in `src/config.py`)

- Fixed reference test set of 2,000 rows per dataset; samples are drawn from the rest.
- Sample sizes n = 50, 100, 250, 500, 1000; 20 repeats each.
- Each sample: 75% trains the model, 25% calibrates conformal prediction. So at n = 50
  the model trains on only 38 rows and calibrates on 12. Point metrics are therefore
  for a model trained on 0.75 x n rows; this must be stated in the write-up.
- Noise levels 0, 0.1, 0.25, 0.5. Numeric: Gaussian jitter = level x feature std.
  Categorical: value replaced by a random category with probability = level.
- Two noise settings: "both" (noise in train, calibration and test: same distribution,
  so the conformal guarantee should hold) and "test_only" (clean training/calibration,
  noisy test: a distribution shift, so the guarantee can fail).
- Conformal target: 90% coverage (alpha = 0.1), using the "1 - probability of true
  class" score.
- All models use default settings (no tuning), for fairness and simplicity.
- Every model sees the same samples and the same noise in each repeat.

## 2026-10-28 — Setup problems hit (and fixes)

- PowerShell blocks `.venv\Scripts\activate` -> call `.venv\Scripts\python` directly.
- `pip` without the venv path used global Python 3.15 -> no PyTorch build -> install failed.
- PyTorch `WinError 126` (shm.dll) -> missing Microsoft Visual C++ Redistributable; installed it.
- TabPFN needs a Prior Labs API key as well as Hugging Face access. Its browser login
  crashes on Windows (`WinError 10038`: Python's `select` only works on network sockets
  on Windows) -> set `TABPFN_TOKEN` and `TABPFN_NO_BROWSER=1` instead.

## Early observations from the quick check (NOT results)

These come from 2 repeats only, without TabPFN. They are just sanity checks:
- Adult, n = 50: gradient boosting scored 0.762 accuracy in both repeats, which is about
  the share of the majority class. It is likely just predicting "<=50K" for everyone at
  that size. Balanced accuracy will show this clearly.
- Wine is hard: accuracy is only around 0.43-0.49 at n <= 250 for the classic models.

## 2026-10-28 — Runs and checks

- Classic models: full run on a cloud workspace (CPU). TabPFN: full run on Colab T4 GPU
  (`raw_results_tabpfn.csv`). Seeds make both identical to what any machine would produce.
- Follow-up check: logistic regression, n = 250 and 1000, **200 repeats**
  (`results/tables/check_lr_200_repeats.csv`). Spread matches theory (ratio 0.86-0.99), so the
  ~20% excess seen with 20 repeats was sampling noise. Lesson: repeats are correlated across models
  (same calibration rows), so agreement between models is not independent evidence.

## Open questions

- Colab: upload data files manually (current notebook) or download from UCI? — manual (decided)
- Why is TabPFN robust to test-time noise on Wine but not on Adult? Needs more datasets.
