# Predictions (written BEFORE looking at the full results)

**Written:** 2026-10-28, about 19:05 IST, while the full Colab run was still in progress
(Wine at n = 250). No full-run results had been looked at.
**Honest disclosure:** the 2-repeat quick check (classic models only) had already been
seen. It showed coverage of roughly 0.81–0.97, too few repeats to judge the spread.

Why this file exists: we state what theory predicts first, then check it. If the results
disagree, that is reported, not hidden.

---

## 1. Terms used

- **Coverage** (how often the answer set contains the true class). Target: 90%.
- **alpha** (the allowed miss rate) = 0.1, so the target is 1 - alpha = 90%.
- **Calibration rows** (rows held back only to measure how wrong the model usually is):
  n_cal = round(0.25 x n). Python rounds 62.5 down to 62.
- **Score** (how surprised the model was by the true answer) = 1 - the probability the
  model gave the true class.
- **Cut-off q** (the surprise level above which a class is left out of the answer set) =
  the k-th smallest calibration score, where k = ceil((n_cal + 1) x 0.9)
  (ceil = round up).
- A test row is **covered** if its score is at or below q.
- **Exchangeability** (calibration and test rows are drawn the same way, so none is special).
- **Beta distribution** (a curve describing a random percentage between 0% and 100%; its two
  numbers set where it peaks and how wide it is).
- **Standard deviation, sd** (the typical distance of one run from the average).

## 2. The theory (split conformal prediction)

**Result 1 (average coverage).** Under exchangeability, a new test row's score is equally
likely to land in any of the n_cal + 1 gaps between and around the sorted calibration
scores. It is below the cut-off in k of them, so average coverage = k / (n_cal + 1),
which is always at least 90%. This is the "guarantee".

**Result 2 (run-to-run spread).** For one particular calibration set, the coverage equals
the share of all possible scores that fall below q. q is an **order statistic** (the k-th
smallest value of a random sample), and order statistics of percentages follow Beta curves:

    coverage of one run ~ Beta(k, n_cal + 1 - k)

This holds for ANY model and ANY data, given exchangeability and no **ties** (several
rows with exactly the same score). Because it holds for every trained model, it still
holds when the training rows change between repeats, as they do in our experiment.

Sources: Vovk (2012), "Conditional validity of inductive conformal predictors";
Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and
Distribution-Free Uncertainty Quantification".

## 3. Numerical predictions (clean data "none", and noise in train + calibration + test "both")

Computed with `scipy.stats.beta`.

| n | n_cal | k | Beta curve | mean coverage | sd | middle 90% of runs | share of runs below 85% |
|---|---|---|---|---|---|---|---|
| 50 | 12 | 12 | Beta(12, 1) | 0.923 | 0.071 | 0.779–0.996 | 14% |
| 100 | 25 | 24 | Beta(24, 2) | 0.923 | 0.051 | 0.824–0.986 | 9% |
| 250 | 62 | 57 | Beta(57, 6) | 0.905 | 0.037 | 0.838–0.957 | 8% |
| 500 | 125 | 114 | Beta(114, 12) | 0.905 | 0.026 | 0.859–0.944 | 3% |
| 1000 | 250 | 226 | Beta(226, 25) | 0.900 | 0.019 | 0.868–0.930 | 1% |

Extra wobble because the test set has 2,000 rows rather than infinitely many:
about sqrt(0.09 / 2000) = +/-0.007 per run.

## 4. Specific, checkable predictions

- **P1 (average holds).** For "none" and "both", the mean coverage over 20 repeats is
  within the theoretical mean +/- about 2 x sd / sqrt(20), for every model and dataset.
- **P2 (spread matches).** The sd of coverage across the 20 repeats is close to the Beta sd
  above. With only 20 repeats the measured sd is itself only accurate to about +/-16%,
  so "close" means within roughly that margin.
- **P3 (the model doesn't matter for coverage).** All four models show similar coverage
  mean and spread. They differ in **set size** (how many classes each answer set contains;
  smaller = more useful): better models give smaller sets.
- **P4 (noise in "both" is still fine).** The same noise in training, calibration and test
  keeps exchangeability, so P1–P2 still hold. Sets get larger because models become
  less certain.
- **P5 (test-only noise breaks the guarantee).** With clean calibration but noisy test data,
  exchangeability fails. Prediction: coverage falls below the theory, and falls more as
  the noise level rises. The size of the drop is not predicted (no theory covers it).
- **P6 (ties).** Random forest probabilities are coarse (vote shares out of 300 trees), so
  scores tie often. Prediction: random forest coverage comes out slightly ABOVE theory.
- **P7 (usefulness).** At n = 50, sets are large (often 3+ of the 5 wine classes, and often
  both classes for Adult), so the guarantee is close to useless there. Set size shrinks
  as n grows.
- **P8 (TabPFN).** Open question with no theory: does TabPFN give the smallest sets at a
  given n, and does it lose less coverage under test-only noise? Our guess: smaller sets,
  yes. Robustness: unknown.

## 5. How these will be checked

`src/coverage_theory.py` (to be written after the run) will compare the measured coverage
mean, sd and share below 85% against the table above, for every model, dataset, n and
noise setting, and plot the measured runs against the predicted Beta curve.
