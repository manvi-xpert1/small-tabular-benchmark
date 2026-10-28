"""
All experiment settings in one place.

Change numbers here (not inside the other files) so every run is easy to describe
and repeat.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders (worked out relative to this file, so it runs from anywhere)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TABLES_DIR = PROJECT_DIR / "results" / "tables"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"

# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------

# Size of the fixed "reference" test set taken from each full dataset.
# Every small-sample model is judged on this same set, so results are comparable.
# Capped at 2,000 rows to keep TabPFN quick.
TEST_SIZE = 2000

# Small training-sample sizes to try (number of rows drawn from the pool).
SAMPLE_SIZES = [50, 100, 250, 500, 1000]

# How many times each sample size is redrawn with a new random seed.
# The spread across repeats shows how much results "swing" between samples.
N_REPEATS = 20

# Share of each small sample held back for conformal calibration
# (calibration = measuring how wrong the model usually is, to size the answer sets).
CALIBRATION_SHARE = 0.25

# Noise levels.
#  - Numeric features: Gaussian noise (random jitter) with standard deviation
#    = level x that feature's standard deviation. 0.5 = jitter half as big as the
#    feature's natural spread.
#  - Categorical features: each value is replaced by a random category with
#    probability = level.
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5]

# Conformal prediction: target miss rate (alpha). 0.1 means "the answer set should
# contain the true class at least 90% of the time".
ALPHA = 0.1

# Number of bins for expected calibration error (ECE).
ECE_BINS = 10

# Models compared (names used in results tables).
MODEL_NAMES = ["logistic_regression", "random_forest", "gradient_boosting", "tabpfn"]

# Master random seed. Every other seed is derived from this, so the whole
# experiment is repeatable.
BASE_SEED = 42

# ---------------------------------------------------------------------------
# A much smaller "quick" setting for checking the code runs end to end.
# ---------------------------------------------------------------------------
QUICK_SAMPLE_SIZES = [50, 250]
QUICK_N_REPEATS = 2
QUICK_NOISE_LEVELS = [0.0, 0.25]
