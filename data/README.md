# Data

Raw data files are **not** stored in git. Download them yourself and place them in this folder.

## 1. Wine Quality (main dataset)

- Page: https://archive.ics.uci.edu/dataset/186/wine+quality
- Click **Download**, unzip, and put these two files here:
  - `winequality-red.csv`
  - `winequality-white.csv`
- Note: these files use **semicolons** (`;`) between columns, not commas.

## 2. Adult Income (second dataset)

- Page: https://archive.ics.uci.edu/dataset/2/adult
- Click **Download**, unzip, and put these two files here:
  - `adult.data` (training part)
  - `adult.test` (test part — its first line is a comment and its labels end with a full stop, e.g. `>50K.`; the loader will handle this)

## Expected folder after downloading

```
data/
  README.md
  winequality-red.csv
  winequality-white.csv
  adult.data
  adult.test
```

Both datasets are licensed CC BY 4.0 (free to use with credit to the UCI Machine Learning Repository).
