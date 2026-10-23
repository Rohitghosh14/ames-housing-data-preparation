# Ames Housing: Data Preparation

End-to-end data preparation on the Ames Housing dataset: EDA, missing values, outliers, encoding, Yeo-Johnson transforms, scaling and feature construction, with a leakage-free train/test split.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![pandas](https://img.shields.io/badge/pandas-data%20prep-informational) ![scikit-learn](https://img.shields.io/badge/scikit--learn-preprocessing-orange)

## Table of contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Pipeline at a glance](#pipeline-at-a-glance)
4. [Step-by-step](#step-by-step)
5. [Key results](#key-results)
6. [Repository structure](#repository-structure)
7. [How to run](#how-to-run)
8. [Tech stack](#tech-stack)
9. [What I learned](#what-i-learned)
10. [Limitations and next steps](#limitations-and-next-steps)

## Overview

Raw data is rarely ready for a model. This project takes the Ames Housing dataset (house sale prices in Ames, Iowa) from a raw CSV to a clean, fully numeric dataset, and documents every decision along the way.

The goal is not to predict prices. The goal is to prepare the data **correctly**:
- the train/test split is made **before** any cleaning, so no information leaks from the test set;
- every step that learns from data (imputers, encoders, transformers, scalers) is fitted on the **train set only**;
- every decision (what to drop, fill, remove or transform) is written down with its reason in the [data-quality report](docs/DATA_QUALITY_REPORT.md).

## Dataset

- **Source:** Kaggle, [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), file `train.csv`.
- **Size:** 1,460 houses, 79 features, plus `Id` and the target `SalePrice`.
- **Types:** 36 numeric and 43 text columns.
- The raw file is **not included** in this repository. Download `train.csv` from Kaggle and place it in `data/raw/`.

## Pipeline at a glance

```
train.csv (1,460 houses)
   |
   |-- EDA: shape, types, missing values, skew, correlations
   |
   |-- Train/test split (80/20, random_state=42)   <-- before any cleaning
   |       |-- train: 1,168 houses
   |       |-- test:    292 houses
   |
   |-- Missing values: drop, fill with "None", median, mode, flag, KNN
   |-- Outliers: IQR check, then remove 2 odd houses (train only)
   |-- Encoding: ordinal (9 quality columns), one-hot (31 text columns)
   |-- Skew: Yeo-Johnson on 16 columns, log1p on the target
   |-- Scaling: StandardScaler on 45 numeric columns
   |-- Feature construction: TotalSF, HouseAge, TotalBath, HasPool
   |
   +--> data/processed/  (255 numeric columns, 0 missing values)
```

## Step-by-step

### 1. Exploratory data analysis
- Checked shape, column types, `describe()` for numbers and text, and missing-value counts.
- `SalePrice` is right-skewed (skew 1.88 on the full data), with a mean of about 181k and a median of about 163k.
- Strongest correlations with price: `OverallQual` (0.79), `GrLivArea` (0.71), `GarageCars` (0.64), `GarageArea` (0.62), `TotalBsmtSF` (0.61).
- The scatter plot of `GrLivArea` against `SalePrice` showed a few very large houses with low prices.

### 2. Train/test split
- 80/20 split with `random_state=42`: 1,168 train houses and 292 test houses.
- `SalePrice` and `Id` are separated from the features first.

### 3. Missing values
In Ames, most empty cells do not mean "lost data". They mean the feature does not exist (no pool, no garage). Each case got its own treatment:

| Problem | Decision | Why |
|---|---|---|
| `PoolQC` (99.5%), `MiscFeature` (96.1%), `Alley` (93.7%) | Dropped | Almost every house has the same value, "None". |
| 12 columns (garage, basement, fireplace, fence, veneer type) | Filled with `"None"` | The gap means "no such feature". A fixed value cannot leak. |
| `MasVnrArea` (6 rows) | Train median | Few gaps. The median resists outliers. |
| `Electrical` (1 row) | Most frequent value (`SBrkr`) | A single truly missing value. |
| `GarageYrBlt` (64 rows) | Flag column plus train median | All 64 are no-garage houses (verified with a crosstab). The flag tells the model the year is not real. |
| `LotFrontage` (217 rows, 18.6%) | KNN imputer (5 neighbours) | Truly missing, and related to lot size and floor area. |

Result: 0 missing values in both train and test.

### 4. Outliers
- IQR rule on `GrLivArea` flagged 23 train houses (upper limit 2,761 sq ft).
- Compared each flagged house's size with its price before deciding.
- **Removed 2 houses** with more than 4,000 sq ft but a price below 300,000. They break the size-price pattern.
- **Kept the other 21.** They are large houses with high prices, which is real information.
- Rows were removed from the train set only. The test set stays untouched.

### 5. Encoding
- **Ordinal encoding** for 9 quality columns with a natural order: `None < Po < Fa < TA < Gd < Ex` (0 to 5).
- **One-hot encoding** for the 31 columns with no order (for example `Neighborhood` with 25 values). Names seen only in the test set are ignored instead of crashing.
- Result: 251 columns, all numeric.

### 6. Skew and transforms
- **Target:** `log1p` reduced the skew of `SalePrice` from 1.74 to 0.13 on the train set. Convert predictions back with `np.expm1`.
- **Features:** Yeo-Johnson (works for left skew, right skew and zeros) on the 16 continuous columns with |skew| > 0.75. For example, `LotArea` went from 12.35 to 0.01 and `GrLivArea` from 0.98 to about 0.
- Zero-heavy columns (`PoolArea`, `MiscVal`, `3SsnPorch`, `LowQualFinSF`) stay lopsided, because no transform can turn a large spike of zeros into a bell shape.

### 7. Scaling
- `StandardScaler` (mean 0, spread 1) on 45 numeric columns, fitted on train and applied to both.
- 0/1 columns (one-hot and flags) are not scaled.

### 8. Feature construction
Built from the raw values, because the formulas need real units:

| Feature | Formula |
|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` |
| `HouseAge` | `YrSold - YearBuilt` |
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` |
| `HasPool` | 1 if `PoolArea > 0`, else 0 |

`TotalSF`, `HouseAge` and `TotalBath` were scaled with a scaler fitted on train. `HasPool` stays 0/1.

## Key results

| | Value |
|---|---|
| Raw dataset | 1,460 houses, 79 features |
| Train / test | 1,166 / 292 houses (1,168 before outlier removal) |
| Final columns | 255, all numeric |
| Missing values | 0 in train and 0 in test |
| `SalePrice` skew (train) | 1.74 before `log1p`, 0.13 after |

## Repository structure

```
.
|-- data/
|   `-- processed/          # cleaned train/test features and targets
|-- docs/
|   `-- DATA_QUALITY_REPORT.md
|-- notebook.ipynb          # the full analysis, run top to bottom
|-- pyproject.toml          # project settings and packages
|-- uv.lock                 # exact package versions
|-- .python-version         # Python 3.12
`-- README.md
```

`data/raw/` is not tracked. Download `train.csv` from Kaggle and place it there.

**Files in `data/processed/`:**
- `X_train.csv` (1,166 x 255) and `X_test.csv` (292 x 255): cleaned features.
- `y_train.csv` and `y_test.csv`: `SalePrice` in dollars.
- `y_train_log.csv` and `y_test_log.csv`: `SalePrice` as `log1p` values.

## How to run

You need [uv](https://docs.astral.sh/uv/) installed.

```bash
# 1. Get the code
git clone https://github.com/Rohitghosh14/ames-housing-data-preparation.git
cd ames-housing-data-preparation

# 2. Create the environment (Python 3.12) and install packages
uv sync

# 3. Download train.csv from Kaggle and put it here:
#    data/raw/train.csv
```

Then open `notebook.ipynb`, select the `.venv` kernel and choose **Run All**.

## Tech stack

| Tool | Used for |
|---|---|
| Python 3.12 | Language |
| pandas, NumPy | Tables and math |
| matplotlib, seaborn | Charts |
| scikit-learn | Split, imputers, encoders, Yeo-Johnson, scaler |
| uv | Environment and package management |
| Git and GitHub | Version control |

## What I learned

- **Split first.** Cleaning before splitting lets test information leak into training, which makes results look better than they are.
- **Ask why a value is missing.** In this dataset, most empty cells mean "none", and that decides the fix.
- **Look at an outlier before removing it.** A large house with a high price is real. A large house with a low price is the one that breaks the pattern.
- **Match the transform to the shape.** Yeo-Johnson handles both skew directions, and a spike of zeros cannot be fixed by any transform.
- **Keep every decision written down.** The data-quality report lets anyone retrace how the dataset was made.

## Limitations and next steps

- This project prepares the data. It does not train or evaluate a model.
- Zero-heavy columns remain skewed after the transform.
- The steps run as separate notebook cells. A follow-up project will chain them into a single scikit-learn `Pipeline` with cross-validation and model comparison.

## Author

Rohit Ghosh ([@Rohitghosh14](https://github.com/Rohitghosh14))