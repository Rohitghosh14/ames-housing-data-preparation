# Data-Quality Report: Ames Housing

Source: Kaggle "House Prices - Advanced Regression Techniques", `train.csv` (1,460 houses, 79 features plus `Id` and `SalePrice`).

## 1. Split
- 80/20 train/test split, `random_state=42` (1,168 / 292 houses), done before any cleaning.
- Every learned step (imputers, encoders, transformers, scalers) is fitted on the train set only.

## 2. Missing values
| Problem | Decision |
|---|---|
| `PoolQC` (99.5%), `MiscFeature` (96.1%), `Alley` (93.7%) mostly empty | Dropped. Empty means "feature absent". |
| 12 columns where empty means "no such feature" (garage, basement, fireplace, fence, veneer type) | Filled with `"None"`. |
| `MasVnrArea` (6 rows) | Train median. |
| `Electrical` (1 row) | Most frequent value, `SBrkr`. |
| `GarageYrBlt` (64 rows, all no-garage houses) | Flag column `GarageYrBlt_missing` plus train median. |
| `LotFrontage` (217 rows, 18.6%) | KNN imputer (5 neighbours) using `LotArea`, `1stFlrSF`, `GrLivArea`. |

Result: 0 missing values in train and test.

## 3. Outliers
- IQR on `GrLivArea` flagged 23 train houses (upper limit 2,761 sq ft).
- Removed only the 2 houses with area above 4,000 sq ft but price below 300,000. The other 21 are large, high-priced houses and are kept.
- Train set: 1,166 houses. The test set is not filtered.

## 4. Encoding
- 9 quality columns: ordinal, `None < Po < Fa < TA < Gd < Ex`.
- 31 other text columns: one-hot (unseen test categories are ignored).
- Result: 251 columns, all numeric.

## 5. Skew and scaling
- `SalePrice` skew is 1.74. `log1p` brings it to 0.13 (use `np.expm1` to convert predictions back to dollars). Saved targets stay in dollars.
- Yeo-Johnson on 16 continuous columns with |skew| > 0.75. Zero-heavy columns (`PoolArea`, `MiscVal`, `3SsnPorch`, `LowQualFinSF`) stay lopsided because most houses have 0.
- StandardScaler on 45 numeric columns. 0/1 columns are not scaled.

## 6. Constructed features
`TotalSF`, `HouseAge`, `TotalBath` (scaled) and `HasPool` (0/1), built from the raw values.

## 7. Final output
`data/processed/`: `X_train.csv` (1,166 x 255), `X_test.csv` (292 x 255), `y_train.csv`, `y_test.csv`. No missing values.