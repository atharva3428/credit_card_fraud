# Credit Card Fraud Detection

A data analysis pipeline for the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Dataset
- **Source:** Kaggle — ULB Machine Learning Group
- **Records:** 284,807 transactions over 2 days
- **Features:** 30 (Time, V1–V28 PCA components, Amount) + 1 target (Class)
- **Class imbalance:** 0.17% fraud (492 out of 284,807)

> **Note:** Raw and processed CSV files are excluded from this repository (`.gitignore`) due to file size. Download the dataset from Kaggle and place `creditcard.csv` in the project root.

---

## Pipeline

### 1. `data_profiling.py` — Data Profiling
- Shape, dtypes, missing values, duplicates
- Class distribution & imbalance ratio
- Descriptive statistics for all features
- Skewness, kurtosis, outlier detection
- Feature correlation with target

### 2. `data_cleaning.py` — Data Cleaning
- Drop 1,081 duplicate rows
- Flag zero-amount transactions (`is_zero_amount`)
- IQR-based outlier capping on `Amount` (upper fence: $185.38)
- Log transformation of Amount → `Amount_log` (skew: 16.98 → -0.18)
- StandardScaler on `Time` → `Time_scaled`

### 3. `data_wrangling.py` — Data Wrangling & Feature Engineering
- Time features: `Time_hour`, `is_daytime`, `Day_period`
- Amount bins (zero / micro / small / medium / large / xlarge / xxlarge) — one-hot encoded
- V-feature aggregates: `V_mean`, `V_std`, `V_min`, `V_max`, `V_range`, `V_pos_ct`, `V_neg_ct`
- Interaction terms: `V14_x_V17`, `V14_x_V12`
- Stratified 80:20 train/test split
- SMOTE oversampling on training set (fraud: 378 → 226,602)

---

## Output Files (not tracked in git)

| File | Description |
|------|-------------|
| `creditcard_cleaned.csv` | After deduplication, capping, log-transform |
| `creditcard_wrangled.csv` | Full feature-engineered dataset (50 features) |
| `train_smote.csv` | Balanced training set after SMOTE (453,204 rows) |
| `test.csv` | Untouched test set (56,746 rows) |

---

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

---

## Usage

```bash
python data_profiling.py
python data_cleaning.py
python data_wrangling.py
```
