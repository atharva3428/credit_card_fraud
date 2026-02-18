# encoding: utf-8
"""
==========================================================
STEP 2 - DATA WRANGLING
==========================================================
Tasks:
  1. Load cleaned dataset
  2. Feature engineering  (time-of-day, amount bins, V-aggregates)
  3. Drop raw / redundant columns
  4. Final feature matrix (X) and target (y)
  5. Train / Test split  (stratified, 80:20)
  6. Handle class imbalance  (SMOTE on training set only)
  7. Save wrangled splits
==========================================================
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional SMOTE - graceful skip if imbalanced-learn not installed
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

CSV_IN   = "D:/CREDIT_CARD_FRAUD/creditcard_cleaned.csv"
OUT_DIR  = "D:/CREDIT_CARD_FRAUD/"

print("=" * 65)
print("STEP 2 - DATA WRANGLING")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
print(f"\nLoaded cleaned dataset : {df.shape}")
print(f"Columns available      : {df.columns.tolist()}\n")

# ─────────────────────────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
print("[1] FEATURE ENGINEERING")

# 1a. Time-of-day (seconds in a 24-hour cycle)
SECONDS_IN_DAY = 86400
df["Time_of_day_s"]  = df["Time"] % SECONDS_IN_DAY          # 0-86399
df["Time_hour"]      = (df["Time_of_day_s"] // 3600).astype(int)  # 0-23

# Peak hour flag  (8am-8pm = daytime)
df["is_daytime"] = ((df["Time_hour"] >= 8) & (df["Time_hour"] < 20)).astype(int)
print(f"  1a. Time_of_day_s, Time_hour, is_daytime  created.")
print(f"      Daytime txns : {df['is_daytime'].sum():,}  |  Night txns : {(df['is_daytime']==0).sum():,}")

# 1b. Which 2-day period
df["Day_period"] = (df["Time"] // SECONDS_IN_DAY).astype(int)  # 0 or 1
print(f"  1b. Day_period (0=day1, 1=day2) created.")
print(f"      Day 0 : {(df['Day_period']==0).sum():,}  |  Day 1 : {(df['Day_period']==1).sum():,}")

# 1c. Amount bins (categorical buckets)
bins   = [-0.01, 0, 10, 50, 100, 500, 1000, np.inf]
labels = ["zero","micro","small","medium","large","xlarge","xxlarge"]
df["Amount_bin"] = pd.cut(df["Amount"], bins=bins, labels=labels)
print(f"\n  1c. Amount_bin distribution:")
bin_dist = df.groupby("Amount_bin", observed=True)["Class"].agg(["count","sum"])
bin_dist.columns = ["total","fraud"]
bin_dist["fraud_%"] = (bin_dist["fraud"] / bin_dist["total"] * 100).round(3)
print(bin_dist.to_string())

# One-hot encode Amount_bin
df = pd.get_dummies(df, columns=["Amount_bin"], drop_first=False)

# 1d. V-feature aggregates (row-level stats across V1-V28)
v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
df["V_mean"]   = df[v_cols].mean(axis=1)
df["V_std"]    = df[v_cols].std(axis=1)
df["V_min"]    = df[v_cols].min(axis=1)
df["V_max"]    = df[v_cols].max(axis=1)
df["V_range"]  = df["V_max"] - df["V_min"]
df["V_pos_ct"] = (df[v_cols] > 0).sum(axis=1)   # count of positive V features
df["V_neg_ct"] = (df[v_cols] < 0).sum(axis=1)   # count of negative V features
print(f"\n  1d. V-aggregate features created: V_mean, V_std, V_min, V_max, V_range, V_pos_ct, V_neg_ct")

# 1e. High-correlation interaction (V14 * V17 - top correlated with fraud)
df["V14_x_V17"] = df["V14"] * df["V17"]
df["V14_x_V12"] = df["V14"] * df["V12"]
print(f"  1e. Interaction terms: V14_x_V17, V14_x_V12  created.")

print(f"\n  Shape after feature engineering : {df.shape}")

# ─────────────────────────────────────────────────────────────────
# 2. DROP REDUNDANT / RAW COLUMNS
# ─────────────────────────────────────────────────────────────────
print("\n[2] DROPPING REDUNDANT COLUMNS")
drop_cols = ["Time", "Amount", "Amount_capped", "Time_of_day_s"]
existing_drops = [c for c in drop_cols if c in df.columns]
df.drop(columns=existing_drops, inplace=True)
print(f"  Dropped : {existing_drops}")
print(f"  Shape   : {df.shape}")

# ─────────────────────────────────────────────────────────────────
# 3. FEATURE MATRIX & TARGET
# ─────────────────────────────────────────────────────────────────
print("\n[3] FEATURE MATRIX & TARGET")
target_col = "Class"
# Cast boolean dummies to int if needed
bool_cols = df.select_dtypes(include="bool").columns.tolist()
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop(columns=[target_col])
y = df[target_col]
print(f"  X shape : {X.shape}   |   y shape : {y.shape}")
print(f"  Feature names ({len(X.columns)}):")
for i, col in enumerate(X.columns, 1):
    print(f"    {i:>3}. {col}")

# ─────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  (stratified 80:20)
# ─────────────────────────────────────────────────────────────────
print("\n[4] TRAIN / TEST SPLIT  (stratified 80:20)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train : X={X_train.shape}  y={y_train.shape}  | fraud={y_train.sum():,}")
print(f"  Test  : X={X_test.shape}  y={y_test.shape}  | fraud={y_test.sum():,}")
print(f"  Fraud ratio - Train: {y_train.mean()*100:.4f}%  Test: {y_test.mean()*100:.4f}%")

# ─────────────────────────────────────────────────────────────────
# 5. SCALE CONTINUOUS FEATURES (fit on train, transform both)
# ─────────────────────────────────────────────────────────────────
print("\n[5] SCALING CONTINUOUS FEATURES (StandardScaler, fit on train only)")
scale_cols = (
    [c for c in X_train.columns if c.startswith("V")]
    + ["Amount_log", "Time_scaled", "V_mean", "V_std", "V_min", "V_max",
       "V_range", "V14_x_V17", "V14_x_V12"]
)
scale_cols = [c for c in scale_cols if c in X_train.columns]

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])
print(f"  Scaled {len(scale_cols)} columns.")

# ─────────────────────────────────────────────────────────────────
# 6. HANDLE CLASS IMBALANCE — SMOTE (on train only)
# ─────────────────────────────────────────────────────────────────
print("\n[6] CLASS IMBALANCE  -  SMOTE Oversampling")
if SMOTE_AVAILABLE:
    print(f"  Before SMOTE  ->  Class 0: {(y_train==0).sum():,}  | Class 1: {(y_train==1).sum():,}")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"  After  SMOTE  ->  Class 0: {(y_train_sm==0).sum():,}  | Class 1: {(y_train_sm==1).sum():,}")
    print(f"  Resampled train shape : {X_train_sm.shape}")
else:
    print("  [SKIP] imbalanced-learn not installed. Using class_weight='balanced' instead.")
    print("         Install via:  pip install imbalanced-learn")
    X_train_sm, y_train_sm = X_train.copy(), y_train.copy()

# ─────────────────────────────────────────────────────────────────
# 7. FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────
print("\n[7] WRANGLING SUMMARY")
summary = {
    "Original rows"         : 284807,
    "After cleaning"        : 283726,
    "Final features"        : X.shape[1],
    "Train samples"         : X_train_sm.shape[0],
    "Train fraud samples"   : int(y_train_sm.sum()),
    "Test samples"          : X_test.shape[0],
    "Test fraud samples"    : int(y_test.sum()),
}
for k, v in summary.items():
    print(f"  {k:<25} : {v:,}")

# ─────────────────────────────────────────────────────────────────
# 8. SAVE WRANGLED FILES
# ─────────────────────────────────────────────────────────────────
print("\n[8] SAVING WRANGLED FILES")

# Full wrangled dataset (before split)
df_full = X.copy()
df_full["Class"] = y.values
df_full.to_csv(OUT_DIR + "creditcard_wrangled.csv", index=False)
print(f"  creditcard_wrangled.csv  saved  {df_full.shape}")

# Train & Test splits (SMOTE applied to train)
train_df = pd.DataFrame(X_train_sm, columns=X.columns)
train_df["Class"] = y_train_sm.values
train_df.to_csv(OUT_DIR + "train_smote.csv", index=False)
print(f"  train_smote.csv          saved  {train_df.shape}")

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["Class"] = y_test.values
test_df.to_csv(OUT_DIR + "test.csv", index=False)
print(f"  test.csv                 saved  {test_df.shape}")

print("\n" + "=" * 65)
print("DATA WRANGLING COMPLETE")
print("=" * 65)
