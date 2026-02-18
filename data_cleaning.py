# encoding: utf-8
"""
==========================================================
STEP 1 - DATA CLEANING
==========================================================
Tasks:
  1. Drop duplicate rows
  2. Confirm & report missing values
  3. Identify & cap outliers (IQR capping on Amount)
  4. Remove / flag zero-amount transactions
  5. Log-transform Amount  ->  Amount_log
  6. Standardise Time      ->  Time_scaled
  7. Save cleaned dataset
==========================================================
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CSV_IN  = "D:/CREDIT_CARD_FRAUD/creditcard.csv"
CSV_OUT = "D:/CREDIT_CARD_FRAUD/creditcard_cleaned.csv"

print("=" * 65)
print("STEP 1 - DATA CLEANING")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
print(f"\nOriginal shape : {df.shape}")

# ────────────────────────────────────────────────────────────────
# 1. MISSING VALUES
# ────────────────────────────────────────────────────────────────
print("\n[1] Missing Values Check")
missing = df.isnull().sum().sum()
print(f"  Total missing cells : {missing}")
if missing == 0:
    print("  [OK] No missing values - nothing to impute.")

# ────────────────────────────────────────────────────────────────
# 2. DUPLICATE ROWS
# ────────────────────────────────────────────────────────────────
print("\n[2] Duplicate Rows")
before = len(df)
dups   = df.duplicated().sum()
print(f"  Duplicates found : {dups:,}")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
after = len(df)
print(f"  Rows removed     : {before - after:,}")
print(f"  Shape after drop : {df.shape}")

# ────────────────────────────────────────────────────────────────
# 3. ZERO-AMOUNT TRANSACTIONS
# ────────────────────────────────────────────────────────────────
print("\n[3] Zero-Amount Transactions")
zero_mask  = df["Amount"] == 0
zero_total = zero_mask.sum()
zero_fraud = df.loc[zero_mask, "Class"].sum()
print(f"  Total zero-amount rows : {zero_total:,}")
print(f"  Of which fraud         : {zero_fraud}")
print(f"  Strategy : Add binary flag 'is_zero_amount' and keep rows")
df["is_zero_amount"] = zero_mask.astype(int)

# ────────────────────────────────────────────────────────────────
# 4. OUTLIER CAPPING — Amount (IQR method, 1.5×IQR)
# ────────────────────────────────────────────────────────────────
print("\n[4] Outlier Capping on 'Amount'  (IQR × 1.5)")
Q1  = df["Amount"].quantile(0.25)
Q3  = df["Amount"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f"  Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
print(f"  Lower fence : {lower:.2f}  |  Upper fence : {upper:.2f}")

n_low  = (df["Amount"] < lower).sum()
n_high = (df["Amount"] > upper).sum()
print(f"  Rows below lower fence : {n_low:,}")
print(f"  Rows above upper fence : {n_high:,}  -> capped to {upper:.2f}")

df["Amount_capped"] = df["Amount"].clip(lower=max(lower, 0), upper=upper)
print(f"  'Amount_capped' column created.")

# ────────────────────────────────────────────────────────────────
# 5. LOG TRANSFORMATION — Amount
# ────────────────────────────────────────────────────────────────
print("\n[5] Log Transformation of 'Amount'")
# log1p handles zero-amount rows safely
df["Amount_log"] = np.log1p(df["Amount_capped"])
print(f"  'Amount_log' created via log1p(Amount_capped)")
print(f"  Skew before : {df['Amount'].skew():.4f}")
print(f"  Skew after  : {df['Amount_log'].skew():.4f}")

# ────────────────────────────────────────────────────────────────
# 6. SCALE 'Time'
# ────────────────────────────────────────────────────────────────
print("\n[6] Standardise 'Time'")
scaler = StandardScaler()
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
print(f"  'Time_scaled' created.")
print(f"  Mean ~ {df['Time_scaled'].mean():.6f}  Std ~ {df['Time_scaled'].std():.4f}")

# ────────────────────────────────────────────────────────────────
# 7. DROP RAW COLUMNS NO LONGER NEEDED (keep originals too for reference)
# ────────────────────────────────────────────────────────────────
print("\n[7] Column Summary After Cleaning")
print(f"  Columns : {df.shape[1]}")
print(f"  Rows    : {df.shape[0]:,}")

# ────────────────────────────────────────────────────────────────
# 8. FINAL CLASS DISTRIBUTION CHECK
# ────────────────────────────────────────────────────────────────
print("\n[8] Class Distribution After Cleaning")
cc = df["Class"].value_counts().sort_index()
cc_pct = (cc / len(df) * 100).round(4)
print(pd.DataFrame({"count": cc, "percent_%": cc_pct}).to_string())

# ────────────────────────────────────────────────────────────────
# 9. SAVE
# ────────────────────────────────────────────────────────────────
df.to_csv(CSV_OUT, index=False)
print(f"\n[DONE] Cleaned dataset saved -> {CSV_OUT}")
print(f"  Final shape : {df.shape}")
print("\n" + "=" * 65)
print("DATA CLEANING COMPLETE")

print("=" * 65)
