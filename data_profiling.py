import pandas as pd
import numpy as np

CSV_PATH = "D:/CREDIT_CARD_FRAUD/creditcard.csv"

print("=" * 70)
print("CREDIT CARD FRAUD - DATA PROFILING REPORT")
print("=" * 70)

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ── 2. Basic info ─────────────────────────────────────────────────────────────
print("\n[1] SHAPE")
print(f"  Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")

print("\n[2] COLUMN DTYPES")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {str(dtype):<12} : {count} column(s)")

# ── 3. Missing values ─────────────────────────────────────────────────────────
print("\n[3] MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
missing_df = missing_df[missing_df["missing_count"] > 0]
if missing_df.empty:
    print("  No missing values found.")
else:
    print(missing_df.to_string())

# ── 4. Duplicate rows ─────────────────────────────────────────────────────────
print("\n[4] DUPLICATE ROWS")
dup_count = df.duplicated().sum()
print(f"  Total duplicates : {dup_count:,}  ({dup_count/len(df)*100:.4f}%)")

# ── 5. Class distribution (target) ───────────────────────────────────────────
print("\n[5] TARGET CLASS DISTRIBUTION  (0=Legit, 1=Fraud)")
class_counts = df["Class"].value_counts().sort_index()
class_pct    = (class_counts / len(df) * 100).round(4)
class_df = pd.DataFrame({"count": class_counts, "percent_%": class_pct})
print(class_df.to_string())
fraud_ratio = class_counts[1] / class_counts[0]
print(f"\n  Imbalance ratio  (legit:fraud) = {class_counts[0]:,} : {class_counts[1]:,}"
      f"  ({1/fraud_ratio:.0f}:1)")

# ── 6. Descriptive statistics ─────────────────────────────────────────────────
print("\n[6] DESCRIPTIVE STATISTICS")
desc = df.describe().T
desc["range"] = desc["max"] - desc["min"]
desc["cv_%"]  = (desc["std"] / desc["mean"].replace(0, np.nan) * 100).round(2)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 120)
print(desc[["count","mean","std","min","25%","50%","75%","max","range","cv_%"]].to_string())

# ── 7. Time column analysis ────────────────────────────────────────────────────
print("\n[7] TIME COLUMN ANALYSIS")
print(f"  Min  : {df['Time'].min():,.0f} s  ({df['Time'].min()/3600:.2f} h)")
print(f"  Max  : {df['Time'].max():,.0f} s  ({df['Time'].max()/3600:.2f} h)")
print(f"  Span : ~{df['Time'].max()/3600/24:.1f} days of transaction data")

# ── 8. Amount column analysis ─────────────────────────────────────────────────
print("\n[8] AMOUNT COLUMN ANALYSIS")
print(f"  Mean   : ${df['Amount'].mean():.2f}")
print(f"  Median : ${df['Amount'].median():.2f}")
print(f"  Std    : ${df['Amount'].std():.2f}")
print(f"  Min    : ${df['Amount'].min():.2f}")
print(f"  Max    : ${df['Amount'].max():.2f}")
print(f"  Skew   : {df['Amount'].skew():.4f}  (>0 right-skewed)")

print("\n  Amount by Class:")
amt_by_class = df.groupby("Class")["Amount"].agg(["mean","median","max","std"]).round(2)
amt_by_class.index = amt_by_class.index.map({0: "Legit", 1: "Fraud"})
print(amt_by_class.to_string())

# ── 9. Zero-amount transactions ────────────────────────────────────────────────
print("\n[9] ZERO-AMOUNT TRANSACTIONS")
zero_amt = df[df["Amount"] == 0]
print(f"  Count: {len(zero_amt):,}  ({len(zero_amt)/len(df)*100:.2f}%)")
if len(zero_amt) > 0:
    print(f"  Of those, fraud: {zero_amt['Class'].sum():,}")

# ── 10. PCA feature (V1-V28) skewness & kurtosis ─────────────────────────────
print("\n[10] PCA FEATURES (V1-V28) — SKEWNESS & KURTOSIS (top 10 most skewed)")
v_cols = [c for c in df.columns if c.startswith("V")]
skew_kurt = pd.DataFrame({
    "skewness": df[v_cols].skew(),
    "kurtosis": df[v_cols].kurt()
}).round(4)
skew_kurt["abs_skew"] = skew_kurt["skewness"].abs()
top10 = skew_kurt.nlargest(10, "abs_skew")[["skewness","kurtosis"]]
print(top10.to_string())

# ── 11. Correlation of features with Class ────────────────────────────────────
print("\n[11] FEATURE CORRELATION WITH CLASS (top 10 absolute)")
corr_with_class = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
print(corr_with_class.head(10).to_string())

# ── 12. Outlier summary (IQR method) ─────────────────────────────────────────
print("\n[12] OUTLIER SUMMARY — IQR METHOD (Amount & Time)")
for col in ["Amount", "Time"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"  {col:<8}: {len(outliers):,} outliers  ({len(outliers)/len(df)*100:.2f}%)")

print("\n" + "=" * 70)
print("END OF PROFILING REPORT")
print("=" * 70)
