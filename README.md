# Credit Card Fraud Detection

A full data analysis and business intelligence pipeline for the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Dataset
- **Source:** Kaggle — ULB Machine Learning Group
- **Records:** 284,807 transactions over 2 days
- **Features:** 30 (Time, V1–V28 PCA components, Amount) + 1 target (Class)
- **Class imbalance:** 0.17% fraud (492 out of 284,807) — ratio of **578:1**

> **Note:** Raw and processed CSV files are excluded from this repository (`.gitignore`) due to file size. Download the dataset from Kaggle and place `creditcard.csv` in the project root.

---

## Pipeline

### 1. `data_profiling.py` — Data Profiling
- Shape, dtypes, missing values, duplicates
- Class distribution & imbalance ratio
- Descriptive statistics for all 31 features
- Skewness, kurtosis, outlier detection (IQR)
- Feature correlation with target (`Class`)

### 2. `data_cleaning.py` — Data Cleaning
- Drop 1,081 duplicate rows (283,726 remain)
- Flag zero-amount transactions → `is_zero_amount` binary column
- IQR-based outlier capping on `Amount` (upper fence: $185.38, 31,685 rows capped)
- Log transformation of Amount → `Amount_log` (skew reduced: 16.98 → -0.18)
- StandardScaler on `Time` → `Time_scaled` (mean ≈ 0, std ≈ 1)

### 3. `data_wrangling.py` — Data Wrangling & Feature Engineering
- Time features: `Time_hour`, `is_daytime`, `Day_period`
- Amount bins (zero / micro / small / medium / large / xlarge / xxlarge) — one-hot encoded
- V-feature aggregates: `V_mean`, `V_std`, `V_min`, `V_max`, `V_range`, `V_pos_ct`, `V_neg_ct`
- Interaction terms: `V14_x_V17`, `V14_x_V12` (top correlated with fraud)
- Stratified 80:20 train/test split
- SMOTE oversampling on training set only (fraud: 378 → 226,602 — balanced 50:50)

### 4. `eda_business_insights.py` — EDA & Business Insights
Full exploratory data analysis framed as actionable business intelligence. Produces **6 charts** and a console report answering:

| Business Question | Finding |
|---|---|
| When does fraud peak? | **02:00 AM** — fraud rate 1.45% (5× daily average) |
| What amounts are riskiest? | Zero-amount highest rate (1.38%); $100–500 highest $ loss ($21K) |
| What is the fraud median amount? | **$9.82** vs legit **$22.00** — fraudsters stay small |
| Total financial exposure? | **$58,591** across 473 fraud cases; avg $123.87/incident |
| Probe transactions? | **25 zero-amount fraud** cases — card verification attacks |
| Fraud velocity? | Median gap between frauds: **2 minutes** — burst attacks |
| Strongest ML features? | **V17, V14, V12** — KS statistic > 0.70 |

#### Charts Generated

| File | Description |
|------|-------------|
| `fig1_overview_dashboard.png` | Executive dashboard: class imbalance, $ exposure, hourly fraud rate, 24h volume vs fraud overlay, amount boxplot |
| `fig2_time_intelligence.png` | Polar clock chart, Day 1 vs Day 2 comparison, fraud velocity histogram, cumulative arrival curves |
| `fig3_amount_risk.png` | Fraud rate by amount band, KDE density (legit vs fraud), $ exposure per band, log-transform effect |
| `fig4_feature_signals.png` | KDE separation plots for top 6 PCA features (V17, V14, V12, V10, V4, V11) with KS statistics |
| `fig5_fraud_fingerprint.png` | Full feature correlation bar chart + radar chart of fraud vs legit mean profile |
| `fig6_business_scorecard.png` | KPI metrics table + 6 strategic business recommendations |

#### Strategic Recommendations

1. **Overnight Controls (23:00–03:00)** — fraud rate spikes at 02:00; deploy step-up authentication or lower limits
2. **Micro-Transaction Monitoring ($0–$10)** — highest fraud rate band; flag repeated small amounts from same card
3. **Zero-Amount Alerts** — 25 fraud probes detected; auto-block zero-auth on new/unseen cards
4. **Real-Time V17/V14 Scoring** — KS > 0.70; weight these features heavily in any live scoring model
5. **Balanced Model Training** — 578:1 imbalance; use SMOTE + F1/Recall metrics, not accuracy
6. **Large Transaction Review Queue** — $500–$1K band drives biggest $ losses; route to human review below confidence threshold

---

## Output Files (not tracked in git)

| File | Description |
|------|-------------|
| `creditcard_cleaned.csv` | After deduplication, capping, log-transform (283,726 × 35) |
| `creditcard_wrangled.csv` | Full feature-engineered dataset (283,726 × 51, 50 features) |
| `train_smote.csv` | Balanced training set after SMOTE (453,204 × 51) |
| `test.csv` | Untouched stratified test set (56,746 × 51) |

---

### 5. `dashboard.py` — Interactive Streamlit Dashboard
A fully interactive, 5-page dashboard for live transaction monitoring and business storytelling.

| Page | What It Shows |
|------|--------------|
| 📊 **Executive Overview** | KPI metrics, class imbalance donut, $ exposure by band, 24h fraud rate, amount boxplot, scorecard table |
| 🕐 **Time Intelligence** | 24h volume vs fraud overlay, polar clock chart, Day 1 vs Day 2, fraud velocity histogram, cumulative arrivals |
| 💰 **Amount Risk Profiling** | Fraud rate & $ exposure by band, interactive KDE density, log-transform comparison, treemap |
| 🔬 **Feature Signals** | KS-statistic ranking for all 28 PCA features, interactive KDE + violin explorer, correlation bar chart |
| 🚨 **Live Transaction Monitor** | Adjustable fraud score threshold, precision/recall metrics, flagged transaction table with progress bars, score vs amount scatter |

**Key dashboard features:**
- Global sidebar filters (day, hour range, max amount) applied across all pages
- Rule-based fraud scoring model using V17, V14, V12, V10, Amount_log
- Live precision/recall/exposure metrics that update with threshold slider
- Colour-coded transaction table with Streamlit `ProgressColumn` for fraud scores

---

## Requirements

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy plotly streamlit
```

---

## Usage

Run scripts in order:

```bash
python data_profiling.py        # Step 1: Profile the raw data
python data_cleaning.py         # Step 2: Clean & transform
python data_wrangling.py        # Step 3: Feature engineering + SMOTE splits
python eda_business_insights.py # Step 4: Generate EDA charts (6 PNGs)
streamlit run dashboard.py      # Step 5: Launch interactive dashboard
```

The dashboard opens automatically at **http://localhost:8501**

---

## Repository Structure

```
credit_card_fraud/
├── data_profiling.py            # Data profiling report
├── data_cleaning.py             # Cleaning pipeline
├── data_wrangling.py            # Feature engineering & splits
├── eda_business_insights.py     # EDA & business storytelling (6 charts)
├── dashboard.py                 # Streamlit interactive dashboard (5 pages)
├── requirements.txt             # Python dependencies
├── fig1_overview_dashboard.png  # Executive overview chart
├── fig2_time_intelligence.png   # Time-based fraud analysis
├── fig3_amount_risk.png         # Amount risk profiling
├── fig4_feature_signals.png     # Feature separation (KDE + KS)
├── fig5_fraud_fingerprint.png   # Correlation + radar chart
├── fig6_business_scorecard.png  # KPI table + recommendations
├── .gitignore                   # Excludes CSV files
└── README.md
```
