# encoding: utf-8
"""
=============================================================
CREDIT CARD FRAUD DETECTION — INTERACTIVE STREAMLIT DASHBOARD
=============================================================
Pages:
  1. Executive Overview      — KPIs, class distribution, exposure
  2. Time Intelligence       — Hourly/daily fraud patterns
  3. Amount Risk Profiling   — Transaction amount analysis
  4. Feature Signals         — PCA feature separation & correlations
  5. Live Transaction Monitor— Flagged transactions + model scoring
=============================================================
Run: streamlit run dashboard.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
FRAUD_CLR  = "#E63946"
LEGIT_CLR  = "#457B9D"
ACCENT_CLR = "#F4A261"
BG_CARD    = "#1E2130"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
.fraud-badge  { background:#E63946; color:white; padding:2px 10px;
                border-radius:12px; font-size:0.75rem; font-weight:600; }
.legit-badge  { background:#457B9D; color:white; padding:2px 10px;
                border-radius:12px; font-size:0.75rem; font-weight:600; }
.insight-box  { background:#FFF8E7; border-left:4px solid #F4A261;
                padding:12px 16px; border-radius:4px; margin:8px 0;
                font-size:0.88rem; color:#333; }
.section-title{ font-size:1.1rem; font-weight:700; color:#E63946;
                letter-spacing:0.05em; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
DATA_PATH = "D:/CREDIT_CARD_FRAUD/creditcard_cleaned.csv"

@st.cache_data(show_spinner="Loading transaction data...")
def load_data(path):
    df = pd.read_csv(path)
    SECONDS_IN_DAY = 86400
    df["Hour"]      = (df["Time"] % SECONDS_IN_DAY // 3600).astype(int)
    df["Day"]       = (df["Time"] // SECONDS_IN_DAY).astype(int)
    df["Day_label"] = df["Day"].map({0: "Day 1", 1: "Day 2"})
    bins   = [-0.01, 0, 10, 50, 100, 500, 1000, np.inf]
    labels = ["$0 (zero)", "$0-10 (micro)", "$10-50 (small)",
              "$50-100 (medium)", "$100-500 (large)",
              "$500-1K (xlarge)", "$1K+ (xxlarge)"]
    df["Amount_band"] = pd.cut(df["Amount"], bins=bins, labels=labels)
    # Simple rule-based fraud score for "live monitor"
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    df["fraud_score"] = (
        0.30 * df["V17"].abs() / (df["V17"].abs().max() + 1e-9) +
        0.25 * df["V14"].abs() / (df["V14"].abs().max() + 1e-9) +
        0.20 * df["V12"].abs() / (df["V12"].abs().max() + 1e-9) +
        0.15 * df["Amount_log"] / (df["Amount_log"].max() + 1e-9) +
        0.10 * df["V10"].abs() / (df["V10"].abs().max() + 1e-9)
    ).clip(0, 1)
    return df

df = load_data(DATA_PATH)
fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

# ── Pre-compute hourly stats ───────────────────────────────────────────────────
hourly = df.groupby("Hour")["Class"].agg(["sum", "count"]).reset_index()
hourly.columns = ["Hour", "fraud_count", "total"]
hourly["fraud_rate"] = hourly["fraud_count"] / hourly["total"] * 100
hourly["legit_count"] = hourly["total"] - hourly["fraud_count"]

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-cards.png", width=64)
    st.title("🛡️ Fraud Detection")
    st.caption("Credit Card Fraud Analysis Dashboard")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Executive Overview",
         "🕐 Time Intelligence",
         "💰 Amount Risk Profiling",
         "🔬 Feature Signals",
         "🚨 Live Transaction Monitor"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Filters**")
    day_filter = st.multiselect("Day", ["Day 1", "Day 2"], default=["Day 1", "Day 2"])
    hour_range = st.slider("Hour of Day", 0, 23, (0, 23))
    amount_max = st.number_input("Max Amount ($)", value=float(df["Amount"].max()), step=100.0)

    # Apply filters
    mask = (
        df["Day_label"].isin(day_filter) &
        df["Hour"].between(hour_range[0], hour_range[1]) &
        (df["Amount"] <= amount_max)
    )
    dff   = df[mask].copy()
    fraudf = dff[dff["Class"] == 1]
    legitf = dff[dff["Class"] == 0]

    st.divider()
    st.caption(f"Showing **{len(dff):,}** transactions")
    st.caption(f"**{len(fraudf):,}** fraud · **{len(legitf):,}** legit")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.title("📊 Executive Overview")
    st.caption("High-level KPIs and business exposure summary")
    st.divider()

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    fraud_rate  = len(fraudf) / len(dff) * 100 if len(dff) else 0
    exposure    = fraudf["Amount"].sum()
    avg_fraud   = fraudf["Amount"].mean() if len(fraudf) else 0
    max_fraud   = fraudf["Amount"].max() if len(fraudf) else 0

    k1.metric("Total Transactions",  f"{len(dff):,}")
    k2.metric("Fraud Cases",         f"{len(fraudf):,}",
              delta=f"{fraud_rate:.4f}% of total", delta_color="inverse")
    k3.metric("Total Exposure ($)",  f"${exposure:,.2f}",
              delta="Revenue at risk", delta_color="inverse")
    k4.metric("Avg Fraud Amount",    f"${avg_fraud:.2f}",
              delta=f"vs Legit ${legitf['Amount'].mean():.2f}")
    k5.metric("Max Single Fraud",    f"${max_fraud:,.2f}")

    st.divider()

    # ── Row 1: Donut + Exposure bar ───────────────────────────────────────────
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<p class="section-title">CLASS DISTRIBUTION</p>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["Legit", "Fraud"],
            values=[len(legitf), len(fraudf)],
            hole=0.55,
            marker_colors=[LEGIT_CLR, FRAUD_CLR],
            textinfo="label+percent",
            textfont_size=13,
        ))
        fig_donut.update_layout(
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            annotations=[dict(text=f"578:1<br>Ratio", x=0.5, y=0.5,
                              font_size=14, showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown(
            '<div class="insight-box">⚠️ <b>578 legit for every 1 fraud.</b> '
            'Standard accuracy is misleading — a model predicting '
            '"always legit" scores 99.8% accuracy but catches zero fraud.</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown('<p class="section-title">FINANCIAL EXPOSURE BY AMOUNT BAND</p>', unsafe_allow_html=True)
        band_exp = fraudf.groupby("Amount_band", observed=True).agg(
            Exposure=("Amount", "sum"),
            Cases=("Amount", "count"),
            Avg=("Amount", "mean")
        ).reset_index()
        fig_band = px.bar(
            band_exp, x="Amount_band", y="Exposure",
            color="Cases", color_continuous_scale=["#FFF3CD", FRAUD_CLR],
            text=band_exp["Exposure"].apply(lambda x: f"${x:,.0f}"),
            labels={"Amount_band": "Amount Band", "Exposure": "Total Fraud ($)"},
            height=280,
        )
        fig_band.update_traces(textposition="outside", textfont_size=10)
        fig_band.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_band, use_container_width=True)

    st.divider()

    # ── Row 2: Hourly overview heatmap + Day comparison ───────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<p class="section-title">FRAUD RATE BY HOUR (ALL DATA)</p>', unsafe_allow_html=True)
        fig_hour = px.bar(
            hourly, x="Hour", y="fraud_rate",
            color="fraud_rate",
            color_continuous_scale=["#E8F4F8", FRAUD_CLR],
            labels={"Hour": "Hour of Day", "fraud_rate": "Fraud Rate (%)"},
            height=270,
        )
        fig_hour.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False,
                               xaxis=dict(dtick=2))
        peak = hourly.loc[hourly["fraud_rate"].idxmax(), "Hour"]
        fig_hour.add_vline(x=peak, line_dash="dash", line_color=FRAUD_CLR,
                           annotation_text=f"Peak: {peak:02d}:00",
                           annotation_font_color=FRAUD_CLR)
        st.plotly_chart(fig_hour, use_container_width=True)
        st.markdown(
            f'<div class="insight-box">🕐 <b>Fraud peaks at {peak:02d}:00</b> — '
            f'overnight hours carry disproportionate risk. '
            f'Deploy step-up authentication between 23:00–03:00.</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown('<p class="section-title">AMOUNT DISTRIBUTION: LEGIT vs FRAUD</p>', unsafe_allow_html=True)
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=legitf["Amount"].clip(upper=500), name="Legit",
            marker_color=LEGIT_CLR, boxmean=True, notched=True
        ))
        fig_box.add_trace(go.Box(
            y=fraudf["Amount"].clip(upper=500), name="Fraud",
            marker_color=FRAUD_CLR, boxmean=True, notched=True
        ))
        fig_box.update_layout(
            yaxis_title="Amount ($ clipped at 500)",
            height=270, margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown(
            f'<div class="insight-box">💡 Fraud median = <b>${fraudf["Amount"].median():.2f}</b> '
            f'vs Legit median = <b>${legitf["Amount"].median():.2f}</b>. '
            f'Fraudsters deliberately keep amounts small to fly under radar thresholds.</div>',
            unsafe_allow_html=True
        )

    # ── Scorecard table ───────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">KEY METRICS SCORECARD</p>', unsafe_allow_html=True)
    scorecard = pd.DataFrame({
        "Metric": [
            "Total Transactions", "Fraud Cases", "Fraud Rate",
            "Class Imbalance Ratio", "Total Fraud Exposure",
            "Avg Fraud Amount", "Avg Legit Amount", "Max Fraud Amount",
            "Zero-Amount Fraud Cases", "Fraud Peak Hour"
        ],
        "Value": [
            f"{len(dff):,}",
            f"{len(fraudf):,}",
            f"{fraud_rate:.4f}%",
            "578 : 1",
            f"${exposure:,.2f}",
            f"${avg_fraud:.2f}",
            f"${legitf['Amount'].mean():.2f}",
            f"${max_fraud:,.2f}",
            f"{(fraudf['Amount']==0).sum()}",
            f"{peak:02d}:00"
        ],
        "Business Implication": [
            "Dataset baseline volume",
            "Direct loss events",
            "Needle-in-haystack — use Recall, not Accuracy",
            "Severe imbalance — SMOTE / class weights required",
            "Revenue at risk without detection model",
            "Per-incident average loss",
            "Normal transaction size baseline",
            "Worst-case single exposure",
            "Card verification probes — block zero-auth",
            "Tighten overnight transaction controls"
        ]
    })
    st.dataframe(scorecard, use_container_width=True, hide_index=True,
                 column_config={"Metric": st.column_config.TextColumn(width="medium"),
                                "Value":  st.column_config.TextColumn(width="small")})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TIME INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🕐 Time Intelligence":
    st.title("🕐 Time Intelligence")
    st.caption("When do fraudsters strike? Temporal pattern analysis.")
    st.divider()

    # ── Row 1: Volume vs Fraud dual-axis ──────────────────────────────────────
    st.markdown('<p class="section-title">24-HOUR TRANSACTION VOLUME vs FRAUD FREQUENCY</p>',
                unsafe_allow_html=True)
    h_legit = dff[dff["Class"]==0].groupby("Hour").size().reset_index(name="Legit")
    h_fraud = dff[dff["Class"]==1].groupby("Hour").size().reset_index(name="Fraud")
    h_merge = pd.merge(h_legit, h_fraud, on="Hour", how="left").fillna(0)

    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(go.Bar(x=h_merge["Hour"], y=h_merge["Legit"],
                              name="Legit Volume", marker_color=LEGIT_CLR,
                              opacity=0.6), secondary_y=False)
    fig_dual.add_trace(go.Scatter(x=h_merge["Hour"], y=h_merge["Fraud"],
                                  name="Fraud Count", mode="lines+markers",
                                  line=dict(color=FRAUD_CLR, width=3),
                                  marker=dict(size=8)), secondary_y=True)
    fig_dual.update_yaxes(title_text="Legit Transaction Count", secondary_y=False)
    fig_dual.update_yaxes(title_text="Fraud Count", secondary_y=True,
                          color=FRAUD_CLR)
    fig_dual.update_xaxes(title_text="Hour of Day", dtick=1)
    fig_dual.update_layout(height=320, margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_dual, use_container_width=True)
    st.markdown(
        '<div class="insight-box">📌 <b>Volume ≠ Risk.</b> Low transaction volumes '
        'at night do not mean low fraud risk — the fraud RATE is highest precisely '
        'when transaction volume is lowest, as anomalies are harder to detect in sparse data.</div>',
        unsafe_allow_html=True
    )

    st.divider()
    c1, c2 = st.columns(2)

    # ── Hourly fraud rate polar ────────────────────────────────────────────────
    with c1:
        st.markdown('<p class="section-title">FRAUD RATE CLOCK (POLAR VIEW)</p>',
                    unsafe_allow_html=True)
        h_rate = dff.groupby("Hour")["Class"].agg(["sum","count"]).reset_index()
        h_rate.columns = ["Hour","Fraud","Total"]
        h_rate["Rate"] = h_rate["Fraud"] / h_rate["Total"] * 100
        h_rate["Hour_label"] = h_rate["Hour"].apply(lambda x: f"{x:02d}:00")
        avg_rate = h_rate["Rate"].mean()
        h_rate["color"] = h_rate["Rate"].apply(
            lambda x: FRAUD_CLR if x > avg_rate else LEGIT_CLR)

        fig_polar = px.bar_polar(
            h_rate, r="Rate", theta="Hour_label",
            color="Rate", color_continuous_scale=["#E8F4F8","#FFF3CD", FRAUD_CLR],
            template="plotly_dark",
            range_r=[0, h_rate["Rate"].max() * 1.2],
        )
        fig_polar.update_layout(height=380, margin=dict(t=20, b=20),
                                coloraxis_showscale=False,
                                polar=dict(bgcolor="#0E1117"))
        st.plotly_chart(fig_polar, use_container_width=True)

    # ── Day 1 vs Day 2 ─────────────────────────────────────────────────────────
    with c2:
        st.markdown('<p class="section-title">DAY 1 vs DAY 2 FRAUD BREAKDOWN</p>',
                    unsafe_allow_html=True)
        day_stats = dff.groupby(["Day_label","Class"]).size().unstack(fill_value=0)
        day_stats.columns = ["Legit","Fraud"]
        day_stats = day_stats.reset_index()
        day_stats["Fraud_Rate_%"] = (
            day_stats["Fraud"]/(day_stats["Legit"]+day_stats["Fraud"])*100
        ).round(4)

        fig_day = make_subplots(specs=[[{"secondary_y": True}]])
        fig_day.add_trace(go.Bar(x=day_stats["Day_label"], y=day_stats["Legit"],
                                 name="Legit", marker_color=LEGIT_CLR, opacity=0.7),
                          secondary_y=False)
        fig_day.add_trace(go.Bar(x=day_stats["Day_label"], y=day_stats["Fraud"]*50,
                                 name="Fraud ×50", marker_color=FRAUD_CLR, opacity=0.9),
                          secondary_y=False)
        fig_day.add_trace(go.Scatter(x=day_stats["Day_label"],
                                     y=day_stats["Fraud_Rate_%"],
                                     mode="lines+markers+text",
                                     name="Fraud Rate %",
                                     text=day_stats["Fraud_Rate_%"].apply(lambda x: f"{x:.4f}%"),
                                     textposition="top center",
                                     line=dict(color=ACCENT_CLR, width=3),
                                     marker=dict(size=12, symbol="diamond")),
                          secondary_y=True)
        fig_day.update_yaxes(title_text="Transaction Count", secondary_y=False)
        fig_day.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True,
                             color=ACCENT_CLR)
        fig_day.update_layout(height=380, margin=dict(t=20, b=20),
                              barmode="group", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_day, use_container_width=True)

    st.divider()

    # ── Fraud Velocity ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">FRAUD VELOCITY — TIME GAP BETWEEN CONSECUTIVE FRAUD EVENTS</p>',
                unsafe_allow_html=True)
    fraud_sorted = fraudf["Time"].sort_values().reset_index(drop=True)
    inter_gaps   = (fraud_sorted.diff().dropna() / 60).clip(upper=200)  # minutes

    fig_vel = px.histogram(
        inter_gaps, nbins=60,
        labels={"value": "Minutes Between Consecutive Fraud Events", "count": "Frequency"},
        color_discrete_sequence=[FRAUD_CLR], height=300,
        opacity=0.85,
    )
    med_gap = inter_gaps.median()
    p10_gap = inter_gaps.quantile(0.1)
    fig_vel.add_vline(x=med_gap, line_dash="dash", line_color="white",
                      annotation_text=f"Median: {med_gap:.0f} min",
                      annotation_font_color="white")
    fig_vel.add_vline(x=p10_gap, line_dash="dot", line_color=ACCENT_CLR,
                      annotation_text=f"10th pct: {p10_gap:.0f} min",
                      annotation_font_color=ACCENT_CLR)
    fig_vel.update_layout(margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig_vel, use_container_width=True)
    st.markdown(
        f'<div class="insight-box">⚡ <b>Burst attacks detected.</b> '
        f'Median gap between fraud events = <b>{med_gap:.0f} minutes</b>. '
        f'10% of consecutive frauds happen within <b>{p10_gap:.0f} minutes</b> of each other. '
        f'Real-time velocity rules (e.g. max 2 txns per card per 5 min) are essential.</div>',
        unsafe_allow_html=True
    )

    # ── Cumulative arrival ────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">CUMULATIVE ARRIVAL — LEGIT vs FRAUD OVER 48 HOURS</p>',
                unsafe_allow_html=True)
    legit_t = legitf["Time"].sort_values() / 3600
    fraud_t = fraudf["Time"].sort_values() / 3600
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=legit_t.values,
        y=np.linspace(0, 100, len(legit_t)),
        name="Legit (cumulative %)", line=dict(color=LEGIT_CLR, width=2)
    ))
    fig_cum.add_trace(go.Scatter(
        x=fraud_t.values,
        y=np.linspace(0, 100, len(fraud_t)),
        name="Fraud (cumulative %)", line=dict(color=FRAUD_CLR, width=2.5)
    ))
    fig_cum.update_layout(
        xaxis_title="Hours Since Recording Start",
        yaxis_title="Cumulative % of Transactions",
        height=300, margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AMOUNT RISK PROFILING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Amount Risk Profiling":
    st.title("💰 Amount Risk Profiling")
    st.caption("What transaction sizes carry the most risk?")
    st.divider()

    # ── KPI row ───────────────────────────────────────────────────────────────
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Fraud Median Amount",  f"${fraudf['Amount'].median():.2f}",
              delta=f"vs Legit ${legitf['Amount'].median():.2f}", delta_color="off")
    a2.metric("Fraud Mean Amount",    f"${fraudf['Amount'].mean():.2f}")
    a3.metric("Zero-Amount Frauds",   f"{(fraudf['Amount']==0).sum()}",
              delta="Card probe attacks", delta_color="inverse")
    a4.metric("Fraud $ Exposure",     f"${fraudf['Amount'].sum():,.0f}")

    st.divider()

    # ── Fraud rate per band ───────────────────────────────────────────────────
    st.markdown('<p class="section-title">FRAUD RATE & EXPOSURE BY AMOUNT BAND</p>',
                unsafe_allow_html=True)

    band_stats = dff.groupby("Amount_band", observed=True)["Class"].agg(
        ["sum","count"]).reset_index()
    band_stats.columns = ["Band","Fraud","Total"]
    band_stats["Fraud_Rate_%"]  = (band_stats["Fraud"] / band_stats["Total"] * 100).round(4)
    band_stats["Exposure_$"]    = dff[dff["Class"]==1].groupby(
        "Amount_band", observed=True)["Amount"].sum().values

    c1, c2 = st.columns(2)
    with c1:
        fig_rate = px.bar(
            band_stats, x="Band", y="Fraud_Rate_%",
            color="Fraud_Rate_%",
            color_continuous_scale=["#E8F4F8", "#FFF3CD", FRAUD_CLR],
            text=band_stats["Fraud_Rate_%"].apply(lambda x: f"{x:.3f}%"),
            labels={"Band": "Amount Band", "Fraud_Rate_%": "Fraud Rate (%)"},
            height=350,
        )
        fig_rate.update_traces(textposition="outside", textfont_size=10)
        fig_rate.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False,
                               xaxis_tickangle=-20)
        avg_rate = band_stats["Fraud_Rate_%"].mean()
        fig_rate.add_hline(y=avg_rate, line_dash="dash", line_color="gray",
                           annotation_text=f"Avg: {avg_rate:.3f}%")
        st.plotly_chart(fig_rate, use_container_width=True)
        st.markdown(
            '<div class="insight-box">🎯 <b>Zero-amount band has the highest fraud RATE (1.38%).</b> '
            'These are silent card verification probes — the fraudster confirms the '
            'card is active before executing a real transaction.</div>',
            unsafe_allow_html=True
        )

    with c2:
        fig_exp = px.bar(
            band_stats, x="Band", y="Exposure_$",
            color="Exposure_$",
            color_continuous_scale=["#FFF3CD", FRAUD_CLR],
            text=band_stats["Exposure_$"].apply(lambda x: f"${x:,.0f}"),
            labels={"Band": "Amount Band", "Exposure_$": "Total Fraud Exposure ($)"},
            height=350,
        )
        fig_exp.update_traces(textposition="outside", textfont_size=10)
        fig_exp.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False,
                               xaxis_tickangle=-20)
        st.plotly_chart(fig_exp, use_container_width=True)
        st.markdown(
            '<div class="insight-box">💸 <b>$100–500 "large" band drives the most $ losses.</b> '
            'Two distinct fraud strategies: micro-probers (high frequency, low value) '
            'and high-value attackers (low frequency, high damage).</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── KDE + Log transform side by side ─────────────────────────────────────
    st.markdown('<p class="section-title">AMOUNT DENSITY & LOG-TRANSFORM EFFECT</p>',
                unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        clip_val = st.slider("Clip amount at ($) for density view", 50, 2000, 500, step=50)
        fig_kde = go.Figure()
        from scipy.stats import gaussian_kde
        for subset, label, color in [
            (legitf["Amount"].clip(upper=clip_val), "Legit", LEGIT_CLR),
            (fraudf["Amount"].clip(upper=clip_val), "Fraud", FRAUD_CLR)
        ]:
            kde = gaussian_kde(subset, bw_method=0.3)
            x_range = np.linspace(0, clip_val, 300)
            fig_kde.add_trace(go.Scatter(
                x=x_range, y=kde(x_range), name=label,
                fill="tozeroy", fillcolor=color.replace(")", ",0.25)").replace("rgb","rgba") if color.startswith("rgb") else color + "40",
                line=dict(color=color, width=2.5)
            ))
        fig_kde.add_vline(x=fraudf["Amount"].median(), line_dash="dash",
                          line_color=FRAUD_CLR,
                          annotation_text=f"Fraud median ${fraudf['Amount'].median():.0f}")
        fig_kde.add_vline(x=legitf["Amount"].median(), line_dash="dash",
                          line_color=LEGIT_CLR,
                          annotation_text=f"Legit median ${legitf['Amount'].median():.0f}")
        fig_kde.update_layout(
            xaxis_title="Transaction Amount ($)",
            yaxis_title="Density",
            height=350, margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig_kde, use_container_width=True)

    with c4:
        fig_log = go.Figure()
        for subset, label, color in [
            (legitf["Amount_log"], "Legit (log)", LEGIT_CLR),
            (fraudf["Amount_log"], "Fraud (log)", FRAUD_CLR)
        ]:
            kde2 = gaussian_kde(subset, bw_method=0.3)
            xr   = np.linspace(subset.min(), subset.max(), 300)
            fig_log.add_trace(go.Scatter(
                x=xr, y=kde2(xr), name=label,
                fill="tozeroy",
                fillcolor=color + "40",
                line=dict(color=color, width=2.5)
            ))
        fig_log.update_layout(
            xaxis_title="log(1 + Amount)",
            yaxis_title="Density",
            height=350, margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig_log, use_container_width=True)
        st.markdown(
            '<div class="insight-box">📐 <b>Log transform separates the distributions.</b> '
            'After log-scaling, fraud peaks at lower values while legit spreads right — '
            'Amount_log becomes a strong ML feature with skew reduced from 16.98 to -0.18.</div>',
            unsafe_allow_html=True
        )

    # ── Sunburst: Band → Class ────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">FRAUD COMPOSITION TREEMAP BY AMOUNT BAND</p>',
                unsafe_allow_html=True)
    tree_df = dff.groupby(["Amount_band","Class"], observed=True).agg(
        Count=("Amount","count"), Total=("Amount","sum")).reset_index()
    tree_df["Class_label"] = tree_df["Class"].map({0:"Legit", 1:"Fraud"})
    fig_tree = px.treemap(
        tree_df, path=["Amount_band","Class_label"],
        values="Count", color="Class_label",
        color_discrete_map={"Legit": LEGIT_CLR, "Fraud": FRAUD_CLR},
        height=350,
    )
    fig_tree.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig_tree, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Signals":
    st.title("🔬 Feature Signals")
    st.caption("Which PCA features best separate fraud from legitimate transactions?")
    st.divider()

    from scipy.stats import ks_2samp

    v_cols = [f"V{i}" for i in range(1, 29)]
    ks_results = []
    for col in v_cols:
        stat, pval = ks_2samp(legitf[col].dropna(), fraudf[col].dropna())
        corr = dff[col].corr(dff["Class"])
        ks_results.append({"Feature": col, "KS_Statistic": round(stat, 4),
                            "P_Value": pval, "Correlation": round(corr, 4)})
    ks_df = pd.DataFrame(ks_results).sort_values("KS_Statistic", ascending=False)

    # ── Ranked bar ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">FEATURE RANKING BY KS STATISTIC (FRAUD SEPARATION POWER)</p>',
                unsafe_allow_html=True)
    fig_ks = px.bar(
        ks_df, x="Feature", y="KS_Statistic",
        color="KS_Statistic",
        color_continuous_scale=["#E8F4F8","#FFF3CD", FRAUD_CLR],
        text=ks_df["KS_Statistic"].apply(lambda x: f"{x:.3f}"),
        height=320,
        labels={"KS_Statistic": "KS Statistic (higher = better separation)"}
    )
    fig_ks.update_traces(textposition="outside", textfont_size=9)
    fig_ks.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
    fig_ks.add_hline(y=0.5, line_dash="dash", line_color="white",
                     annotation_text="KS=0.5 threshold")
    st.plotly_chart(fig_ks, use_container_width=True)
    st.markdown(
        '<div class="insight-box">🔍 <b>KS Statistic measures how differently '
        'fraud and legit distributions are shaped for each feature.</b> '
        'KS > 0.5 means strong separation — V17, V14, V12 should be top-weighted '
        'in any fraud scoring model.</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Feature explorer ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">INTERACTIVE FEATURE EXPLORER</p>',
                unsafe_allow_html=True)
    col_sel = st.selectbox(
        "Select feature to inspect:",
        options=ks_df["Feature"].tolist(),
        index=0,
        help="Features sorted by fraud-separation power (KS statistic)"
    )

    row = ks_df[ks_df["Feature"] == col_sel].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("KS Statistic",   f"{row['KS_Statistic']:.4f}")
    m2.metric("Correlation w/ Class", f"{row['Correlation']:.4f}")
    m3.metric("Fraud Mean",     f"{fraudf[col_sel].mean():.4f}")
    m4.metric("Legit Mean",     f"{legitf[col_sel].mean():.4f}")

    c1, c2 = st.columns(2)
    with c1:
        clip_f = st.slider(f"Clip {col_sel} at ±", 3, 20, 10)
        kde_fig = go.Figure()
        for subset, label, color in [
            (legitf[col_sel].clip(-clip_f, clip_f), "Legit", LEGIT_CLR),
            (fraudf[col_sel].clip(-clip_f, clip_f), "Fraud", FRAUD_CLR)
        ]:
            try:
                kde3 = gaussian_kde(subset, bw_method=0.3)
                xr3  = np.linspace(-clip_f, clip_f, 300)
                kde_fig.add_trace(go.Scatter(
                    x=xr3, y=kde3(xr3), name=label,
                    fill="tozeroy", fillcolor=color+"40",
                    line=dict(color=color, width=2.5)
                ))
            except Exception:
                pass
        kde_fig.update_layout(
            xaxis_title=col_sel, yaxis_title="Density",
            height=320, margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(kde_fig, use_container_width=True)

    with c2:
        fig_violin = go.Figure()
        fig_violin.add_trace(go.Violin(
            y=legitf[col_sel].clip(-clip_f, clip_f), name="Legit",
            fillcolor=LEGIT_CLR, line_color=LEGIT_CLR,
            meanline_visible=True, box_visible=True, opacity=0.7
        ))
        fig_violin.add_trace(go.Violin(
            y=fraudf[col_sel].clip(-clip_f, clip_f), name="Fraud",
            fillcolor=FRAUD_CLR, line_color=FRAUD_CLR,
            meanline_visible=True, box_visible=True, opacity=0.7
        ))
        fig_violin.update_layout(
            yaxis_title=col_sel, height=320,
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.05),
            violinmode="overlay"
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    st.divider()

    # ── Correlation heatmap ──────────────────────────────────────────────────
    st.markdown('<p class="section-title">FULL FEATURE CORRELATION WITH FRAUD CLASS</p>',
                unsafe_allow_html=True)
    corr_vals = ks_df.set_index("Feature")[["Correlation"]].sort_values("Correlation")
    corr_vals["Color"] = corr_vals["Correlation"].apply(
        lambda x: FRAUD_CLR if x < 0 else LEGIT_CLR)
    fig_corr = go.Figure(go.Bar(
        x=corr_vals["Correlation"], y=corr_vals.index,
        orientation="h",
        marker_color=corr_vals["Color"].tolist(),
        text=corr_vals["Correlation"].apply(lambda x: f"{x:.3f}"),
        textposition="outside"
    ))
    fig_corr.add_vline(x=0, line_color="white", line_width=1)
    fig_corr.update_layout(
        xaxis_title="Pearson Correlation with Class",
        height=520, margin=dict(t=10, b=10),
        xaxis=dict(range=[-0.45, 0.45])
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(
        '<div class="insight-box">📊 <b>Red bars = negatively correlated with fraud '
        '(lower values → more fraud risk). Blue = positive correlation.</b> '
        'V17 and V14 are the most powerful predictors in either direction.</div>',
        unsafe_allow_html=True
    )

    # ── Feature stats table ───────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">FULL FEATURE STATISTICS TABLE</p>',
                unsafe_allow_html=True)
    feat_stats = []
    for col in v_cols:
        feat_stats.append({
            "Feature": col,
            "KS Statistic": ks_df[ks_df["Feature"]==col]["KS_Statistic"].values[0],
            "Correlation": ks_df[ks_df["Feature"]==col]["Correlation"].values[0],
            "Fraud Mean": round(fraudf[col].mean(), 4),
            "Legit Mean": round(legitf[col].mean(), 4),
            "Fraud Std":  round(fraudf[col].std(), 4),
            "Legit Std":  round(legitf[col].std(), 4),
        })
    feat_df = pd.DataFrame(feat_stats).sort_values("KS Statistic", ascending=False)
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LIVE TRANSACTION MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Live Transaction Monitor":
    st.title("🚨 Live Transaction Monitor")
    st.caption("Flagged transactions with rule-based fraud confidence scores.")
    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    score_thresh = c1.slider("🎚️ Fraud Score Threshold", 0.0, 1.0, 0.60, step=0.01)
    show_mode    = c2.radio("Show Transactions", ["🚨 Flagged Only", "🔀 All", "✅ Legit Only"],
                            horizontal=True)
    n_display    = c3.selectbox("Rows to Display", [25, 50, 100, 250, 500], index=1)

    st.divider()

    # ── Summary alerts ────────────────────────────────────────────────────────
    flagged  = dff[dff["fraud_score"] >= score_thresh]
    true_pos = flagged[flagged["Class"] == 1]
    false_pos= flagged[flagged["Class"] == 0]

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("🚨 Flagged Transactions",  f"{len(flagged):,}")
    a2.metric("✅ True Fraud Caught",     f"{len(true_pos):,}",
              delta=f"Recall: {len(true_pos)/max(len(fraudf),1)*100:.1f}%")
    a3.metric("⚠️ False Positives",       f"{len(false_pos):,}",
              delta=f"Precision: {len(true_pos)/max(len(flagged),1)*100:.1f}%",
              delta_color="inverse")
    a4.metric("💰 Exposure Caught ($)",   f"${true_pos['Amount'].sum():,.0f}")
    a5.metric("💸 Exposure Missed ($)",
              f"${fraudf[~fraudf.index.isin(true_pos.index)]['Amount'].sum():,.0f}",
              delta_color="inverse")

    st.divider()

    # ── Score distribution ────────────────────────────────────────────────────
    st.markdown('<p class="section-title">FRAUD SCORE DISTRIBUTION</p>',
                unsafe_allow_html=True)
    fig_score = go.Figure()
    fig_score.add_trace(go.Histogram(
        x=dff[dff["Class"]==0]["fraud_score"], name="Legit",
        marker_color=LEGIT_CLR, opacity=0.7, nbinsx=60
    ))
    fig_score.add_trace(go.Histogram(
        x=dff[dff["Class"]==1]["fraud_score"], name="Fraud",
        marker_color=FRAUD_CLR, opacity=0.9, nbinsx=60
    ))
    fig_score.add_vline(x=score_thresh, line_dash="dash", line_color="white",
                        annotation_text=f"Threshold: {score_thresh:.2f}",
                        annotation_font_color="white")
    fig_score.update_layout(
        barmode="overlay", height=280,
        xaxis_title="Fraud Score (0=safe, 1=high risk)",
        yaxis_title="Count",
        margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig_score, use_container_width=True)
    st.markdown(
        f'<div class="insight-box">🎯 At threshold <b>{score_thresh:.2f}</b>: '
        f'catching <b>{len(true_pos)}</b> of {len(fraudf)} fraud cases '
        f'({len(true_pos)/max(len(fraudf),1)*100:.1f}% recall) '
        f'with <b>{len(false_pos)}</b> false positives '
        f'(precision: {len(true_pos)/max(len(flagged),1)*100:.1f}%). '
        f'Lower the threshold to catch more fraud; raise it to reduce false positives.</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Transaction table ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">TRANSACTION DETAIL TABLE</p>',
                unsafe_allow_html=True)

    display_cols = ["Time", "Amount", "Hour", "Day_label",
                    "Amount_band", "fraud_score", "Class"]

    if show_mode == "🚨 Flagged Only":
        tbl_df = flagged[display_cols].sort_values("fraud_score", ascending=False)
    elif show_mode == "✅ Legit Only":
        tbl_df = dff[dff["Class"]==0][display_cols].sort_values("fraud_score", ascending=False)
    else:
        tbl_df = dff[display_cols].sort_values("fraud_score", ascending=False)

    tbl_show = tbl_df.head(n_display).copy()
    tbl_show["fraud_score"] = tbl_show["fraud_score"].round(4)
    tbl_show["Class_label"] = tbl_show["Class"].map({0: "LEGIT", 1: "FRAUD"})

    st.dataframe(
        tbl_show.drop(columns=["Class"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "fraud_score": st.column_config.ProgressColumn(
                "Fraud Score",
                min_value=0, max_value=1,
                format="%.4f",
            ),
            "Amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
            "Class_label": st.column_config.TextColumn("Label"),
            "Hour": st.column_config.NumberColumn("Hour"),
        }
    )

    st.divider()

    # ── Scatter: Score vs Amount ──────────────────────────────────────────────
    st.markdown('<p class="section-title">FRAUD SCORE vs TRANSACTION AMOUNT (SCATTER)</p>',
                unsafe_allow_html=True)
    scatter_df = dff.copy()
    scatter_df["Label"] = scatter_df["Class"].map({0:"Legit", 1:"Fraud"})
    scatter_df = scatter_df.sample(min(5000, len(scatter_df)), random_state=42)

    fig_scatter = px.scatter(
        scatter_df,
        x="Amount", y="fraud_score",
        color="Label",
        color_discrete_map={"Legit": LEGIT_CLR, "Fraud": FRAUD_CLR},
        opacity=0.5,
        hover_data=["Hour","Day_label","Amount_band"],
        labels={"fraud_score": "Fraud Score", "Amount": "Transaction Amount ($)"},
        height=380,
    )
    fig_scatter.add_hline(y=score_thresh, line_dash="dash", line_color="white",
                          annotation_text="Threshold")
    fig_scatter.update_layout(margin=dict(t=10, b=10),
                              legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown(
        '<div class="insight-box">📍 <b>Fraud cases cluster at low amounts with high scores.</b> '
        'Notice how genuine fraud (red dots) tends to appear at smaller amounts but with '
        'elevated fraud scores — driven primarily by extreme V17/V14 values, not amount size.</div>',
        unsafe_allow_html=True
    )
