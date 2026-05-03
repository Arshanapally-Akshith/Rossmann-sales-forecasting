# ============================================================
# Rossmann Supply Chain Demand Forecasting — Streamlit App
# ============================================================
# Input files needed in same folder as app.py:
#   cleaned_rossmann.csv, sql_features.csv,
#   reconciled_predictions.csv, weekly_performance.csv,
#   top_5_stockout_risk.csv, store.csv, xgboost_model.pkl
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
import gdown
import os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rossmann Supply Chain Intelligence",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #1f2937; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #6b7280; margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #1f2937; margin: 1rem 0 0.5rem;
        border-left: 4px solid #3b82f6; padding-left: 0.6rem;
    }
    .risk-high   { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .risk-medium { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .risk-low    { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────

def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# =============================
# DOWNLOAD FILES FROM DRIVE
# =============================

if "files_downloaded" not in st.session_state:
    download_file("1P3msqhjUgdujfQytcdraDL9PZQ17BYjI", "reconciled_predictions.csv")

    download_file("1vv2Ue6WL2C1fe0mHgIhY-ZGfAvuzuld8", "cleaned_rossmann.csv")

    download_file("1ISSPx2z5cNvlubJZa6gEWKLKLODYNoqW", "sql_features.csv")
    
    st.session_state["files_downloaded"] = True


@st.cache_data
def load_reconciled():
    df = pd.read_csv("reconciled_predictions.csv", parse_dates=["Date"])
    df["Year"]    = df["Date"].dt.year
    df["Month"]   = df["Date"].dt.month
    # Week_dt is a true datetime column — used for add_vline with epoch ms (FIX 4)
    df["Week_dt"] = df["Date"] - pd.to_timedelta(df["Date"].dt.dayofweek, unit="d")
    df["Period"]  = df["Date"].apply(lambda d: "Test" if d >= pd.Timestamp("2015-06-01") else "Train")
    df["error"]     = df["Sales"] - df["reconciled_pred"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["abs_error"] / df["Sales"].clip(lower=1)
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = df["pct_error"] > df["pct_error"].quantile(0.995)
    return df

@st.cache_data
def load_cleaned():
    return pd.read_csv("cleaned_rossmann.csv", parse_dates=["Date"])

@st.cache_data
def load_sql():
    # NB2 always writes Store + Date into sql_features.csv
    return pd.read_csv("sql_features.csv", parse_dates=["Date"])

@st.cache_data
def load_store():
    return pd.read_csv("data/raw/store.csv")

@st.cache_data
def load_weekly():
    df = pd.read_csv("data/outputs/weekly_performance.csv")
    df["Week"] = df["Week"].astype(str)   # string axis — safe for add_vrect (FIX 5)
    return df

@st.cache_data
def load_risk():
    df = pd.read_csv("data/outputs/top_5_stockout_risk.csv")
    df["Risk"] = df["Anomaly_Count_Last_4_Weeks"].apply(
        lambda x: "🔴 HIGH" if x >= 5 else ("🟡 MEDIUM" if x >= 3 else "🟢 LOW")
    )
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load("models/xgboost_model.pkl")
    except Exception:
        return None

# Load all data
with st.spinner("Loading data..."):
    df_recon  = load_reconciled()
    df_clean  = load_cleaned()
    df_sql    = load_sql()
    df_store  = load_store()
    df_weekly = load_weekly()
    df_risk   = load_risk()
    model     = load_model()

df_clean["Date"] = pd.to_datetime(df_clean["Date"])
df_sql["Date"]   = pd.to_datetime(df_sql["Date"])
df_recon["Date"] = pd.to_datetime(df_recon["Date"])

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Rossmann-Supply%20Chain-blue", use_column_width=True)
    st.markdown("### 🏪 Rossmann Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Executive Overview",
         "📈 Demand Patterns",
         "🤖 Model Performance",
         "🚨 Anomaly Intelligence",
         "📉 Backtesting"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Project Metrics**")
    st.metric("Ensemble RMSPE", "1.038%")
    st.metric("Total Stores",   "1,115")
    st.metric("Total Rows",     "844,338")
    st.metric("Anomalies",      "32,618")

    st.markdown("---")
    st.markdown(
        "📓 [GitHub Repo](https://github.com/Arshanapally-Akshith)  \n"
        "👤 Arshanapally Akshith"
    )

# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────
def kpi(col, label, value, delta=None, delta_color="normal"):
    col.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# ════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.markdown('<div class="main-header">📊 Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Total sales performance across all 1,115 Rossmann stores in Germany (2013–2015)</div>', unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    sel_year  = col_f1.multiselect("Year", [2013, 2014, 2015], default=[2013, 2014, 2015])
    sel_type  = col_f2.multiselect(
        "Store Type", ["a", "b", "c", "d"], default=["a", "b", "c", "d"],
        format_func=lambda x: f"Type {x.upper()}"
    )
    sel_promo = col_f3.selectbox("Promo Filter", ["All", "Promo Days", "Non-Promo Days"])

    filt = df_clean[
        df_clean["Date"].dt.year.isin(sel_year) &
        df_clean["StoreType"].isin(sel_type)
    ].copy()
    if sel_promo == "Promo Days":
        filt = filt[filt["Promo"] == 1]
    elif sel_promo == "Non-Promo Days":
        filt = filt[filt["Promo"] == 0]

    st.markdown("---")

    # ── KPI Cards ────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Total Sales",        f"€{filt['Sales'].sum() / 1e9:.2f}B")
    kpi(k2, "Avg Daily Sales",    f"€{filt['Sales'].mean():,.0f}")
    kpi(k3, "Stores Active",      str(filt["Store"].nunique()))
    kpi(k4, "Total Transactions", f"{len(filt):,}")

    st.markdown("---")

    # ── Monthly Sales Trend ───────────────────────────────────
    st.markdown('<div class="section-title">Monthly Total Sales Trend</div>', unsafe_allow_html=True)
    monthly              = filt.copy()
    monthly["YearMonth"] = monthly["Date"].dt.to_period("M").astype(str)
    monthly_agg          = monthly.groupby("YearMonth")["Sales"].sum().reset_index()

    fig1 = px.line(
        monthly_agg, x="YearMonth", y="Sales",
        labels={"Sales": "Total Sales (€)", "YearMonth": "Month"},
        color_discrete_sequence=["#3b82f6"]
    )
    fig1.update_traces(line_width=2.5)
    fig1.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_tickangle=-45, hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── By StoreType + Promo Effect ──────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Sales by Store Type</div>', unsafe_allow_html=True)
        type_agg          = filt.groupby("StoreType")["Sales"].sum().reset_index()
        type_agg["Label"] = type_agg["StoreType"].apply(lambda x: f"Type {x.upper()}")
        fig2 = px.bar(
            type_agg, x="Label", y="Sales",
            color="Label",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"Sales": "Total Sales (€)", "Label": "Store Type"}
        )
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Promo vs Non-Promo Avg Sales</div>', unsafe_allow_html=True)
        promo_agg = filt.groupby("Promo")["Sales"].mean().reset_index()

        promo_agg["Label"] = promo_agg["Promo"].map({0: "No Promo", 1: "Promo"})
        fig3 = px.bar(
            promo_agg, x="Label", y="Sales",
            color="Label",
            color_discrete_map={"No Promo": "#94a3b8", "Promo": "#22c55e"},
            labels={"Sales": "Avg Daily Sales (€)", "Label": ""}
        )
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        # FIX 1: guard both groups exist before indexing — crashes when promo filter hides one group
        has_promo_0 = 0 in promo_agg["Promo"].values
        has_promo_1 = 1 in promo_agg["Promo"].values
        if has_promo_0 and has_promo_1:
            v1         = promo_agg[promo_agg["Promo"] == 1]["Sales"].values[0]
            v0         = promo_agg[promo_agg["Promo"] == 0]["Sales"].values[0]
            promo_lift = (v1 / v0 - 1) * 100
            st.success(f"📈 Promo Lift: **+{promo_lift:.1f}%** average sales uplift on promo days")
        else:
            st.info("Select 'All' in Promo Filter to see the promo lift calculation.")

    # ── DayOfWeek Heatmap ────────────────────────────────────
    st.markdown('<div class="section-title">Avg Sales by Day of Week × Month (Heatmap)</div>', unsafe_allow_html=True)
    dow_map     = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    heat        = filt.copy()
    heat["DayName"]   = heat["DayOfWeek"].map(dow_map)
    heat["MonthName"] = heat["Date"].dt.strftime("%b")
    heat_agg    = heat.groupby(["DayName", "MonthName"])["Sales"].mean().reset_index()
    heat_pivot  = heat_agg.pivot(index="DayName", columns="MonthName", values="Sales")
    day_order   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heat_pivot  = heat_pivot.reindex([d for d in day_order if d in heat_pivot.index])
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heat_pivot  = heat_pivot[[m for m in month_order if m in heat_pivot.columns]]

    fig4 = px.imshow(
        heat_pivot,
        color_continuous_scale="Blues",
        labels={"color": "Avg Sales (€)"},
        aspect="auto"
    )
    fig4.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — DEMAND PATTERNS
# ════════════════════════════════════════════════════════════
elif page == "📈 Demand Patterns":
    st.markdown('<div class="main-header">📈 Demand Patterns & SQL Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rolling averages and deviation signals from DuckDB window functions</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([1, 3])
    store_ids = sorted(df_sql["Store"].unique())
    sel_store = col_s1.selectbox("Select Store", store_ids, index=0)

    store_sql   = df_sql[df_sql["Store"] == sel_store].sort_values("Date")
    store_clean = df_clean[df_clean["Store"] == sel_store].sort_values("Date")

    # FIX 2: must merge store_clean (full df, keeps Store col) not store_clean[["Date","Sales"]]
    # Subsetting to ["Date","Sales"] drops Store before joining on it → KeyError: 'Store'
    merged = store_clean.merge(store_sql, on=["Store", "Date"], how="left")

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Avg Daily Sales",         f"€{store_clean['Sales'].mean():,.0f}")
    kpi(k2, "Avg 4W Rolling Avg",      f"€{store_sql['avg_4w'].mean():,.0f}")
    kpi(k3, "Avg Deviation (dev_4w)",  f"€{store_sql['dev_4w'].mean():,.0f}")
    kpi(k4, "Avg Volatility (std_4w)", f"€{store_sql['std_4w'].mean():,.0f}")
    st.markdown("---")

    st.markdown('<div class="section-title">Actual Sales vs 4-Week Rolling Average</div>', unsafe_allow_html=True)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=merged["Date"], y=merged["Sales"],
        name="Actual Sales", line=dict(color="#3b82f6", width=2)
    ))
    if "avg_4w" in merged.columns:
        fig5.add_trace(go.Scatter(
            x=merged["Date"], y=merged["avg_4w"],
            name="4W Rolling Avg", line=dict(color="#f97316", width=2, dash="dash")
        ))
    fig5.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        xaxis_title="Date", yaxis_title="Sales (€)"
    )
    st.plotly_chart(fig5, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Weekly Deviation from Rolling Baseline</div>', unsafe_allow_html=True)
        if "dev_4w" in store_sql.columns:
            dev_weekly         = store_sql.copy()
            dev_weekly["Week"] = dev_weekly["Date"].dt.to_period("M").astype(str)
            dev_monthly        = dev_weekly.groupby("Week")["dev_4w"].mean().reset_index()
            colours            = ["#22c55e" if v >= 0 else "#ef4444" for v in dev_monthly["dev_4w"]]
            fig6 = go.Figure(go.Bar(
                x=dev_monthly["Week"], y=dev_monthly["dev_4w"],
                marker_color=colours, name="Avg Monthly Deviation"
            ))
            fig6.add_hline(y=0, line_color="black", line_width=1)
            fig6.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_tickangle=-45, yaxis_title="Deviation (€)"
            )
            st.plotly_chart(fig6, use_container_width=True)
            st.caption("🟢 Green = above rolling baseline | 🔴 Red = below baseline (supply risk)")

    with col_r:
        st.markdown('<div class="section-title">Competition Distance vs Avg Sales (All Stores)</div>', unsafe_allow_html=True)
        comp_df = df_clean.groupby("Store").agg(
            AvgSales=("Sales", "mean"),
            CompDist=("CompetitionDistance", "first"),
            StoreType=("StoreType", "first")
        ).reset_index()
        fig7 = px.scatter(
            comp_df, x="CompDist", y="AvgSales",
            color="StoreType",
            labels={"CompDist": "Competition Distance", "AvgSales": "Avg Daily Sales (€)"},
            color_discrete_sequence=px.colors.qualitative.Set1,
            trendline="ols", trendline_scope="overall"
        )
        fig7.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig7, use_container_width=True)
        st.caption("Trend line shows farther competitors → higher sales on average")


# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">XGBoost + CatBoost ensemble — evaluated on unseen test data (Jun–Jul 2015)</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "XGBoost RMSPE",  "1.122%", delta="-0.040% from baseline", delta_color="inverse")
    kpi(k2, "CatBoost RMSPE", "1.121%")
    kpi(k3, "Ensemble RMSPE", "1.038%", delta="Best model", delta_color="off")
    kpi(k4, "Baseline RMSPE", "1.162%")

    st.markdown("---")

    st.markdown('<div class="section-title">Feature Importance (XGBoost) — avg_4w dominates at 52%</div>', unsafe_allow_html=True)
    fi_data = {
        "feature":    ["avg_4w", "dev_4w", "median_4w", "Promo", "lag_365",
                       "DayOfWeek", "StoreType", "SchoolHoliday", "std_4w", "avg_1w"],
        "importance": [0.5224, 0.2648, 0.1093, 0.0614, 0.0084,
                       0.0077, 0.0063, 0.0026, 0.0024, 0.0023]
    }
    fi_df = pd.DataFrame(fi_data).sort_values("importance")
    fig8  = go.Figure(go.Bar(
        x=fi_df["importance"], y=fi_df["feature"],
        orientation="h",
        marker_color="#3b82f6",
        text=[f"{v * 100:.1f}%" for v in fi_df["importance"]],
        textposition="outside"
    ))
    fig8.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Importance Score", yaxis_title=""
    )
    st.plotly_chart(fig8, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Actual vs Predicted — Store Selector</div>', unsafe_allow_html=True)
        store_ids2 = sorted(df_recon["Store"].unique())
        sel_s2     = st.selectbox("Store", store_ids2, key="model_store")
        s2_df      = df_recon[df_recon["Store"] == sel_s2].sort_values("Date").tail(90)
        test_start = pd.Timestamp("2015-06-01")

        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(
            x=s2_df["Date"], y=s2_df["Sales"],
            name="Actual", line=dict(color="#ef4444", width=2.5)
        ))
        fig9.add_trace(go.Scatter(
            x=s2_df["Date"], y=s2_df["reconciled_pred"],
            name="Predicted", line=dict(color="#3b82f6", width=1.5, dash="dash")
        ))

        # FIX 3: add_vrect on datetime axis needs pd.Timestamp objects — NOT strings.
        # x0="2015-06-01" (str) vs x1=Timestamp causes plotly's min([str,Timestamp]) to raise TypeError.
        # Guard: only draw rect when test period overlaps the selected store's window.
        if not s2_df.empty and s2_df["Date"].max() >= test_start:
            rect_x0 = max(test_start, s2_df["Date"].min())   # pd.Timestamp
            rect_x1 = s2_df["Date"].max()                     # pd.Timestamp
            fig9.add_vrect(
                x0=rect_x0, x1=rect_x1,
                fillcolor="#fef3c7", opacity=0.3, line_width=0,
                annotation_text="Test Period", annotation_position="top left"
            )

        fig9.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified", legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig9, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">RMSPE by Store Type (Test Period)</div>', unsafe_allow_html=True)
        rmspe_type = {
            "Store Type": ["Type A (0)", "Type B (1)", "Type C (2)", "Type D (3)"],
            "RMSPE %":    [1.311, 1.665, 0.637, 0.715]
        }
        rt_df      = pd.DataFrame(rmspe_type)
        colours_rt = ["#f97316", "#ef4444", "#22c55e", "#3b82f6"]
        fig10 = go.Figure(go.Bar(
            x=rt_df["Store Type"], y=rt_df["RMSPE %"],
            marker_color=colours_rt,
            text=[f"{v:.2f}%" for v in rt_df["RMSPE %"]],
            textposition="outside"
        ))
        fig10.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="RMSPE (%)", showlegend=False
        )
        st.plotly_chart(fig10, use_container_width=True)
        st.info("Type C and D stores show lowest error — Type B (only 17 stores) has highest variance")

    st.markdown('<div class="section-title">Model Comparison Summary</div>', unsafe_allow_html=True)
    comp_tbl = pd.DataFrame({
        "Model":  ["XGBoost Baseline", "XGBoost Tuned", "CatBoost", "Ensemble (Final)"],
        "RMSPE":  ["1.162%", "1.082%", "1.121%", "1.038%"],
        "Notes":  ["500 estimators, default params",
                   "Optuna 20-trial walk-forward tuning",
                   "Early stopping at iteration 998",
                   "Inverse-error weighted combination ✓"]
    })
    st.dataframe(comp_tbl, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — ANOMALY INTELLIGENCE
# ════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Intelligence":
    st.markdown('<div class="main-header">🚨 Anomaly Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Per-store rolling threshold anomaly detection — which stores need attention right now?</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    period_filter = col_f1.selectbox("Period", ["All", "Train", "Test"])
    year_filter   = col_f2.multiselect("Year", [2013, 2014, 2015], default=[2013, 2014, 2015])
    anom_only     = col_f3.checkbox("Show Anomalies Only", value=False)

    adf = df_recon.copy()
    if period_filter != "All":
        adf = adf[adf["Period"] == period_filter]
    adf = adf[adf["Year"].isin(year_filter)]
    if anom_only:
        adf = adf[adf["is_anomaly"] == True]

    total_anom  = int(df_recon["is_anomaly"].sum())
    anom_rate   = total_anom / len(df_recon) * 100
    top_store   = df_recon[df_recon["is_anomaly"] == True]["Store"].value_counts().idxmax()
    avg_err_pct = df_recon[df_recon["is_anomaly"] == True]["pct_error"].mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Total Anomalies",        f"{total_anom:,}")
    kpi(k2, "Anomaly Rate",           f"{anom_rate:.2f}%")
    kpi(k3, "Top Anomaly Store",      f"Store {top_store}")
    kpi(k4, "Avg Error on Anom Days", f"{avg_err_pct:.2f}%")
    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Top 10 Stores by Anomaly Count</div>', unsafe_allow_html=True)
        top10          = adf[adf["is_anomaly"] == True]["Store"].value_counts().head(10).reset_index()
        top10.columns  = ["Store", "Anomaly Count"]
        top10["Store"] = top10["Store"].astype(str)
        fig11 = go.Figure(go.Bar(
            x=top10["Anomaly Count"], y=top10["Store"],
            orientation="h",
            marker_color="#ef4444",
            text=top10["Anomaly Count"],
            textposition="outside"
        ))
        fig11.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(autorange="reversed"),
            xaxis_title="Anomaly Count"
        )
        st.plotly_chart(fig11, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">⚠️ Top 5 Stockout Risk Stores (Last 4 Weeks)</div>', unsafe_allow_html=True)
        st.dataframe(
            df_risk.rename(columns={
                "Store": "Store ID",
                "Anomaly_Count_Last_4_Weeks": "Anomalies (Last 4W)",
                "Risk": "Risk Level"
            }),
            use_container_width=True, hide_index=True, height=220
        )
        st.error("🔴 All top 5 stores are HIGH risk — immediate stock review required")

        st.markdown('<div class="section-title">Anomalies by Store Type</div>', unsafe_allow_html=True)
        type_anom          = adf[adf["is_anomaly"] == True]["StoreType"].value_counts().reset_index()
        type_anom.columns  = ["StoreType", "Count"]
        type_anom["Label"] = type_anom["StoreType"].apply(lambda x: f"Type {str(x).upper()}")
        fig12 = px.bar(
            type_anom, x="Label", y="Count", color="Label",
            color_discrete_sequence=["#f97316", "#ef4444", "#22c55e", "#3b82f6"],
            labels={"Count": "Anomaly Count", "Label": "Store Type"}, text="Count"
        )
        fig12.update_traces(textposition="outside")
        fig12.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig12, use_container_width=True)

    st.markdown('<div class="section-title">Anomaly Count per Week — Spike Detection (135 Weeks)</div>', unsafe_allow_html=True)

    weekly_anom = (
        adf[adf["is_anomaly"] == True]
        .groupby("Week_dt").size().reset_index(name="Anomalies")
    )

    fig13 = go.Figure()
    fig13.add_trace(go.Scatter(
        x=weekly_anom["Week_dt"], y=weekly_anom["Anomalies"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="#ef4444", width=2),
        name="Weekly Anomalies"
    ))

    # FIX 4: add_vline on a datetime axis — plotly calls float(sum(x)) internally.
    # Passing a string ("2015-06-01") raises TypeError: unsupported operand type for +: int and str.
    # Convert Timestamp to epoch milliseconds (int) which plotly's datetime axis accepts correctly.
    vline_ms = int(pd.Timestamp("2015-06-01").timestamp() * 1000)
    fig13.add_vline(
        x=vline_ms,
        line_color="#f97316", line_dash="dash",
        annotation_text="Test Period Start",
        annotation_position="top right"
    )

    fig13.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        xaxis_title="Week", yaxis_title="Anomaly Count"
    )
    st.plotly_chart(fig13, use_container_width=True)
    st.caption(
        "Early weeks (Jan 2013) show high anomalies — rolling threshold needs 28 days to calibrate. "
        "Test period (post Jun 2015) anomalies confirm the threshold generalises correctly to unseen data."
    )

    st.markdown('<div class="section-title">Store-Level Deep Dive</div>', unsafe_allow_html=True)
    sel_anom_store = st.selectbox("Select Store", sorted(df_recon["Store"].unique()), key="anom_store")
    s_df  = df_recon[df_recon["Store"] == sel_anom_store].sort_values("Date")
    anoms = s_df[s_df["is_anomaly"] == True]

    fig14 = go.Figure()
    fig14.add_trace(go.Scatter(
        x=s_df["Date"], y=s_df["Sales"],
        name="Actual", line=dict(color="#3b82f6", width=1.5)
    ))
    fig14.add_trace(go.Scatter(
        x=s_df["Date"], y=s_df["reconciled_pred"],
        name="Predicted", line=dict(color="#94a3b8", width=1, dash="dash")
    ))
    fig14.add_trace(go.Scatter(
        x=anoms["Date"], y=anoms["Sales"],
        mode="markers", name="Anomaly",
        marker=dict(color="#ef4444", size=7, symbol="triangle-up")
    ))
    fig14.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified", legend=dict(orientation="h", y=-0.25),
        xaxis_title="Date", yaxis_title="Sales (€)"
    )
    st.plotly_chart(fig14, use_container_width=True)
    st.caption(f"Store {sel_anom_store} — Red triangles = anomaly days flagged by the per-store rolling threshold")


# ════════════════════════════════════════════════════════════
# PAGE 5 — BACKTESTING
# ════════════════════════════════════════════════════════════
elif page == "📉 Backtesting":
    st.markdown('<div class="main-header">📉 Model Reliability — Backtesting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Expanding window backtesting — 5 splits across 2.5 years · Per-store weekly RMSE</div>', unsafe_allow_html=True)

    train_rmse = df_recon[df_recon["Period"] == "Train"].eval("(Sales - reconciled_pred)**2").mean() ** 0.5
    test_rmse  = df_recon[df_recon["Period"] == "Test"].eval("(Sales - reconciled_pred)**2").mean() ** 0.5

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Train Period RMSE",  f"€{train_rmse:.1f}")
    kpi(k2, "Test Period RMSE",   f"€{test_rmse:.1f}")
    kpi(k3, "Backtesting Splits", "5")
    kpi(k4, "Total Weeks Tested", "135")
    st.markdown("---")

    st.markdown('<div class="section-title">Weekly Average RMSE — Per-Store Average across 135 Weeks</div>', unsafe_allow_html=True)

    wdf        = df_weekly.copy()
    cutoff_idx = int(len(wdf) * 0.93)

    # Week column is a string (e.g. "2015-W23") — add_vrect with string x0/x1 works correctly
    # on a categorical/string axis. No epoch conversion needed here.
    test_w_start = wdf["Week"].iloc[cutoff_idx] if cutoff_idx < len(wdf) else wdf["Week"].iloc[-1]
    test_w_end   = wdf["Week"].iloc[-1]

    fig15 = go.Figure()
    fig15.add_trace(go.Scatter(
        x=wdf["Week"], y=wdf["RMSE"],
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
        name="Weekly Avg RMSE",
        hovertemplate="Week: %{x}<br>RMSE: €%{y:.1f}<extra></extra>"
    ))
    fig15.add_hline(
        y=wdf["RMSE"].mean(),
        line_dash="dash", line_color="#94a3b8",
        annotation_text=f"Mean RMSE €{wdf['RMSE'].mean():.1f}",
        annotation_position="bottom right"
    )
    # FIX 5: string x-axis (Week column) — add_vrect accepts string x0/x1 safely
    fig15.add_vrect(
        x0=test_w_start, x1=test_w_end,
        fillcolor="#fef3c7", opacity=0.4, line_width=0,
        annotation_text="Test Period", annotation_position="top left"
    )
    visible_ticks = [wdf["Week"].iloc[i] for i in range(0, len(wdf), 6)]
    fig15.update_xaxes(tickvals=visible_ticks, tickangle=-45)
    fig15.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        xaxis_title="Week", yaxis_title="Avg RMSE (€)",
        showlegend=False
    )
    st.plotly_chart(fig15, use_container_width=True)
    st.caption(
        "First weeks (Jan 2013) show higher RMSE — the rolling lag features need 28 days to stabilise. "
        "Test period RMSE is comparable to training RMSE, confirming the model did not overfit."
    )

    st.markdown('<div class="section-title">Expanding Window Split Design</div>', unsafe_allow_html=True)
    splits_df = pd.DataFrame({
        "Split":        ["Split 1", "Split 2", "Split 3", "Split 4", "Split 5"],
        "Train Period": ["Jan 2013 – Dec 2013", "Jan 2013 – Mar 2014",
                         "Jan 2013 – Jun 2014", "Jan 2013 – Sep 2014", "Jan 2013 – Dec 2014"],
        "Test Period":  ["Q1 2014", "Q2 2014", "Q3 2014", "Q4 2014", "H1 2015"],
        "Strategy":     ["Expanding"] * 5
    })
    st.dataframe(splits_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">RMSE Distribution Across All Stores (Test Period)</div>', unsafe_allow_html=True)
    store_rmse = (
        df_recon[df_recon["Period"] == "Test"]
        .groupby("Store")
        .apply(lambda x: np.sqrt(np.mean((x["Sales"] - x["reconciled_pred"]) ** 2)))
        .reset_index(name="RMSE")
    )
    fig16 = px.histogram(
        store_rmse, x="RMSE", nbins=60,
        color_discrete_sequence=["#3b82f6"],
        labels={"RMSE": "Store-Level RMSE (€)", "count": "Number of Stores"}
    )
    fig16.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig16, use_container_width=True)
    st.caption(
        f"Median store RMSE: €{store_rmse['RMSE'].median():.1f} | "
        f"Max: €{store_rmse['RMSE'].max():.1f} | "
        f"Stores with RMSE < €100: {(store_rmse['RMSE'] < 100).sum()} of 1,115"
    )

    st.markdown('<div class="section-title">Prediction Error Distribution (Test Period)</div>', unsafe_allow_html=True)
    test_df = df_recon[df_recon["Period"] == "Test"].copy()
    fig17 = px.histogram(
        test_df, x="error", nbins=100,
        color_discrete_sequence=["#6366f1"],
        labels={"error": "Prediction Error (Sales − Predicted)", "count": "Count"},
        range_x=[-2000, 2000]
    )
    # add_vline with x=0 (integer) on a numeric axis — always safe, no fix needed
    fig17.add_vline(
        x=0, line_color="#ef4444", line_dash="dash",
        annotation_text="Zero Error", annotation_position="top right"
    )
    fig17.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig17, use_container_width=True)
    st.caption(
        f"Mean error: €{test_df['error'].mean():.1f} | "
        f"Std: €{test_df['error'].std():.1f} — symmetric distribution confirms no systematic bias"
    )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:0.8rem;'>"
    "Rossmann Supply Chain Intelligence · Arshanapally Akshith · "
    "<a href='https://github.com/Arshanapally-Akshith' style='color:#9ca3af;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
