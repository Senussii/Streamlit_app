"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SUPPLY CHAIN INTELLIGENCE PLATFORM  v2.0                                    ║
║  End-to-End Predictive Analytics · Galaxy Schema DWH · ML-Powered            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Modules
  1. Executive Dashboard   — KPIs + revenue / margin overview
  2. Demand Forecasting    — LightGBM auto-regressive demand prediction
  3. Inventory Risk        — LightGBM stockout risk classifier
  4. Customer Intelligence — Churn predictor + RFM segmentation
  5. Supplier Analytics    — Supplier quality scoring + fulfillment heatmap
  6. Anomaly Detection     — Isolation Forest on financial transactions

v2 changes to app.py
  • Data loading moved before the sidebar so the status panel can show
    live record/SKU/customer counts on every page.
  • All six ML build calls wrapped in @st.cache_data so models train once
    per session; subsequent visits are instant.
  • Sidebar completely redesigned: animated brand header, live data-health
    panel, restyled radio nav (CSS-only, no JS), model performance metrics
    that populate as the user visits pages, and a pinned footer.
  • Model metrics stored in st.session_state.model_metrics after each
    training run and surfaced in the sidebar.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data_loader import (mart_sales, mart_inventory, mart_purchase,
                          mart_movement, mart_transaction, load_dimensions)
from ml_models   import (build_demand_forecast, build_stockout_classifier,
                          build_churn_predictor, build_supplier_scorer,
                          build_anomaly_detector, build_customer_segments)
import charts as ch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(_css_path):
    with open(_css_path) as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def metric_card(label, value, delta_str="", prefix="", suffix=""):
    delta_html = ""
    if delta_str:
        cls = "delta-pos" if delta_str.startswith("+") else "delta-neg"
        delta_html = f'<div class="metric-delta {cls}">{delta_str}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


def section(title, icon=""):
    st.markdown(f"""
    <div class="section-header"><h3>{icon} {title}</h3></div>
    """, unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = {}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached — runs once per session)
# Moved above the sidebar so the status panel has live counts.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_all_data():
    sale    = mart_sales()
    inv_raw = mart_inventory()
    pur     = mart_purchase()
    mv      = mart_movement()
    txn     = mart_transaction()
    dims    = load_dimensions()
    return sale, inv_raw, pur, mv, txn, dims


with st.spinner("Loading Supply Chain Data Warehouse…"):
    sale, inv_raw, pur, mv, txn, dims = get_all_data()


# ══════════════════════════════════════════════════════════════════════════════
# CACHED MODEL WRAPPERS
# Each model is trained once per unique set of inputs; subsequent page
# visits retrieve results from cache instantly.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _forecast(sale, horizon):
    return build_demand_forecast(sale, horizon_months=horizon)

@st.cache_data(show_spinner=False)
def _stockout(inv_raw, mv):
    return build_stockout_classifier(inv_raw, mv)

@st.cache_data(show_spinner=False)
def _churn(sale):
    return build_churn_predictor(sale)

@st.cache_data(show_spinner=False)
def _supplier(pur, supplier_dim):
    return build_supplier_scorer(pur, supplier_dim)

@st.cache_data(show_spinner=False)
def _anomaly(txn):
    return build_anomaly_detector(txn)

@st.cache_data(show_spinner=False)
def _segments(sale):
    return build_customer_segments(sale)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  v2
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Brand header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none"
                 xmlns="http://www.w3.org/2000/svg">
                <path d="M11 2L3 6.5V15.5L11 20L19 15.5V6.5L11 2Z"
                      stroke="#00D4FF" stroke-width="1.3" stroke-linejoin="round"/>
                <path d="M11 2L11 20" stroke="#7B2FBE" stroke-width="1" opacity="0.6"/>
                <path d="M3 6.5L19 6.5" stroke="#00C49A" stroke-width="1" opacity="0.5"/>
                <circle cx="11" cy="11" r="2.5" fill="#00D4FF" opacity="0.9"/>
            </svg>
        </div>
        <div class="sb-brand-text">
            <div class="sb-brand-name">SupplyIQ</div>
            <div class="sb-brand-tagline">Intelligence Platform</div>
        </div>
        <div class="sb-live-dot" title="Data warehouse connected"></div>
    </div>
    <div class="sb-divider"></div>
    """, unsafe_allow_html=True)

    # ── Live data-health panel ────────────────────────────────────────────────
    total_records = len(sale) + len(inv_raw) + len(pur) + len(mv) + len(txn)
    n_skus        = sale["Stock Item Key"].nunique() if "Stock Item Key" in sale.columns else 0
    n_customers   = sale["Customer Key"].nunique()   if "Customer Key"   in sale.columns else 0

    st.markdown(f"""
    <div class="sb-status">
        <div class="sb-status-head">
            <div class="sb-status-dot"></div>
            <span class="sb-status-label">Data Warehouse · Live</span>
        </div>
        <div class="sb-status-grid">
            <div class="sb-stat">
                <div class="sb-stat-val">{total_records/1e3:.0f}K</div>
                <div class="sb-stat-lbl">Records</div>
            </div>
            <div class="sb-stat">
                <div class="sb-stat-val">{n_skus}</div>
                <div class="sb-stat-lbl">SKUs</div>
            </div>
            <div class="sb-stat">
                <div class="sb-stat-val">{n_customers}</div>
                <div class="sb-stat-lbl">Customers</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section-label">Modules</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Dashboard",
            "📈 Demand Forecasting",
            "📦 Inventory Risk",
            "👥 Customer Intelligence",
            "🏭 Supplier Analytics",
            "🚨 Anomaly Detection",
        ],
        label_visibility="collapsed",
    )

    # ── Model performance panel (populates as user visits pages) ──────────────
    if st.session_state.model_metrics:
        st.markdown('<div class="sb-divider" style="margin-top:12px"></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Model Performance</div>',
                    unsafe_allow_html=True)

        rows_html = ""
        for k, (v, quality) in st.session_state.model_metrics.items():
            cls = {"good": "good", "warn": "warn", "": ""}[quality]
            rows_html += f"""
            <div class="sb-metric-row">
                <span class="sb-metric-name">{k}</span>
                <span class="sb-metric-val {cls}">{v}</span>
            </div>"""
        st.markdown(f'<div class="sb-metrics">{rows_html}</div>',
                    unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sb-footer">
        <div class="sb-footer-line">Galaxy Schema · FY 2013–2016</div>
        <div class="sb-footer-line">6 ML Engines · 14 Tables · v2.0</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Dashboard":
    st.title("Executive Supply Chain Dashboard")
    st.caption("Galaxy Schema · Real-time KPIs · FY2013-2016")

    total_rev    = sale["Total Excluding Tax"].sum()
    total_profit = sale["Profit"].sum()
    margin_pct   = total_profit / total_rev * 100 if total_rev else 0
    total_orders = len(sale)
    total_skus   = sale["Stock Item Key"].nunique()
    avg_order_v  = sale["Total Including Tax"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: metric_card("Total Revenue",   f"${total_rev/1e6:.1f}M")
    with c2: metric_card("Total Profit",    f"${total_profit/1e6:.1f}M")
    with c3: metric_card("Gross Margin",    f"{margin_pct:.1f}", suffix="%")
    with c4: metric_card("Total Orders",    f"{total_orders:,}")
    with c5: metric_card("Active SKUs",     f"{total_skus:,}")
    with c6: metric_card("Avg Order Value", f"${avg_order_v:.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        section("Revenue & Profit Trend", "📊")
        st.plotly_chart(ch.revenue_timeline(sale),
                        use_container_width=True, config={"displayModeBar": False})
    with col2:
        section("Category Performance", "🏷️")
        st.plotly_chart(ch.sales_by_category(sale),
                        use_container_width=True, config={"displayModeBar": False})

    col3, col4 = st.columns([3, 2])
    with col3:
        section("Revenue / COGS / Profit by Category", "💰")
        st.plotly_chart(ch.margin_waterfall(sale),
                        use_container_width=True, config={"displayModeBar": False})
    with col4:
        section("Top 10 Customers by Revenue", "🌟")
        top_cust = (sale.groupby("Customer")
                    .agg(Revenue=("Total Excluding Tax", "sum"),
                         Profit=("Profit", "sum"))
                    .reset_index()
                    .sort_values("Revenue", ascending=False)
                    .head(10))
        st.dataframe(
            top_cust.style.format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}"}),
            use_container_width=True, height=320)

    section("Revenue by Sales Territory", "🗺️")
    terr = (sale.groupby("Sales Territory")
            .agg(Revenue=("Total Excluding Tax", "sum"),
                 Orders=("Sale Key", "count"),
                 Profit=("Profit", "sum"))
            .reset_index().dropna(subset=["Sales Territory"])
            .sort_values("Revenue", ascending=False))
    fig_t = px.bar(terr, x="Sales Territory", y="Revenue",
                   color="Profit", color_continuous_scale="Viridis",
                   template="plotly_dark",
                   labels={"Revenue": "Revenue ($)", "Profit": "Profit ($)"})
    fig_t.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                        font_color="#E0E0E0",
                        coloraxis_colorbar=dict(tickfont=dict(color="#E0E0E0")))
    st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEMAND FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecasting":
    st.title("Demand Forecasting")
    st.caption("LightGBM / HistGradientBoosting · Auto-regressive · Per Stock Category")

    _, col_h = st.columns([3, 1])
    with col_h:
        horizon = st.slider("Forecast Horizon (months)", 1, 12, 3)

    with st.spinner("Training demand forecast model…"):
        info = _forecast(sale, horizon)

    # store metrics in sidebar
    q_mape = "good" if info["mape"] < 10 else "warn" if info["mape"] < 20 else ""
    q_r2   = "good" if info["r2"]   > 0.7 else "warn" if info["r2"] > 0.4 else ""
    st.session_state.model_metrics["Demand MAPE"]    = (f"{info['mape']:.1f}%",  q_mape)
    st.session_state.model_metrics["Demand R²"]      = (f"{info['r2']:.3f}",     q_r2)
    st.session_state.model_metrics["Demand CV-MAPE"] = (f"{info['cv_mape']:.1f}%", q_mape)

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("MAPE",        f"{info['mape']:.1f}",     suffix="%")
    with m2: metric_card("R² Score",    f"{info['r2']:.3f}")
    with m3: metric_card("CV MAPE",     f"{info['cv_mape']:.1f}",  suffix="%")
    with m4: metric_card("Horizon",     f"{horizon}",              suffix=" months")

    st.markdown("<br>", unsafe_allow_html=True)

    cats     = sorted(sale["Stock Category"].dropna().unique())
    selected = st.multiselect("Select Stock Categories to Forecast", cats, default=cats[:4])

    if selected:
        section("Demand Forecast vs Actuals", "📈")
        st.plotly_chart(ch.forecast_chart(info, selected), use_container_width=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            section("Feature Importance", "🔍")
            st.plotly_chart(ch.feat_importance_chart(
                info["feature_importance"], "Demand Model — Feature Importance"),
                use_container_width=True)
        with col_f2:
            section("Forecast Table", "📋")
            fc_show = (info["forecast_df"][info["forecast_df"]["Stock Category"].isin(selected)]
                       [["Stock Category", "Forecast Year", "Forecast Month", "Predicted_Qty"]]
                       .sort_values(["Stock Category", "Forecast Year", "Forecast Month"])
                       .rename(columns={"Predicted_Qty": "Forecasted Units"}))
            fc_show["Forecasted Units"] = fc_show["Forecasted Units"].round(0).astype(int)
            st.dataframe(fc_show, use_container_width=True, height=320)

    section("Actual vs Predicted (Test Set)", "🎯")
    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=info["test_true"], y=info["test_pred"],
        mode="markers", marker=dict(color="#00D4FF", opacity=0.5, size=5),
        name="Predictions"))
    mn = min(info["test_true"].min(), info["test_pred"].min())
    mx = max(info["test_true"].max(), info["test_pred"].max())
    fig_ap.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                mode="lines", line=dict(color="#FF6B35", dash="dot"),
                                name="Perfect Fit"))
    fig_ap.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font_color="#E0E0E0", xaxis_title="Actual Units", yaxis_title="Predicted Units",
        title=dict(text=f"Actual vs Predicted — R²={info['r2']:.3f}",
                   font=dict(color="#00D4FF")),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_ap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — INVENTORY RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Inventory Risk":
    st.title("Inventory Risk Intelligence")
    st.caption("LightGBM Classifier · Corrected Monthly Velocity · Stockout Alerts")

    with st.spinner("Running stockout risk model…"):
        inv_info = _stockout(inv_raw, mv)

    inv_df = inv_info["df"]

    high      = (inv_df["Predicted_Risk_Name"] == "HIGH").sum()
    medium    = (inv_df["Predicted_Risk_Name"] == "MEDIUM").sum()
    stock_val = inv_df["Stock Value"].sum()

    if inv_info["auc"]:
        q = "good" if inv_info["auc"] > 0.85 else "warn"
        st.session_state.model_metrics["Stockout AUC"] = (f"{inv_info['auc']:.3f}", q)

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Total SKUs",        f"{len(inv_df):,}")
    with k2: metric_card("HIGH Risk SKUs",    f"{high}",   delta_str=f"+{high} need action")
    with k3: metric_card("MEDIUM Risk SKUs",  f"{medium}")
    with k4: metric_card("Total Stock Value", f"${stock_val/1e6:.1f}M")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        section("Inventory Risk Matrix", "🎯")
        if "Monthly_Velocity" not in inv_df.columns:
            inv_df["Monthly_Velocity"] = 0
        st.plotly_chart(ch.inventory_risk_scatter(inv_df), use_container_width=True)
    with col2:
        section("Risk Distribution", "📊")
        risk_cnt = inv_df["Predicted_Risk_Name"].value_counts().reset_index()
        risk_cnt.columns = ["Risk", "Count"]
        fig_pie = px.pie(risk_cnt, names="Risk", values="Count", color="Risk",
                         color_discrete_map={"HIGH": "#FF4C6E", "MEDIUM": "#FFD700",
                                             "LOW": "#00C49A"},
                         template="plotly_dark")
        fig_pie.update_layout(paper_bgcolor="#0E1117", font_color="#E0E0E0",
                              legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_pie, use_container_width=True)

    section("HIGH Risk SKUs — Immediate Reorder Needed", "⚠️")
    high_df = (inv_df[inv_df["Predicted_Risk_Name"] == "HIGH"]
               [["Stock Item", "Stock Category", "Subcategory", "Quantity On Hand",
                 "Reorder Level", "Target Stock Level", "Stock Value",
                 "Lead Time Days", "Availability"]]
               .sort_values("Quantity On Hand"))
    st.dataframe(high_df.style.format({
        "Quantity On Hand":   "{:,.0f}",
        "Reorder Level":      "{:,.0f}",
        "Target Stock Level": "{:,.0f}",
        "Stock Value":        "${:,.2f}",
    }), use_container_width=True, height=300)

    col3, col4 = st.columns(2)
    with col3:
        section("Model Feature Importance", "🔍")
        st.plotly_chart(ch.feat_importance_chart(
            inv_info["feature_importance"], "Stockout Classifier — Feature Importance"),
            use_container_width=True)
    with col4:
        section("Classification Report", "📋")
        if inv_info["report"]:
            rpt = pd.DataFrame(inv_info["report"]).T.drop(columns=["support"], errors="ignore")
            st.dataframe(rpt.style.format("{:.2f}"), use_container_width=True)
        if inv_info["auc"]:
            st.metric("ROC-AUC (OvR)", f"{inv_info['auc']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Intelligence":
    st.title("Customer Intelligence")
    st.caption("Churn Predictor · RFM Segmentation · KMeans Clustering")

    tab1, tab2 = st.tabs(["🔮 Churn Prediction", "🗺️ RFM Segmentation"])

    with tab1:
        with st.spinner("Training churn prediction model…"):
            churn_info = _churn(sale)

        rfm          = churn_info["rfm"]
        churn_count  = rfm["Churn_Pred"].sum()
        active_count = (rfm["Churn_Pred"] == 0).sum()
        churn_rate   = rfm["Churn_Pred"].mean() * 100

        if churn_info["auc"]:
            q = "good" if churn_info["auc"] > 0.80 else "warn"
            st.session_state.model_metrics["Churn AUC"] = (f"{churn_info['auc']:.3f}", q)

        k1, k2, k3, k4 = st.columns(4)
        with k1: metric_card("Total Customers",   f"{len(rfm):,}")
        with k2: metric_card("At-Risk Customers", f"{churn_count:,}",
                              delta_str=f"⚠ {churn_rate:.1f}% churn rate")
        with k3: metric_card("Active Customers",  f"{active_count:,}")
        with k4: metric_card("ROC-AUC",
                              f"{churn_info['auc']:.3f}" if churn_info["auc"] else "N/A")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            section("Churn Probability Distribution", "📊")
            st.plotly_chart(ch.churn_dist(rfm), use_container_width=True)
        with col2:
            section("Feature Importance", "🔍")
            st.plotly_chart(ch.feat_importance_chart(
                churn_info["feature_importance"], "Churn Model — Feature Importance"),
                use_container_width=True)

        section("Top At-Risk Customers", "🚨")
        at_risk = (rfm[rfm["Churn_Pred"] == 1]
                   .sort_values("Churn_Prob", ascending=False)
                   .head(50)
                   [["Customer Key", "Recency", "Frequency", "Monetary",
                     "Avg_Margin", "Churn_Prob"]]
                   .rename(columns={"Monetary": "Total Revenue ($)",
                                    "Churn_Prob": "Churn Probability"}))
        at_risk["Churn Probability"] = (at_risk["Churn Probability"] * 100).round(1)
        at_risk["Total Revenue ($)"] = at_risk["Total Revenue ($)"].round(0)
        st.dataframe(at_risk.style.format({
            "Total Revenue ($)": "${:,.0f}",
            "Churn Probability": "{:.1f}%",
            "Avg_Margin":        "{:.1f}%",
        }).background_gradient(subset=["Churn Probability"], cmap="RdYlGn_r"),
            use_container_width=True, height=300)

    with tab2:
        with st.spinner("Segmenting customers via KMeans…"):
            seg_info = _segments(sale)

        rfm_seg = seg_info["rfm"]
        section("Customer RFM Segmentation (3D)", "🌐")
        st.plotly_chart(ch.rfm_3d(rfm_seg), use_container_width=True)

        section("Segment Summary", "📋")
        seg_sum = (rfm_seg.groupby("Segment Name")
                   .agg(Count=("Customer Key", "count"),
                        Avg_Recency=("Recency", "mean"),
                        Avg_Frequency=("Frequency", "mean"),
                        Avg_Monetary=("Monetary", "mean"))
                   .reset_index()
                   .sort_values("Avg_Monetary", ascending=False))
        st.dataframe(seg_sum.style.format({
            "Avg_Recency":   "{:.0f} days",
            "Avg_Frequency": "{:.0f}",
            "Avg_Monetary":  "${:,.0f}",
        }), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SUPPLIER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Supplier Analytics":
    st.title("Supplier Analytics & Quality Scoring")
    st.caption("Random Forest · Hold-out R² · Fulfillment Heatmap · Benchmarking")

    with st.spinner("Scoring suppliers…"):
        sup_info = _supplier(pur, dims["supplier"])

    sup_df     = sup_info["df"]
    avg_score  = sup_df["Quality_Score"].mean()
    top_sup    = sup_df.loc[sup_df["Quality_Score"].idxmax(), "Supplier"]
    low_fulf   = sup_df[sup_df["Avg_Fulfillment"] < 90]["Supplier"].count()
    grade_a    = (sup_df["Grade"] == "A").sum()

    if sup_info.get("test_r2") is not None:
        q = "good" if sup_info["test_r2"] > 0.7 else "warn"
        st.session_state.model_metrics["Supplier R²"] = (f"{sup_info['test_r2']:.3f}", q)

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Avg Quality Score",         f"{avg_score:.0f}", suffix="/100")
    with k2: metric_card("Grade A Suppliers",         f"{grade_a}")
    with k3: metric_card("Low Fulfillment Suppliers", f"{low_fulf}")
    with k4: metric_card("Top Supplier",              top_sup[:20] if top_sup else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)

    section("Supplier Quality Scoreboard", "🏆")
    st.plotly_chart(ch.supplier_scoreboard(sup_df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section("Fulfillment Rate Heatmap", "🔥")
        st.plotly_chart(ch.fulfillment_heatmap(pur), use_container_width=True)
    with col2:
        section("Feature Importance", "🔍")
        st.plotly_chart(ch.feat_importance_chart(
            sup_info["feature_importance"], "Supplier Quality Model — Feature Importance"),
            use_container_width=True)

    section("Full Supplier Scorecard", "📋")
    sc = sup_df[["Supplier", "Grade", "Quality_Score", "Avg_Fulfillment",
                 "Std_Fulfillment", "Total_Orders", "Total_Value",
                 "Supplier Rating", "Lead Time Days (Supplier)"]].sort_values(
                     "Quality_Score", ascending=False)
    st.dataframe(sc.style.format({
        "Quality_Score":   "{:.1f}",
        "Avg_Fulfillment": "{:.1f}%",
        "Std_Fulfillment": "{:.1f}",
        "Total_Value":     "${:,.0f}",
        "Supplier Rating": "{:.1f}",
    }).background_gradient(subset=["Quality_Score"], cmap="RdYlGn"),
        use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("Financial Transaction Anomaly Detection")
    st.caption("Isolation Forest · RobustScaler · Log-transformed Features · 5% Contamination")

    with st.spinner("Running Isolation Forest on transactions…"):
        anom_info = _anomaly(txn)

    anom_df = anom_info["df"]

    rate_q = "good" if anom_info["anomaly_rate"] < 6 else "warn"
    st.session_state.model_metrics["Anomaly Rate"] = (
        f"{anom_info['anomaly_rate']:.1f}%", rate_q
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Total Transactions",  f"{len(anom_df):,}")
    with k2: metric_card("Anomalies Detected",  f"{anom_info['anomaly_count']:,}",
                          delta_str=f"⚠ {anom_info['anomaly_rate']:.1f}% rate")
    with k3: metric_card("Anomalous $ Exposure",
                          f"${anom_df[anom_df['Is_Anomaly']]['Total Including Tax'].abs().sum()/1e3:.0f}K")
    with k4: metric_card("Normal Transactions",  f"{(~anom_df['Is_Anomaly']).sum():,}")

    st.markdown("<br>", unsafe_allow_html=True)

    section("Anomaly Scatter: Transaction Amount vs Outstanding Balance", "🎯")
    st.plotly_chart(ch.anomaly_chart(anom_df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section("Anomaly Score Distribution", "📊")
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(
            x=anom_df[~anom_df["Is_Anomaly"]]["Anomaly_Score"],
            name="Normal", marker_color="#00D4FF", opacity=0.7, nbinsx=40))
        fig_score.add_trace(go.Histogram(
            x=anom_df[anom_df["Is_Anomaly"]]["Anomaly_Score"],
            name="Anomaly", marker_color="#FF4C6E", opacity=0.8, nbinsx=40))
        fig_score.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font_color="#E0E0E0", barmode="overlay",
            title=dict(text="Isolation Forest Score Distribution",
                       font=dict(color="#00D4FF")),
            xaxis_title="Anomaly Score", yaxis_title="Count",
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_score, use_container_width=True)

    with col2:
        section("Anomalies by Payment Method", "💳")
        pm_anom = (anom_df[anom_df["Is_Anomaly"]]
                   .groupby("Payment Method")["Is_Anomaly"]
                   .count().reset_index()
                   .rename(columns={"Is_Anomaly": "Anomalies"}))
        fig_pm = px.bar(pm_anom, x="Payment Method", y="Anomalies",
                        color="Anomalies", color_continuous_scale="Reds",
                        template="plotly_dark")
        fig_pm.update_layout(paper_bgcolor="#0E1117", font_color="#E0E0E0",
                              title=dict(text="Anomalies by Payment Method",
                                         font=dict(color="#00D4FF")))
        st.plotly_chart(fig_pm, use_container_width=True)

    section("Top Anomalous Transactions", "⚠️")
    top_anom = (anom_df[anom_df["Is_Anomaly"]]
                .sort_values("Anomaly_Score")
                .head(100)
                [["Transaction Key", "Date Key", "Customer", "Payment Method",
                  "Total Excluding Tax", "Tax Amount", "Total Including Tax",
                  "Outstanding Balance", "Transaction Type", "Anomaly_Score"]])
    st.dataframe(top_anom.style.format({
        "Total Excluding Tax": "${:,.2f}",
        "Total Including Tax": "${:,.2f}",
        "Outstanding Balance": "${:,.2f}",
        "Anomaly_Score":       "{:.4f}",
    }).background_gradient(subset=["Anomaly_Score"], cmap="Reds_r"),
        use_container_width=True, height=350)
