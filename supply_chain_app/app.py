"""
Supply Chain Intelligence Platform  v2.1
No-scroll layout: every page fits in one viewport (≈ 900 px usable height).

Layout rules
────────────
• _c(fig, h)  — compact helper; sets height + tight margins on any Plotly fig.
• Max 2 chart rows per page / per tab.
• Charts in 2–3 st.columns; never full-width unless it's the only row.
• Slider merged into the KPI row (saves ~50 px).
• <br> spacers removed; section() headers removed (charts have own titles).
• Tables capped at height=185.
• Heavy pages (Supplier, Executive Dashboard) split into tabs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import gc
import hashlib

from data_loader import (mart_sales, mart_inventory, mart_purchase,
                          mart_movement, mart_transaction, load_dimensions,
                          load_facts)
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
_TIGHT = dict(l=32, r=14, t=36, b=26)   # tight margins for compact charts

def _c(fig, h: int = 255):
    """Apply compact height + tight margins to any Plotly figure."""
    fig.update_layout(height=h, margin=_TIGHT)
    return fig


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


# ── Fast DataFrame hash for @st.cache_data (avoids hashing 228 K rows) ───────
def _fast_hash(df: pd.DataFrame) -> int:
    step   = max(1, len(df) // 200)
    sample = pd.util.hash_pandas_object(df.iloc[::step]).values.tobytes()
    return int(hashlib.md5(sample).hexdigest(), 16)


# ── Session state ─────────────────────────────────────────────────────────────
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = {}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════════
def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory ~40 % by downcasting numeric columns."""
    for col in df.select_dtypes("float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


@st.cache_data(show_spinner=False)
def get_all_data():
    sale    = _downcast(mart_sales())
    inv_raw = _downcast(mart_inventory())
    pur     = _downcast(mart_purchase())
    mv      = _downcast(mart_movement())
    txn     = _downcast(mart_transaction())
    dims    = load_dimensions()
    gc.collect()
    return sale, inv_raw, pur, mv, txn, dims


with st.spinner("Loading Supply Chain Data Warehouse…"):
    sale, inv_raw, pur, mv, txn, dims = get_all_data()


# ══════════════════════════════════════════════════════════════════════════════
# CACHED MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _sale_for_forecast(sale):
    """Slim down to columns ml_models needs; ml_models does its own aggregation."""
    needed = ["Calendar Year", "Calendar Month Number", "Stock Category",
              "Quantity", "Unit Price", "Sale Key"]
    cols = [c for c in needed if c in sale.columns]
    return sale[cols].reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=12)
def _forecast(sale_slim, horizon):
    return build_demand_forecast(sale_slim, horizon_months=horizon)


@st.cache_data(show_spinner=False, max_entries=2,
               hash_funcs={pd.DataFrame: _fast_hash})
def _stockout(inv_raw, mv, sale):
    return build_stockout_classifier(inv_raw, mv, sale_df=sale)


@st.cache_data(show_spinner=False, max_entries=2,
               hash_funcs={pd.DataFrame: _fast_hash})
def _churn(sale):
    return build_churn_predictor(sale)


@st.cache_data(show_spinner=False, max_entries=2,
               hash_funcs={pd.DataFrame: _fast_hash})
def _supplier(pur, supplier_dim):
    return build_supplier_scorer(pur, supplier_dim)


@st.cache_data(show_spinner=False, max_entries=2,
               hash_funcs={pd.DataFrame: _fast_hash})
def _anomaly(txn):
    return build_anomaly_detector(txn)


@st.cache_data(show_spinner=False, max_entries=2,
               hash_funcs={pd.DataFrame: _fast_hash})
def _segments(sale):
    return build_customer_segments(sale)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
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
        <div class="sb-live-dot"></div>
    </div>
    <div class="sb-divider"></div>
    """, unsafe_allow_html=True)

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
            <div class="sb-stat"><div class="sb-stat-val">{total_records/1e3:.0f}K</div>
                <div class="sb-stat-lbl">Records</div></div>
            <div class="sb-stat"><div class="sb-stat-val">{n_skus}</div>
                <div class="sb-stat-lbl">SKUs</div></div>
            <div class="sb-stat"><div class="sb-stat-val">{n_customers}</div>
                <div class="sb-stat-lbl">Customers</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Modules</div>', unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠 Executive Dashboard",
        "📈 Demand Forecasting",
        "📦 Inventory Risk",
        "👥 Customer Intelligence",
        "🏭 Supplier Analytics",
        "🚨 Anomaly Detection",
    ], label_visibility="collapsed")

    if st.session_state.model_metrics:
        st.markdown('<div class="sb-divider" style="margin-top:10px"></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Model Performance</div>',
                    unsafe_allow_html=True)
        rows_html = "".join(
            f'<div class="sb-metric-row">'
            f'<span class="sb-metric-name">{k}</span>'
            f'<span class="sb-metric-val {q}">{v}</span></div>'
            for k, (v, q) in st.session_state.model_metrics.items()
        )
        st.markdown(f'<div class="sb-metrics">{rows_html}</div>', unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# Two tabs so every chart fits in one viewport without scrolling.
# Tab 1 "Overview"   → KPIs + Revenue timeline + Category bar + Waterfall
# Tab 2 "Territory"  → Territory bar + Top customers table
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Dashboard":
    st.title("Executive Supply Chain Dashboard")
    st.caption("Galaxy Schema · FY2013-2016 · Real-time KPIs")

    total_rev    = sale["Total Excluding Tax"].sum()
    total_profit = sale["Profit"].sum()
    margin_pct   = total_profit / total_rev * 100 if total_rev else 0

    total_skus   = sale["Stock Item Key"].nunique()
    avg_order_v  = sale["Total Including Tax"].mean()

    # ── Total orders: distinct WWI Order ID from Fact.Order (not Fact.Sale) ──
    # Fact.Sale has one row per line item; Fact.Order is the authoritative
    # order register and gives the correct ~74 K distinct order count.
    try:
        _order_fact   = load_facts()["order"]
        _order_id_col = next(
            (c for c in ["WWI Order ID", "Order ID", "Order Key"]
             if c in _order_fact.columns), None
        )
        total_orders = (
            int(_order_fact[_order_id_col].nunique()) if _order_id_col
            else int(len(_order_fact))
        )
    except Exception:
        if "Order Key" in sale.columns:
            total_orders = int(sale["Order Key"].nunique())
        else:
            total_orders = int(sale.groupby(
                ["Customer Key", "Invoice Date Key", "City Key"]
            ).ngroups)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: metric_card("Total Revenue",   f"${total_rev/1e6:.1f}M")
    with k2: metric_card("Total Profit",    f"${total_profit/1e6:.1f}M")
    with k3: metric_card("Gross Margin",    f"{margin_pct:.1f}",  suffix="%")
    with k4: metric_card("Total Orders",    f"{total_orders:,}")
    with k5: metric_card("Active SKUs",     f"{total_skus:,}")
    with k6: metric_card("Avg Order Value", f"${avg_order_v:.0f}")

    tab_ov, tab_te = st.tabs(["📊 Overview", "🗺️ Territory & Customers"])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab_ov:
        ca, cb, cc = st.columns([4, 3, 3])
        with ca:
            st.plotly_chart(_c(ch.revenue_timeline(sale), 258),
                            use_container_width=True, config={"displayModeBar": False})
        with cb:
            st.plotly_chart(_c(ch.sales_by_category(sale), 258),
                            use_container_width=True, config={"displayModeBar": False})
        with cc:
            st.plotly_chart(_c(ch.margin_waterfall(sale), 258),
                            use_container_width=True, config={"displayModeBar": False})

    # ── Tab 2: Territory & Customers ─────────────────────────────────────────
    with tab_te:
        terr = (sale.groupby("Sales Territory")
                .agg(Revenue=("Total Excluding Tax","sum"),
                     Profit=("Profit","sum"))
                .reset_index().dropna(subset=["Sales Territory"])
                .sort_values("Revenue", ascending=False))
        fig_t = px.bar(
            terr, x="Sales Territory", y="Revenue",
            color="Profit", color_continuous_scale="Viridis",
            template="plotly_dark",
            labels={"Revenue": "Revenue ($)", "Profit": "Profit ($)"})
        fig_t.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="#E0E0E0",
            coloraxis_colorbar=dict(tickfont=dict(color="#E0E0E0")),
            title=dict(text="Revenue by Sales Territory", font=dict(color="#00D4FF")))

        top_cust = (sale.groupby("Customer")
                    .agg(Revenue=("Total Excluding Tax","sum"),
                         Profit=("Profit","sum"))
                    .reset_index()
                    .sort_values("Revenue", ascending=False).head(10)
                    .reset_index(drop=True))
        top_cust.insert(0, "Rank", [f"#{i+1}" for i in range(len(top_cust))])

        ct, cc2 = st.columns([3, 2])
        with ct:
            st.plotly_chart(_c(fig_t, 285),
                            use_container_width=True, config={"displayModeBar": False})
        with cc2:
            st.caption("🌟 Top 10 Customers by Revenue")
            st.dataframe(
                top_cust.style.format({"Revenue": "${:,.0f}", "Profit": "${:,.0f}"}),
                use_container_width=True, height=260, hide_index=True,
                column_config={
                    "Rank":     st.column_config.TextColumn("#",        width="small"),
                    "Customer": st.column_config.TextColumn("Customer", width="large"),
                    "Revenue":  st.column_config.TextColumn("Revenue",  width="medium"),
                    "Profit":   st.column_config.TextColumn("Profit",   width="medium"),
                })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEMAND FORECASTING
# Row 1: KPIs + horizon slider merged in one row (saves ~50 px)
# Row 2: Forecast chart (full-width, 260 px)
# Row 3: Feature importance | Forecast table | Actual vs Predicted  (3-up)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecasting":
    st.title("Demand Forecasting")
    st.caption("LightGBM / HistGradientBoosting · Auto-regressive · Per Stock Category")

    # KPIs + slider in one row
    km1, km2, km3, km4, ks = st.columns([1, 1, 1, 1, 2])

    with ks:
        horizon = st.slider("Forecast horizon (months)", 1, 12, 3, key="fc_horizon")

    with st.spinner("Training demand forecast model…"):
        info = _forecast(_sale_for_forecast(sale), horizon)

    mape  = info.get("mape",  0.0)
    wmape = info.get("wmape", 0.0)
    r2    = info.get("r2",    0.0)
    q_mape = "good" if mape  < 10 else "warn" if mape  < 20 else ""
    q_r2   = "good" if r2    > 0.7 else "warn" if r2    > 0.4 else ""
    st.session_state.model_metrics["Demand MAPE"]     = (f"{mape:.1f}%",  q_mape)
    st.session_state.model_metrics["Demand R²(log)"]  = (f"{r2:.3f}",     q_r2)

    with km1: metric_card("MAPE",           f"{mape:.1f}",  suffix="%")
    with km2: metric_card("WMAPE",          f"{wmape:.1f}", suffix="%")
    with km3: metric_card("R² (log-scale)", f"{r2:.3f}")
    with km4: metric_card("Horizon",        f"{horizon}",   suffix=" mo")

    # Row 2 — forecast chart
    cats     = sorted(sale["Stock Category"].dropna().unique())
    selected = st.multiselect("Stock categories to forecast",
                              cats, default=cats[:4], key="fc_cats")
    if selected:
        st.plotly_chart(
            _c(ch.forecast_chart(info, selected), 260),
            use_container_width=True, config={"displayModeBar": False})

    # Row 3 — Actual vs Predicted full-width (feature importance removed — not for end users)
    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=info["test_true"], y=info["test_pred"], mode="markers",
        marker=dict(color="#00D4FF", opacity=0.45, size=4), name="Predictions"))
    mn = float(min(info["test_true"].min(), info["test_pred"].min()))
    mx = float(max(info["test_true"].max(), info["test_pred"].max()))
    fig_ap.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color="#FF6B35", dash="dot", width=1.5), name="Perfect fit"))
    fig_ap.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font_color="#E0E0E0",
        xaxis_title="Actual Units", yaxis_title="Predicted Units",
        title=dict(text=f"Actual vs Predicted  —  R²={r2:.3f} (log-scale)",
                   font=dict(color="#00D4FF", size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(_c(fig_ap, 240),
                    use_container_width=True, config={"displayModeBar": False})

    # Row 4 — Forecast table FULL WIDTH so every column is visible without scrolling
    st.caption("📋 Forecast Table")
    if selected and not info["forecast_df"].empty:
        fc_show = (
            info["forecast_df"][info["forecast_df"]["Stock Category"].isin(selected)]
            [["Stock Category", "Forecast Year", "Forecast Month", "Predicted_Qty"]]
            .sort_values(["Stock Category", "Forecast Year", "Forecast Month"])
            .rename(columns={
                "Stock Category": "Category",
                "Forecast Year":  "Year",
                "Forecast Month": "Month",
                "Predicted_Qty":  "Forecast Units",
            })
            .reset_index(drop=True)
        )
        fc_show["Forecast Units"] = fc_show["Forecast Units"].round(0).astype(int)
        st.dataframe(
            fc_show,
            use_container_width=True, height=230, hide_index=True,
            column_config={
                "Category":       st.column_config.TextColumn("Category",       width="large"),
                "Year":           st.column_config.NumberColumn("Yr",           width="small", format="%d"),
                "Month":          st.column_config.NumberColumn("Mo",           width="small", format="%d"),
                "Forecast Units": st.column_config.NumberColumn("Forecast Qty", width="medium"),
            },
        )
    else:
        st.info("Select categories above to see forecast values.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — INVENTORY RISK
# Row 1: 4 KPIs
# Row 2: Risk matrix | Risk pie | Feature importance  (3-up, 255 px)
# Row 3: HIGH-risk table | Classification report       (2-up, 185 px)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Inventory Risk":
    st.title("Inventory Risk Intelligence")
    st.caption("LightGBM Classifier · Corrected Monthly Velocity · Stockout Alerts")

    with st.spinner("Running stockout risk model…"):
        inv_info = _stockout(inv_raw, mv, sale)

    inv_df    = inv_info["df"]
    high      = (inv_df["Predicted_Risk_Name"] == "HIGH").sum()
    medium    = (inv_df["Predicted_Risk_Name"] == "MEDIUM").sum()
    stock_val = inv_df["Stock Value"].sum()

    if inv_info["auc"]:
        metric_label = inv_info.get("metric_name", "Balanced Accuracy")
        q = "good" if inv_info["auc"] > 0.70 else "warn"
        st.session_state.model_metrics["Stockout Bal-Acc"] = (f"{inv_info['auc']:.3f}", q)

    # Row 1 — KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Total SKUs",       f"{len(inv_df):,}")
    with k2: metric_card("HIGH Risk SKUs",   f"{high}", delta_str=f"+{high} need action")
    with k3: metric_card("MEDIUM Risk SKUs", f"{medium}")
    with k4: metric_card("Stock Value",      f"${stock_val/1e6:.1f}M")

    # Row 2 — Risk scatter (wider) + pie side by side
    rc1, rc2 = st.columns([3, 1])
    with rc1:
        if "Monthly_Velocity" not in inv_df.columns:
            inv_df["Monthly_Velocity"] = 0
        st.plotly_chart(_c(ch.inventory_risk_scatter(inv_df), 270),
                        use_container_width=True, config={"displayModeBar": False})
    with rc2:
        risk_cnt = inv_df["Predicted_Risk_Name"].value_counts().reset_index()
        risk_cnt.columns = ["Risk", "Count"]
        fig_pie = px.pie(risk_cnt, names="Risk", values="Count",
                         color="Risk",
                         color_discrete_map={"HIGH":"#FF4C6E","MEDIUM":"#FFD700","LOW":"#00C49A"},
                         template="plotly_dark")
        fig_pie.update_layout(
            paper_bgcolor="#0E1117", font_color="#E0E0E0",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title=dict(text="Risk Distribution", font=dict(color="#00D4FF", size=13)))
        st.plotly_chart(_c(fig_pie, 270),
                        use_container_width=True, config={"displayModeBar": False})

    # Row 3 — HIGH Risk table full-width + Classification Report below
    st.caption("⚠️ HIGH Risk SKUs — Immediate Reorder Needed")
    high_df = (inv_df[inv_df["Predicted_Risk_Name"] == "HIGH"]
               [["Stock Item","Stock Category","Quantity On Hand",
                 "Reorder Level","Target Stock Level","Monthly_Velocity",
                 "Days_Coverage","Stock Value","Lead Time Days"]]
               .sort_values("Quantity On Hand"))
    st.dataframe(high_df.style.format({
        "Quantity On Hand":   "{:,.0f}",
        "Reorder Level":      "{:,.0f}",
        "Target Stock Level": "{:,.0f}",
        "Monthly_Velocity":   "{:,.1f}",
        "Days_Coverage":      "{:,.0f}",
        "Stock Value":        "${:,.0f}",
    }), use_container_width=True, height=240, hide_index=True,
    column_config={
        "Stock Item":          st.column_config.TextColumn("SKU",         width="large"),
        "Stock Category":      st.column_config.TextColumn("Category",    width="medium"),
        "Quantity On Hand":    st.column_config.NumberColumn("QoH",       width="small"),
        "Reorder Level":       st.column_config.NumberColumn("Reorder",   width="small"),
        "Target Stock Level":  st.column_config.NumberColumn("Target",    width="small"),
        "Monthly_Velocity":    st.column_config.NumberColumn("Mo.Vel",    width="small"),
        "Days_Coverage":       st.column_config.NumberColumn("Days Cov",  width="small"),
        "Stock Value":         st.column_config.TextColumn("$ Value",     width="medium"),
        "Lead Time Days":      st.column_config.NumberColumn("Lead d",    width="small"),
    })

    if inv_info["report"]:
        with st.expander("📋 Classification Report", expanded=False):
            rpt = pd.DataFrame(inv_info["report"]).T.drop(columns=["support"], errors="ignore")
            st.dataframe(rpt.style.format("{:.2f}"), use_container_width=True)
        if inv_info["auc"]:
            _mn = inv_info.get("metric_name", "Balanced Accuracy")
            st.metric(_mn, f"{inv_info['auc']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER INTELLIGENCE
# Tab Churn:       KPIs + 3-up charts (dist | feat imp | at-risk table)
# Tab Segmentation: 2-up (3D scatter | segment summary)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Intelligence":
    st.title("Customer Intelligence")
    st.caption("Churn Predictor · RFM Segmentation · KMeans Clustering")

    tab_ch, tab_seg = st.tabs(["🔮 Churn Prediction", "🗺️ RFM Segmentation"])

    with tab_ch:
        with st.spinner("Training churn model…"):
            churn_info = _churn(sale)

        rfm         = churn_info["rfm"]
        churn_count = rfm["Churn_Pred"].sum()
        active_cnt  = (rfm["Churn_Pred"] == 0).sum()
        churn_rate  = rfm["Churn_Pred"].mean() * 100

        if churn_info["auc"]:
            q = "good" if churn_info["auc"] > 0.80 else "warn"
            st.session_state.model_metrics["Churn AUC"] = (f"{churn_info['auc']:.3f}", q)

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1: metric_card("Total Customers",    f"{len(rfm):,}")
        with k2: metric_card("At-Risk",            f"{churn_count:,}",
                              delta_str=f"⚠ {churn_rate:.1f}% rate")
        with k3: metric_card("Low-Risk Customers", f"{active_cnt:,}")
        with k4: metric_card("ROC-AUC",
                              f"{churn_info['auc']:.3f}" if churn_info["auc"] else "N/A")

        # 2-up: churn distribution (left) | at-risk table (right, wider)
        cc1, cc2 = st.columns([2, 3])
        with cc1:
            st.plotly_chart(_c(ch.churn_dist(rfm), 280),
                            use_container_width=True, config={"displayModeBar": False})
        with cc2:
            at_risk = (rfm[rfm["Churn_Pred"] == 1]
                       .sort_values("Churn_Prob", ascending=False).head(40)
                       [["Customer","Recency","Frequency","Monetary","Churn_Prob"]]
                       .rename(columns={"Monetary":"Revenue ($)",
                                        "Churn_Prob":"Churn %"}))
            at_risk["Churn %"]    = (at_risk["Churn %"] * 100).round(1)
            at_risk["Revenue ($)"]= at_risk["Revenue ($)"].round(0)
            st.caption("🚨 Top At-Risk Customers")
            st.dataframe(at_risk.style.format({
                "Revenue ($)": "${:,.0f}", "Churn %": "{:.1f}%",
            }).background_gradient(subset=["Churn %"], cmap="RdYlGn_r"),
                use_container_width=True, height=280, hide_index=True,
                column_config={
                    "Customer":    st.column_config.TextColumn("Customer",    width="large"),
                    "Recency":     st.column_config.NumberColumn("Rec(d)",    width="small"),
                    "Frequency":   st.column_config.NumberColumn("Orders",    width="small"),
                    "Revenue ($)": st.column_config.TextColumn("Revenue",     width="medium"),
                    "Churn %":     st.column_config.TextColumn("Churn Risk",  width="medium"),
                })

    with tab_seg:
        with st.spinner("Segmenting customers…"):
            seg_info = _segments(sale)

        rfm_seg   = seg_info["rfm"]
        sil_score = seg_info.get("silhouette", None)
        best_k    = seg_info.get("best_k", 4)

        if sil_score is not None:
            q_sil = "good" if sil_score > 0.45 else "warn" if sil_score > 0.25 else ""
            st.session_state.model_metrics["Seg Silhouette"] = (f"{sil_score:.3f}", q_sil)

        st.info(
            "ℹ️ **Segment counts differ from Churn counts by design.** "
            "Segmentation (below) uses unsupervised GMM clustering on RFM behaviour — "
            "it assigns every customer to exactly one of the clusters and labels the "
            "worst cluster 'Churned/Lost'.  "
            "The Churn tab uses a supervised ML model with Youden-J threshold — it "
            "independently scores each customer's churn probability.  "
            "The two methods answer different questions and their counts will not match."
        )

        # KPI strip — use sale-level nunique so the number is consistent
        # with the Churn tab (both count unique Customer Keys in the sale fact).
        _total_cust = sale["Customer Key"].nunique()
        sk1, sk2, sk3 = st.columns(3)
        with sk1: metric_card("Segments Found",   str(best_k))
        with sk2: metric_card("Silhouette Score",
                               f"{sil_score:.3f}" if sil_score is not None else "N/A")
        with sk3: metric_card("Total Customers",  f"{_total_cust:,}")

        seg_sum = (rfm_seg.groupby("Segment Name")
                   .agg(Count=("Customer Key","count"),
                        Avg_Recency=("Recency","mean"),
                        Avg_Frequency=("Frequency","mean"),
                        Avg_Monetary=("Monetary","mean"))
                   .reset_index()
                   .sort_values("Avg_Monetary", ascending=False))

        # 3D scatter full-width
        st.plotly_chart(_c(ch.rfm_3d(rfm_seg), 340),
                        use_container_width=True, config={"displayModeBar": False})

        # Segment summary full-width below — all columns visible
        st.caption("📋 Segment Summary")
        st.dataframe(seg_sum.style.format({
            "Avg_Recency":   "{:.0f}",
            "Avg_Frequency": "{:.0f}",
            "Avg_Monetary":  "${:,.0f}",
        }), use_container_width=True, height=215, hide_index=True,
        column_config={
            "Segment Name":  st.column_config.TextColumn("Segment",       width="large"),
            "Count":         st.column_config.NumberColumn("Customers",   width="small"),
            "Avg_Recency":   st.column_config.NumberColumn("Avg Rec(d)",  width="small"),
            "Avg_Frequency": st.column_config.NumberColumn("Avg Orders",  width="small"),
            "Avg_Monetary":  st.column_config.TextColumn("Avg Revenue",   width="medium"),
        })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SUPPLIER ANALYTICS
# Tab Scoring:     KPIs + 2-up (scoreboard | feature importance)
# Tab Fulfillment: 2-up (heatmap | full scorecard table)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Supplier Analytics":
    st.title("Supplier Analytics & Quality Scoring")
    st.caption("Volume-unbiased scoring · 5-pillar composite · Trend-aware · HistGB + RF blend")

    with st.spinner("Scoring suppliers…"):
        sup_info = _supplier(pur, dims["supplier"])

    sup_df    = sup_info["df"]
    avg_score = sup_df["Quality_Score"].mean()
    top_sup   = sup_df.loc[sup_df["Quality_Score"].idxmax(), "Supplier"]
    # v3: use True_Fulfillment (unit-weighted) for low-fulfillment count
    fill_col  = "True_Fulfillment" if "True_Fulfillment" in sup_df.columns else "Avg_Fulfillment"
    low_fulf  = (sup_df[fill_col] < 90).sum()
    grade_a   = (sup_df["Grade"] == "A").sum()
    improving = (sup_df.get("Trend_Direction", pd.Series([])) == "↑ Improving").sum()

    if sup_info.get("test_r2") is not None:
        q = "good" if sup_info["test_r2"] > 0.7 else "warn"
        st.session_state.model_metrics["Supplier R²"] = (f"{sup_info['test_r2']:.3f}", q)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: metric_card("Avg Quality Score",        f"{avg_score:.0f}", suffix="/100")
    with k2: metric_card("Grade A Suppliers",        f"{grade_a}")
    with k3: metric_card("Low Fill-Rate (< 90 %)",   f"{low_fulf}")
    with k4: metric_card("↑ Improving Trend",        f"{improving}")
    with k5: metric_card("Top Supplier",             (top_sup or "N/A")[:20])

    tab_sc, tab_fu, tab_pillar = st.tabs(["🏆 Quality Scoring", "🔥 Fulfillment", "📊 Pillar Breakdown"])

    with tab_sc:
        # Scoreboard full-width — no feature importance for end users
        st.plotly_chart(_c(ch.supplier_scoreboard(sup_df), 320),
                        use_container_width=True, config={"displayModeBar": False})

    with tab_fu:
        # Fulfillment heatmap full-width
        st.plotly_chart(_c(ch.fulfillment_heatmap(pur), 320),
                        use_container_width=True, config={"displayModeBar": False})

        # Full scorecard — "Supplier Rating" removed per business request
        st.caption("📋 Full Supplier Scorecard  *(True Fill = unit-weighted Σreceived/Σordered)*")
        show_cols = ["Supplier", "Grade", "Quality_Score"]
        show_cols += [c for c in ["True_Fulfillment", "Trend_Direction",
                                   "Total_Orders", "Total_Value"] if c in sup_df.columns]
        sc_df = sup_df[show_cols].sort_values("Quality_Score", ascending=False)
        fmt   = {"Quality_Score": "{:.1f}", "Total_Value": "${:,.0f}"}
        if "True_Fulfillment" in sc_df.columns:
            fmt["True_Fulfillment"] = "{:.1f}%"
        st.dataframe(
            sc_df.style.format(fmt)
                       .background_gradient(subset=["Quality_Score"], cmap="RdYlGn"),
            use_container_width=True, height=310, hide_index=True,
            column_config={
                "Supplier":         st.column_config.TextColumn("Supplier",    width="large"),
                "Grade":            st.column_config.TextColumn("Grd",         width="small"),
                "Quality_Score":    st.column_config.NumberColumn("Score /100", width="small", format="%.1f"),
                "True_Fulfillment": st.column_config.TextColumn("Fill %",      width="small"),
                "Trend_Direction":  st.column_config.TextColumn("Trend",       width="medium"),
                "Total_Orders":     st.column_config.NumberColumn("Orders",    width="small"),
                "Total_Value":      st.column_config.TextColumn("$ Value",     width="medium"),
            },
        )

    with tab_pillar:
        st.caption("📐 Pillar scores show where each supplier excels or lags (each 0-100)")
        pillar_cols = [c for c in ["Supplier", "Grade", "P_Reliability", "P_Consistency",
                                    "P_Trend", "P_Volume", "P_Attributes", "Quality_Score"]
                       if c in sup_df.columns]
        if len(pillar_cols) > 2:
            p_df = sup_df[pillar_cols].sort_values("Quality_Score", ascending=False)
            gradient_cols = [c for c in p_df.columns if c.startswith("P_") or c == "Quality_Score"]
            st.dataframe(
                p_df.style
                    .format({c: "{:.1f}" for c in gradient_cols})
                    .background_gradient(subset=gradient_cols, cmap="RdYlGn", vmin=0, vmax=100),
                use_container_width=True, height=350, hide_index=True,
                column_config={
                    "Supplier":       st.column_config.TextColumn("Supplier",    width="large"),
                    "Grade":          st.column_config.TextColumn("Grd",         width="small"),
                    "P_Reliability":  st.column_config.NumberColumn("Reliab.",   width="small", format="%.1f"),
                    "P_Consistency":  st.column_config.NumberColumn("Consist.",  width="small", format="%.1f"),
                    "P_Trend":        st.column_config.NumberColumn("Trend",     width="small", format="%.1f"),
                    "P_Volume":       st.column_config.NumberColumn("Volume",    width="small", format="%.1f"),
                    "P_Attributes":   st.column_config.NumberColumn("Attrib.",   width="small", format="%.1f"),
                    "Quality_Score":  st.column_config.NumberColumn("Total /100",width="medium", format="%.1f"),
                },
            )
        else:
            st.info("Pillar scores not available — re-run scoring.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ANOMALY DETECTION
# Row 1: 4 KPIs
# Row 2: Scatter | Score dist | Payment method  (3-up, 255 px)
# Row 3: Anomalous transactions table           (full-width, 185 px)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("Financial Transaction Anomaly Detection")
    st.caption("Isolation Forest · RobustScaler · Log-transformed Features · 5 % Contamination")

    with st.spinner("Running Isolation Forest…"):
        # Use session_state instead of @st.cache_data to avoid DataFrame
        # hashing issues that caused the page not to load.
        if "anom_info" not in st.session_state:
            try:
                st.session_state["anom_info"] = build_anomaly_detector(txn)
            except Exception as _e:
                st.error(
                    f"**Anomaly detection failed:** {_e}\n\n"
                    "Check that `Fact.Transaction.csv` contains "
                    "`Total Including Tax` and `Outstanding Balance` columns."
                )
                st.stop()
        anom_info = st.session_state["anom_info"]

    anom_df = anom_info["df"].copy()
    # Ensure Is_Anomaly is strictly boolean regardless of dtype from cache/downcast
    anom_df["Is_Anomaly"] = anom_df["Is_Anomaly"].astype(bool)
    is_anom_mask  = anom_df["Is_Anomaly"]
    not_anom_mask = ~anom_df["Is_Anomaly"]
    rate_q  = "good" if anom_info["anomaly_rate"] < 6 else "warn"
    st.session_state.model_metrics["Anomaly Rate"] = (
        f"{anom_info['anomaly_rate']:.1f}%", rate_q)

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Total Transactions",  f"{len(anom_df):,}")
    with k2: metric_card("Anomalies Detected",  f"{anom_info['anomaly_count']:,}",
                          delta_str=f"⚠ {anom_info['anomaly_rate']:.1f}% rate")
    _exposure = anom_df.loc[is_anom_mask, "Total Including Tax"].abs().sum() if "Total Including Tax" in anom_df.columns else 0
    _exp_str  = (f"${_exposure/1e6:.1f}M" if _exposure >= 1e6
                 else f"${_exposure/1e3:.1f}K" if _exposure >= 1e3
                 else f"${_exposure:,.0f}")
    with k3: metric_card("$ Exposure", _exp_str)
    with k4: metric_card("Normal Transactions", f"{not_anom_mask.sum():,}")

    # Row 2 — charts (3-up)
    ac1, ac2, ac3 = st.columns([3, 3, 2])
    with ac1:
        st.plotly_chart(_c(ch.anomaly_chart(anom_df), 255),
                        use_container_width=True, config={"displayModeBar": False})
    with ac2:
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(
            x=anom_df.loc[not_anom_mask, "Anomaly_Score"],
            name="Normal",  marker_color="#00D4FF", opacity=0.7, nbinsx=35))
        fig_score.add_trace(go.Histogram(
            x=anom_df.loc[is_anom_mask, "Anomaly_Score"],
            name="Anomaly", marker_color="#FF4C6E", opacity=0.8, nbinsx=35))
        fig_score.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font_color="#E0E0E0", barmode="overlay",
            title=dict(text="Score Distribution", font=dict(color="#00D4FF", size=13)),
            xaxis_title="Score", yaxis_title="Count",
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(_c(fig_score, 255),
                        use_container_width=True, config={"displayModeBar": False})
    with ac3:
        if "Payment Method" in anom_df.columns:
            pm_anom = (anom_df.loc[is_anom_mask]
                       .groupby("Payment Method")["Is_Anomaly"].count()
                       .reset_index().rename(columns={"Is_Anomaly":"Anomalies"}))
            fig_pm = px.bar(pm_anom, x="Anomalies", y="Payment Method",
                            orientation="h",
                            color="Anomalies", color_continuous_scale="Reds",
                            template="plotly_dark")
            fig_pm.update_layout(
                paper_bgcolor="#0E1117", font_color="#E0E0E0",
                title=dict(text="By Payment Method", font=dict(color="#00D4FF", size=13)),
                showlegend=False)
            st.plotly_chart(_c(fig_pm, 255),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Payment Method column not available.")

    # Row 3 — anomalous transactions table
    st.caption("⚠️ Top Anomalous Transactions")
    _want_cols = ["Transaction Key", "WWI Transaction ID", "Date Key", "Customer",
                  "Payment Method", "Total Including Tax", "Outstanding Balance",
                  "Transaction Type", "Anomaly_Score"]
    _have_cols = [c for c in _want_cols if c in anom_df.columns]
    top_anom = (anom_df.loc[is_anom_mask]
                .sort_values("Anomaly_Score").head(100)[_have_cols])
    fmt_cols = {}
    if "Total Including Tax"  in top_anom.columns: fmt_cols["Total Including Tax"]  = "${:,.2f}"
    if "Outstanding Balance"  in top_anom.columns: fmt_cols["Outstanding Balance"]  = "${:,.2f}"
    if "Anomaly_Score"        in top_anom.columns: fmt_cols["Anomaly_Score"]        = "{:.4f}"

    _col_cfg = {}
    if "Transaction Key"    in _have_cols: _col_cfg["Transaction Key"]    = st.column_config.NumberColumn("Txn Key",      width="small")
    if "WWI Transaction ID" in _have_cols: _col_cfg["WWI Transaction ID"] = st.column_config.NumberColumn("WWI Txn ID",   width="small")
    if "Date Key"           in _have_cols: _col_cfg["Date Key"]           = st.column_config.DateColumn("Date",           width="small")
    if "Customer"           in _have_cols: _col_cfg["Customer"]           = st.column_config.TextColumn("Customer",       width="medium")
    if "Payment Method"     in _have_cols: _col_cfg["Payment Method"]     = st.column_config.TextColumn("Payment",        width="medium")
    if "Total Including Tax" in _have_cols: _col_cfg["Total Including Tax"] = st.column_config.TextColumn("Total incl. Tax", width="medium")
    if "Outstanding Balance" in _have_cols: _col_cfg["Outstanding Balance"] = st.column_config.TextColumn("Outstanding",  width="medium")
    if "Transaction Type"   in _have_cols: _col_cfg["Transaction Type"]   = st.column_config.TextColumn("Txn Type",       width="medium")
    if "Anomaly_Score"      in _have_cols: _col_cfg["Anomaly_Score"]      = st.column_config.NumberColumn("Score",        width="small", format="%.4f")

    st.dataframe(top_anom.style.format(fmt_cols)
                 .background_gradient(subset=["Anomaly_Score"] if "Anomaly_Score" in top_anom.columns else [],
                                      cmap="Reds_r"),
        use_container_width=True, height=300, hide_index=True,
        column_config=_col_cfg)