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


def _ptitle(text: str) -> None:
    """Page title with a consistent breathing-room spacer below the heading."""
    st.title(text)
    st.markdown('<div style="margin-bottom:1.3rem"></div>', unsafe_allow_html=True)


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

    # ── Operations Pulse ──────────────────────────────────────────────────────
    _latest_date = pd.NaT
    if "Invoice Date Key" in sale.columns:
        _latest_date = sale["Invoice Date Key"].dropna().max()
    _date_str = _latest_date.strftime("%b %Y") if pd.notna(_latest_date) else "N/A"

    # MoM revenue trend (month-level view — pulse through the current month)
    _mom_str, _mom_color = "N/A", "#AAB4BE"
    if "Invoice Date Key" in sale.columns:
        _sale_dated    = sale.dropna(subset=["Invoice Date Key"])
        _latest_period = _sale_dated["Invoice Date Key"].dt.to_period("M").max()
        _prev_period   = _latest_period - 1
        _r_cm = _sale_dated[_sale_dated["Invoice Date Key"].dt.to_period("M") == _latest_period]["Total Excluding Tax"].sum()
        _r_pm = _sale_dated[_sale_dated["Invoice Date Key"].dt.to_period("M") == _prev_period]["Total Excluding Tax"].sum()
        if _r_pm > 0:
            _mom        = (_r_cm - _r_pm) / _r_pm * 100
            _mom_arrow  = "▲" if _mom >= 0 else "▼"
            _mom_color  = "#00C49A" if _mom >= 0 else "#FF4C6E"
            _mom_str    = f"{_mom_arrow} {abs(_mom):.1f}%"

    # Inventory alerts
    _reorder_alerts   = int(inv_raw["Reorder Flag"].sum())   if "Reorder Flag"   in inv_raw.columns else 0
    _overstock_alerts = int(inv_raw["Overstock Flag"].sum()) if "Overstock Flag" in inv_raw.columns else 0
    _alert_color      = "#FF4C6E" if _reorder_alerts > 0 else "#00C49A"
    _n_skus           = sale["Stock Item Key"].nunique() if "Stock Item Key" in sale.columns else 0
    _n_cust           = sale["Customer Key"].nunique()   if "Customer Key"   in sale.columns else 0
    _badge_cls        = "badge-high" if _overstock_alerts > 0 else "badge-low"
    _badge_txt        = f"⚠ {_overstock_alerts} Overstock" if _overstock_alerts > 0 else "✓ Stock OK"

    st.markdown(f"""
    <div class="sb-pulse">
        <div class="sb-pulse-head">
            <div class="sb-status-dot"></div>
            <span class="sb-pulse-title">📡 Operations Pulse</span>
        </div>
        <div class="sb-pulse-date">Data through <b style="color:#8BA0B4">{_date_str}</b></div>
        <div class="sb-pulse-grid">
            <div class="sb-pulse-item">
                <div class="sb-pulse-val" style="color:{_mom_color}">{_mom_str}</div>
                <div class="sb-pulse-lbl">MoM Revenue</div>
            </div>
            <div class="sb-pulse-item">
                <div class="sb-pulse-val" style="color:{_alert_color}">{_reorder_alerts:,}</div>
                <div class="sb-pulse-lbl">Reorder Alerts</div>
            </div>
            <div class="sb-pulse-item">
                <div class="sb-pulse-val">{_n_skus:,}</div>
                <div class="sb-pulse-lbl">Active SKUs</div>
            </div>
            <div class="sb-pulse-item">
                <div class="sb-pulse-val">{_n_cust:,}</div>
                <div class="sb-pulse-lbl">Customers</div>
            </div>
        </div>
        <div class="sb-pulse-footer">
            <span class="sb-pulse-badge {_badge_cls}">{_badge_txt}</span>
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




# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# 4-chart 2×2 grid (no tabs) — every chart follows ch.LAYOUT identity
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Dashboard":
    _ptitle("Executive Supply Chain Dashboard")

    total_rev    = sale["Total Excluding Tax"].sum()
    total_profit = sale["Profit"].sum()
    margin_pct   = total_profit / total_rev * 100 if total_rev else 0

    total_skus   = sale["Stock Item Key"].nunique()
    avg_order_v  = sale["Total Including Tax"].mean()

    # ── Total orders: distinct WWI Order ID from Fact.Order (not Fact.Sale) ──
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

    # ── 2 × 2 grid — all 4 charts, no tabs ──────────────────────────────────
    _H = 270

    row1_left, row1_right = st.columns(2)
    with row1_left:
        _rv = ch.revenue_timeline(sale)
        _c(_rv, _H)
        _rv.update_layout(margin=dict(l=32, r=14, t=72, b=26))
        st.plotly_chart(_rv, use_container_width=True, config={"displayModeBar": False})
    with row1_right:
        # Territory revenue bar — top margin matches revenue timeline (t=72) for vertical alignment
        _terr_fig = ch.top_territory_chart(sale)
        _c(_terr_fig, _H)
        _terr_fig.update_layout(margin=dict(l=32, r=14, t=72, b=26))
        st.plotly_chart(_terr_fig, use_container_width=True, config={"displayModeBar": False})

    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.plotly_chart(_c(ch.margin_waterfall(sale), _H),
                        use_container_width=True, config={"displayModeBar": False})
    with row2_right:
        # Top 10 Customers — horizontal bar, unified with ch.LAYOUT
        _top_cust_ov = (sale.groupby("Customer")
                        .agg(Revenue=("Total Excluding Tax", "sum"),
                             Profit=("Profit", "sum"))
                        .reset_index()
                        .sort_values("Revenue", ascending=False).head(10)
                        .sort_values("Revenue", ascending=True))
        _cust_max_rev = _top_cust_ov["Revenue"].max()
        _fig_cust_bar = go.Figure(go.Bar(
            y=_top_cust_ov["Customer"],
            x=_top_cust_ov["Revenue"],
            orientation="h",
            marker=dict(
                color=_top_cust_ov["Revenue"],
                colorscale=[[0, "rgba(0,212,255,0.75)"], [1, "rgba(0,50,130,1.0)"]],
                showscale=False,
            ),
            cliponaxis=False,
            text=_top_cust_ov["Revenue"].apply(lambda v: f"${v/1e3:,.0f}K"),
            textposition="outside",
            textfont=dict(size=9, color="#E0E0E0"),
            hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>",
        ))
        _fig_cust_bar.update_layout(
            **ch.LAYOUT,
            title=dict(text="🌟 Top 10 Customers by Revenue",
                       font=dict(size=13, color="#00D4FF")),
            xaxis=dict(tickformat="$,.0s", title="Revenue ($)",
                       range=[0, _cust_max_rev * 1.22]),
            yaxis_title="",
        )
        st.plotly_chart(_c(_fig_cust_bar, _H),
                        use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEMAND FORECASTING
# Row 1: KPIs + horizon slider merged in one row (saves ~50 px)
# Row 2: Forecast chart (full-width, 260 px)
# Row 3: Forecast table (full-width)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecasting":
    _ptitle("Demand Forecasting")

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

    # Row 3 — Forecast table FULL WIDTH so every column is visible without scrolling
    st.markdown('<p class="tbl-title">📋 Forecast Table</p>', unsafe_allow_html=True)
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
            use_container_width=True, height=min(len(fc_show)*35+38, 300), hide_index=True,
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
# Row 1: 5 KPIs (incl. Balanced Accuracy)
# Row 2: Risk matrix (wide) | Risk pie
# Row 3: HIGH-risk table full-width
# Row 4: MEDIUM-risk table (1 row)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Inventory Risk":
    _ptitle("Inventory Risk Intelligence")

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

    # Row 1 — KPIs (5 cards including Balanced Accuracy)
    _bal_acc_val = inv_info["auc"] if inv_info["auc"] else 0.0
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: metric_card("Total SKUs",        f"{len(inv_df):,}")
    with k2: metric_card("HIGH Risk SKUs",    f"{high}", delta_str=f"+{high} need action")
    with k3: metric_card("MEDIUM Risk SKUs",  f"{medium}")
    with k4: metric_card("Stock Value",       f"${stock_val/1e6:.1f}M")
    with k5: metric_card("Balanced Accuracy", f"{_bal_acc_val:.3f}")

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
                         color_discrete_map={"HIGH": "#FF4C6E", "MEDIUM": "#FFD700", "LOW": "#00C49A"},
                         template="plotly_dark", hole=0.30)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                              insidetextfont=dict(size=10))
        fig_pie.update_layout(
            **ch.LAYOUT,
            title=dict(text="Risk Distribution", font=dict(color="#00D4FF", size=13)),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0.5, y=-0.05,
                        xanchor="center"))
        st.plotly_chart(_c(fig_pie, 270),
                        use_container_width=True, config={"displayModeBar": False})

    # Row 3 — HIGH Risk table + MEDIUM Risk table below
    st.markdown('<p class="tbl-title">⚠️ HIGH Risk SKUs — Immediate Reorder Needed</p>', unsafe_allow_html=True)
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
    }), use_container_width=True, height=min(max(len(high_df),1)*35+38, 320), hide_index=True,
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

    # MEDIUM Risk table (1 representative row)
    st.markdown('<p class="tbl-title">🟡 MEDIUM Risk SKUs — Monitor Closely</p>', unsafe_allow_html=True)
    medium_df = (inv_df[inv_df["Predicted_Risk_Name"] == "MEDIUM"]
                 [["Stock Item","Stock Category","Quantity On Hand",
                   "Reorder Level","Target Stock Level","Monthly_Velocity",
                   "Days_Coverage","Stock Value","Lead Time Days"]]
                 .sort_values("Quantity On Hand").head(1)
                 .reset_index(drop=True))
    if not medium_df.empty:
        st.dataframe(medium_df.style.format({
            "Quantity On Hand":   "{:,.0f}",
            "Reorder Level":      "{:,.0f}",
            "Target Stock Level": "{:,.0f}",
            "Monthly_Velocity":   "{:,.1f}",
            "Days_Coverage":      "{:,.0f}",
            "Stock Value":        "${:,.0f}",
        }), use_container_width=True, height=73, hide_index=True,
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
    else:
        st.success("✅ No MEDIUM risk SKUs found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER INTELLIGENCE
# Tab Churn:       KPIs + 3-up charts (dist | feat imp | at-risk table)
# Tab Segmentation: 2-up (3D scatter | segment summary)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Intelligence":
    _ptitle("Customer Intelligence")

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

        # Compute prediction accuracy % — more interpretable than ROC-AUC for business users
        _rfm_clean = rfm.dropna(subset=["Churned", "Churn_Pred"])
        _pred_acc  = (_rfm_clean["Churned"] == _rfm_clean["Churn_Pred"]).mean() * 100

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1: metric_card("Total Customers",    f"{len(rfm):,}")
        with k2: metric_card("At-Risk",            f"{churn_count:,}",
                              delta_str=f"⚠ {churn_rate:.1f}% rate")
        with k3: metric_card("Low-Risk Customers", f"{active_cnt:,}")
        with k4: metric_card("Pred. Accuracy",     f"{_pred_acc:.1f}", suffix="%")

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
            at_risk["Churn %"]     = (at_risk["Churn %"] * 100).round(1)
            at_risk["Revenue ($)"] = at_risk["Revenue ($)"].round(0)
            _ar_h = min(len(at_risk) * 35 + 38, 280)   # dynamic height — no empty rows
            st.markdown('<p class="tbl-title">🚨 Top At-Risk Customers</p>', unsafe_allow_html=True)
            st.dataframe(at_risk.style.format({
                "Revenue ($)": "${:,.0f}", "Churn %": "{:.1f}%",
            }).background_gradient(subset=["Churn %"], cmap="RdYlGn_r"),
                use_container_width=True, height=_ar_h, hide_index=True,
                column_config={
                    "Customer":    st.column_config.TextColumn("Customer",   width="small"),
                    "Recency":     st.column_config.NumberColumn("Rec(d)",   width="small"),
                    "Frequency":   st.column_config.NumberColumn("Orders",   width="small"),
                    "Revenue ($)": st.column_config.TextColumn("Revenue",    width="small"),
                    "Churn %":     st.column_config.TextColumn("Churn Risk", width="small"),
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
        st.markdown('<p class="tbl-title">📋 Segment Summary</p>', unsafe_allow_html=True)
        st.dataframe(seg_sum.style.format({
            "Avg_Recency":   "{:.0f}",
            "Avg_Frequency": "{:.0f}",
            "Avg_Monetary":  "${:,.0f}",
        }), use_container_width=True, height=min(len(seg_sum)*35+38, 280), hide_index=True,
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
    _ptitle("Supplier Analytics & Quality Scoring")

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
    _at_risk_sup = int((sup_df["Quality_Score"] < 50).sum()) if "Quality_Score" in sup_df.columns else 0
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: metric_card("Avg Quality Score",        f"{avg_score:.0f}", suffix="/100")
    with k2: metric_card("Grade A Suppliers",        f"{grade_a}")
    with k3: metric_card("Low Fill-Rate (< 90 %)",   f"{low_fulf}")
    with k4: metric_card("⚠️ At-Risk Suppliers",     f"{_at_risk_sup}", delta_str="Score < 50")
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
        st.markdown('<p class="tbl-title">📋 Full Supplier Scorecard</p>', unsafe_allow_html=True)
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
            use_container_width=True, height=min(len(sc_df)*35+38, 400), hide_index=True,
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
        st.markdown('<p class="tbl-title">📐 Supplier Pillar Scores</p>', unsafe_allow_html=True)
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
                use_container_width=True, height=min(len(p_df)*35+38, 420), hide_index=True,
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
    _ptitle("Financial Transaction Anomaly Detection")

    with st.spinner("Running Isolation Forest (runs once, then cached in session)…"):
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
    _exp_str  = (f"${_exposure/1e9:.2f}B" if _exposure >= 1e9
                 else f"${_exposure/1e6:.1f}M" if _exposure >= 1e6
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
            **ch.LAYOUT, barmode="overlay",
            title=dict(text="Score Distribution", font=dict(color="#00D4FF", size=13)),
            xaxis_title="Score", yaxis_title="Count",
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                        x=0.5, y=1.12, xanchor="center", yanchor="top"))
        st.plotly_chart(_c(fig_score, 255),
                        use_container_width=True, config={"displayModeBar": False})
    with ac3:
        if "Payment Method" in anom_df.columns:
            pm_anom = (anom_df.loc[is_anom_mask]
                       .groupby("Payment Method")["Is_Anomaly"].count()
                       .reset_index().rename(columns={"Is_Anomaly": "Anomalies"}))
            fig_pm = px.pie(
                pm_anom, values="Anomalies", names="Payment Method",
                color_discrete_sequence=ch.PALETTE,
                template="plotly_dark", hole=0.35)
            fig_pm.update_traces(textposition="inside", textinfo="percent+label",
                                 insidetextfont=dict(size=10))
            fig_pm.update_layout(**ch.LAYOUT)
            fig_pm.update_layout(
                title=dict(text="By Payment Method", font=dict(color="#00D4FF", size=13)),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(_c(fig_pm, 255),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Payment Method column not available.")

    # Row 3 — anomalous transactions table
    st.markdown('<p class="tbl-title">⚠️ Top Anomalous Transactions</p>', unsafe_allow_html=True)
    _want_cols = ["Transaction Key", "WWI Transaction ID", "Date Key",
                  "Payment Method", "Total Including Tax", "Outstanding Balance",
                  "Transaction Type", "Anomaly_Score"]
    _have_cols = [c for c in _want_cols if c in anom_df.columns]
    top_anom = (anom_df.loc[is_anom_mask]
                .sort_values("Anomaly_Score").head(100)[_have_cols]).copy()
    # Strip time component so the Date column shows a pure date
    if "Date Key" in top_anom.columns:
        top_anom["Date Key"] = pd.to_datetime(top_anom["Date Key"], errors="coerce").dt.date
    fmt_cols = {}
    if "Total Including Tax"  in top_anom.columns: fmt_cols["Total Including Tax"]  = "${:,.2f}"
    if "Outstanding Balance"  in top_anom.columns: fmt_cols["Outstanding Balance"]  = "${:,.2f}"
    if "Anomaly_Score"        in top_anom.columns: fmt_cols["Anomaly_Score"]        = "{:.4f}"

    _col_cfg = {}
    if "Transaction Key"     in _have_cols: _col_cfg["Transaction Key"]     = st.column_config.NumberColumn("Txn Key",          width="small")
    if "WWI Transaction ID"  in _have_cols: _col_cfg["WWI Transaction ID"]  = st.column_config.NumberColumn("WWI Txn ID",       width="small")
    if "Date Key"            in _have_cols: _col_cfg["Date Key"]            = st.column_config.TextColumn("Date",               width="small")
    if "Payment Method"      in _have_cols: _col_cfg["Payment Method"]      = st.column_config.TextColumn("Payment",            width="medium")
    if "Total Including Tax"  in _have_cols: _col_cfg["Total Including Tax"] = st.column_config.TextColumn("Total incl. Tax",   width="medium")
    if "Outstanding Balance"  in _have_cols: _col_cfg["Outstanding Balance"] = st.column_config.TextColumn("Outstanding",       width="medium")
    if "Transaction Type"    in _have_cols: _col_cfg["Transaction Type"]    = st.column_config.TextColumn("Txn Type",           width="medium")
    if "Anomaly_Score"       in _have_cols: _col_cfg["Anomaly_Score"]       = st.column_config.NumberColumn("Score",            width="small", format="%.4f")

    st.dataframe(top_anom.style.format(fmt_cols)
                 .background_gradient(subset=["Anomaly_Score"] if "Anomaly_Score" in top_anom.columns else [],
                                      cmap="Reds_r"),
        use_container_width=True, height=min(len(top_anom)*35+38, 400), hide_index=True,
        column_config=_col_cfg)