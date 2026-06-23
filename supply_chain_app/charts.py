"""
Chart utilities — all Plotly, dark theme, supply-chain palette.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

PALETTE = ["#00D4FF","#7B2FBE","#FF6B35","#00C49A","#FFD700",
           "#FF4C6E","#3DBEFF","#A855F7","#34D399","#F59E0B"]

LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(family="Inter, sans-serif", size=12, color="#E0E0E0"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _fig(fig, title=""):
    fig.update_layout(**LAYOUT, title=dict(text=title, font=dict(size=15, color="#00D4FF")))
    return fig


# ── KPI tiles ─────────────────────────────────────────────────────────────────
def kpi_tile(value, label, delta=None, prefix="", suffix=""):
    indicator = go.Figure(go.Indicator(
        mode="number+delta" if delta is not None else "number",
        value=float(value),
        delta={"reference": float(value) - float(delta), "valueformat": ".2%"} if delta else None,
        number={"prefix": prefix, "suffix": suffix,
                "font": {"size": 42, "color": "#00D4FF"}},
        title={"text": label, "font": {"size": 14, "color": "#AAB4BE"}},
    ))
    indicator.update_layout(**LAYOUT, height=140, margin=dict(l=10,r=10,t=10,b=10))
    return indicator


# ── Revenue over time ─────────────────────────────────────────────────────────
def revenue_timeline(sale_df):
    agg = (sale_df.groupby(["Calendar Year","Calendar Month Number"])
           .agg(Revenue=("Total Excluding Tax","sum"),
                Profit=("Profit","sum"))
           .reset_index()
           .assign(Period=lambda d: pd.to_datetime(
               d["Calendar Year"].astype(str) + "-" +
               d["Calendar Month Number"].astype(str).str.zfill(2))))
    agg = agg.sort_values("Period")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=agg["Period"], y=agg["Revenue"],
                         name="Revenue", marker_color="#00D4FF", opacity=0.8))
    fig.add_trace(go.Scatter(x=agg["Period"], y=agg["Profit"],
                             name="Profit", line=dict(color="#FF6B35", width=2)),
                  secondary_y=True)
    fig.update_layout(**LAYOUT, title=dict(text="Revenue & Profit Over Time",
                                           font=dict(size=15, color="#00D4FF")),
                      legend=dict(bgcolor="rgba(0,0,0,0)",
                                  orientation="h",
                                  x=0.5, y=1.22,
                                  xanchor="center", yanchor="top"))
    return fig


# ── Sales by category ─────────────────────────────────────────────────────────
def sales_by_category(sale_df):
    agg = (sale_df.groupby("Stock Category")
           .agg(Revenue=("Total Excluding Tax","sum"),
                Profit=("Profit","sum"),
                Units=("Quantity","sum"))
           .reset_index()
           .sort_values("Revenue", ascending=True))
    fig = go.Figure()
    fig.add_trace(go.Bar(y=agg["Stock Category"], x=agg["Revenue"],
                         orientation="h", name="Revenue",
                         marker_color="#00D4FF", opacity=0.85))
    fig.add_trace(go.Bar(y=agg["Stock Category"], x=agg["Profit"],
                         orientation="h", name="Profit",
                         marker_color="#00C49A", opacity=0.85))
    fig.update_layout(**LAYOUT, barmode="overlay",
                      title=dict(text="Revenue vs Profit by Category",
                                 font=dict(size=15,color="#00D4FF")),
                      xaxis_title="Amount ($)",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


# ── Forecast chart ────────────────────────────────────────────────────────────
def forecast_chart(info, selected_categories):
    fc   = info["forecast_df"]
    hist = info["agg_df"]

    fig = go.Figure()
    colors = PALETTE
    for i, cat in enumerate(selected_categories[:8]):
        c = colors[i % len(colors)]
        h = hist[hist["Stock Category"]==cat].sort_values(["Calendar Year","Calendar Month Number"])
        f = fc[fc["Stock Category"]==cat].sort_values(["Forecast Year","Forecast Month"])

        h_dates = pd.to_datetime(
            h["Calendar Year"].astype(str)+"-"+h["Calendar Month Number"].astype(str).str.zfill(2))
        f_dates = pd.to_datetime(
            f["Forecast Year"].astype(str)+"-"+f["Forecast Month"].astype(str).str.zfill(2))

        fig.add_trace(go.Scatter(x=h_dates, y=h["Total_Qty"],
                                  name=f"{cat} (Actual)", line=dict(color=c, width=1.5)))
        fig.add_trace(go.Scatter(x=f_dates, y=f["Predicted_Qty"],
                                  name=f"{cat} (Forecast)",
                                  line=dict(color=c, width=2.5, dash="dot")))

    fig.update_layout(**LAYOUT,
                      title=dict(text="Demand Forecast by Stock Category",
                                 font=dict(size=15,color="#00D4FF")),
                      xaxis_title="Period", yaxis_title="Units",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


# ── Inventory risk matrix ─────────────────────────────────────────────────────
def inventory_risk_scatter(inv_df):
    """
    Three explicit traces — LOW (small, background), MEDIUM (medium), HIGH (large, foreground).
    Drawing order LOW→MEDIUM→HIGH guarantees red/yellow dots are always on top.
    Size is fixed per risk level so a $33 HIGH-risk SKU is just as visible as a $1M LOW-risk one.
    (The old px.scatter with size='Stock Value' made HIGH items microscopically small.)
    """
    df = inv_df.copy().dropna(subset=["Quantity On Hand", "Monthly_Velocity"])
    risk_col = "Predicted_Risk_Name" if "Predicted_Risk_Name" in df.columns else "Risk_Label_Name"

    cfg = {
        "LOW":    dict(color="#00C49A", size=8,  opacity=0.55, symbol="circle"),
        "MEDIUM": dict(color="#FFD700", size=13, opacity=0.80, symbol="diamond"),
        "HIGH":   dict(color="#FF4C6E", size=18, opacity=1.00, symbol="circle"),
    }

    fig = go.Figure()
    for level in ["LOW", "MEDIUM", "HIGH"]:          # LOW first → HIGH on top
        sub = df[df[risk_col] == level]
        if sub.empty:
            continue
        c = cfg[level]
        hover = (
            "<b>" + sub["Stock Item"].fillna("").astype(str) + "</b><br>"
            "Category: "    + sub["Stock Category"].fillna("").astype(str) + "<br>"
            "QoH: "         + sub["Quantity On Hand"].apply(lambda v: f"{v:,.0f}") + "<br>"
            "Mo.Velocity: " + sub["Monthly_Velocity"].apply(lambda v: f"{v:,.1f}") + "<br>"
            "Days Cover: "  + sub["Days_Coverage"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "N/A")
        )
        fig.add_trace(go.Scatter(
            x=sub["Quantity On Hand"], y=sub["Monthly_Velocity"],
            mode="markers",
            name=level,
            marker=dict(color=c["color"], size=c["size"],
                        opacity=c["opacity"], symbol=c["symbol"],
                        line=dict(color="white", width=0.5) if level == "HIGH" else dict(width=0)),
            hovertemplate=hover + "<extra></extra>",
        ))

    fig.update_layout(**LAYOUT,
                      title=dict(text="Inventory Risk Matrix: Stock vs Velocity",
                                 font=dict(size=15, color="#00D4FF")),
                      xaxis_title="Stock on Hand",
                      yaxis_title="Monthly Velocity (units)",
                      legend=dict(bgcolor="rgba(0,0,0,0)", title="Risk Level"))
    return fig


# ── Top Sales Territory ───────────────────────────────────────────────────────
def top_territory_chart(sale_df):
    """Horizontal bar — revenue by sales territory (top 8), theme-aligned colors."""
    terr = (sale_df.groupby("Sales Territory")
            .agg(Revenue=("Total Excluding Tax", "sum"),
                 Profit=("Profit", "sum"))
            .reset_index()
            .dropna(subset=["Sales Territory"])
            .sort_values("Revenue", ascending=True)
            .tail(8))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=terr["Sales Territory"], x=terr["Revenue"],
        orientation="h", name="Revenue",
        marker=dict(
            color=terr["Revenue"],
            colorscale=[[0, "rgba(0,212,255,0.75)"], [1, "rgba(0,50,130,1.0)"]],
            showscale=False,
        ),
        cliponaxis=False,
        text=terr["Revenue"].apply(lambda v: f"${v/1e6:.1f}M"),
        textposition="outside",
        textfont=dict(size=10, color="#E0E0E0"),
    ))
    max_rev = terr["Revenue"].max()
    fig.update_layout(**LAYOUT,
                      title=dict(text="Revenue by Sales Territory",
                                 font=dict(size=15, color="#00D4FF")),
                      xaxis_title="Revenue ($)", yaxis_title="",
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      xaxis=dict(tickformat="$,.0s",
                                 range=[0, max_rev * 1.20]))
    return fig


# ── Supplier scoreboard ───────────────────────────────────────────────────────
def supplier_scoreboard(sup_df):
    df = sup_df.sort_values("Quality_Score", ascending=True).tail(20)
    colors = df["Quality_Score"].apply(
        lambda s: "#00C49A" if s>=80 else "#FFD700" if s>=65 else "#FF4C6E")
    fig = go.Figure(go.Bar(
        y=df["Supplier"], x=df["Quality_Score"],
        orientation="h", marker_color=colors,
        text=df["Grade"], textposition="outside",
    ))
    fig.update_layout(**LAYOUT,
                      title=dict(text="Supplier Quality Scores (0-100)",
                                 font=dict(size=15,color="#00D4FF")),
                      xaxis=dict(range=[0,110]), yaxis_title="",
                      xaxis_title="Quality Score")
    return fig


# ── Anomaly scatter ───────────────────────────────────────────────────────────
def anomaly_chart(txn_df):
    """
    WebGL scatter (go.Scattergl) with normal transactions sampled to max 3 000 pts.

    Root cause of browser freeze: SVG scatter with 94 K points creates 94 K DOM
    nodes — any browser will hang.  Two fixes:
      1. go.Scattergl renders via WebGL, not SVG → can handle 1 M+ points.
      2. Normal transactions are random-sampled to 3 000 for display; ALL
         anomalies are always shown (typically ~4 950 rows).
    """
    df = txn_df.copy()
    is_anom   = df["Is_Anomaly"].astype(bool)
    normal    = df[~is_anom]
    anomalous = df[is_anom]

    # Sample normals — display 3 K representative background points
    _MAX_NORMAL = 3_000
    if len(normal) > _MAX_NORMAL:
        normal = normal.sample(_MAX_NORMAL, random_state=42)

    x_col = "Total Including Tax"
    y_col = "Outstanding Balance"

    fig = go.Figure()

    # ① Normal — visible diamond markers (background)
    fig.add_trace(go.Scattergl(
        x=normal[x_col], y=normal[y_col],
        mode="markers",
        name=f"Normal (sample of {_MAX_NORMAL:,})",
        marker=dict(color="#00D4FF", size=6, opacity=0.45, symbol="diamond"),
        hovertemplate=(
            "<b>Normal</b><br>"
            "Total incl. Tax: $%{x:,.0f}<br>"
            "Outstanding Balance: $%{y:,.0f}<extra></extra>"
        ),
    ))

    # ② Anomalies — vivid red WebGL dots (foreground, all shown)
    fig.add_trace(go.Scattergl(
        x=anomalous[x_col], y=anomalous[y_col],
        mode="markers",
        name=f"🚨 Anomaly ({len(anomalous):,})",
        marker=dict(color="#FF4C6E", size=6, opacity=0.90),
        hovertemplate=(
            "<b>🚨 Anomaly</b><br>"
            "Total incl. Tax: $%{x:,.0f}<br>"
            "Outstanding Balance: $%{y:,.0f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **LAYOUT,
        title=dict(text="Transaction Anomaly Detection",
                   font=dict(size=15, color="#00D4FF")),
        xaxis_title="Total Including Tax ($)",
        yaxis_title="Outstanding Balance ($)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Customer RFM 3D ───────────────────────────────────────────────────────────
def rfm_3d(rfm_df):
    seg_colors = {
        "Champions":"#00C49A", "Loyal Customers":"#00D4FF",
        "At-Risk Customers":"#FFD700", "Churned/Lost":"#FF4C6E",
    }
    fig = px.scatter_3d(
        rfm_df, x="Recency", y="Frequency", z="Monetary",
        color="Segment Name",
        color_discrete_map=seg_colors,
        opacity=0.75, size_max=6,
        hover_data=["Customer","Region"],
    )
    fig.update_layout(**LAYOUT,
                      title=dict(text="Customer RFM Segmentation (3D)",
                                 font=dict(size=15,color="#00D4FF")),
                      scene=dict(bgcolor="#0E1117"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


# ── Feature importance bar ────────────────────────────────────────────────────
def feat_importance_chart(feat_imp, title="Feature Importance"):
    fi = feat_imp.sort_values("Importance")
    fig = go.Figure(go.Bar(
        y=fi["Feature"], x=fi["Importance"],
        orientation="h",
        marker=dict(color=fi["Importance"], colorscale="Viridis"),
    ))
    fig.update_layout(**LAYOUT,
                      title=dict(text=title, font=dict(size=14,color="#00D4FF")),
                      xaxis_title="Importance", height=300)
    return fig


# ── Churn probability distribution ───────────────────────────────────────────
def churn_dist(rfm_df):
    """
    Distribution of churn probabilities split by MODEL PREDICTION (Churn_Pred),
    not the training label (Churned).  This ensures the chart matches the
    At-Risk / Low-Risk KPI numbers shown above it.
    Note: Segment Summary uses unsupervised clustering — those counts are
    independent and will naturally differ.
    """
    active  = rfm_df[rfm_df["Churn_Pred"] == 0]["Churn_Prob"]
    at_risk = rfm_df[rfm_df["Churn_Pred"] == 1]["Churn_Prob"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=active,   name=f"Predicted Active ({len(active):,})",
        marker_color="#00C49A", opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(
        x=at_risk,  name=f"Predicted At-Risk ({len(at_risk):,})",
        marker_color="#FF4C6E", opacity=0.8, nbinsx=30))
    fig.update_layout(**LAYOUT, barmode="overlay",
                      title=dict(text="Churn Probability — Model Predictions",
                                 font=dict(size=15, color="#00D4FF")),
                      xaxis_title="Churn Probability",
                      yaxis_title="Customer Count",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


# ── Fulfillment heatmap ───────────────────────────────────────────────────────
def fulfillment_heatmap(purchase_df):
    df = purchase_df.copy()
    df = df.dropna(subset=["Supplier","Year","Fulfillment Rate"])
    pivot = df.pivot_table(
        values="Fulfillment Rate", index="Supplier",
        columns="Year", aggfunc="mean").fillna(0)

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.astype(str), y=pivot.index,
        colorscale="RdYlGn", zmin=0, zmax=100,
        hovertemplate="Supplier: %{y}<br>Year: %{x}<br>Fulfillment: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**LAYOUT,
                      title=dict(text="Supplier Fulfillment Rate Heatmap (%)",
                                 font=dict(size=15,color="#00D4FF")),
                      height=max(400, len(pivot)*22))
    return fig


# ── Margin waterfall ──────────────────────────────────────────────────────────
def margin_waterfall(sale_df):
    cats = (sale_df.groupby("Stock Category")
            .agg(Revenue=("Total Excluding Tax","sum"),
                 Profit=("Profit","sum"))
            .assign(COGS=lambda d: d["Revenue"]-d["Profit"])
            .reset_index()
            .sort_values("Revenue", ascending=False).head(8))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Revenue", x=cats["Stock Category"],
                         y=cats["Revenue"], marker_color="#00D4FF"))
    fig.add_trace(go.Bar(name="COGS", x=cats["Stock Category"],
                         y=-cats["COGS"], marker_color="#FF4C6E"))
    fig.add_trace(go.Bar(name="Profit", x=cats["Stock Category"],
                         y=cats["Profit"], marker_color="#00C49A"))

    fig.update_layout(**LAYOUT, barmode="overlay",
                      title=dict(text="Revenue / COGS / Profit by Category",
                                 font=dict(size=15,color="#00D4FF")),
                      yaxis_title="Amount ($)",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig