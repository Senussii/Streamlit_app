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
                                  x=0.01, y=0.99,
                                  xanchor="left", yanchor="top"))
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
    # Drop rows missing the key axes; Monthly_Velocity is already populated
    # by build_stockout_classifier (0 only when no movement matched the SKU).
    df = inv_df.copy().dropna(subset=["Quantity On Hand", "Monthly_Velocity"])
    color_map = {"LOW": "#00C49A", "MEDIUM": "#FFD700", "HIGH": "#FF4C6E"}
    risk_col = "Predicted_Risk_Name" if "Predicted_Risk_Name" in df.columns else "Risk_Label_Name"
    fig = px.scatter(
        df, x="Quantity On Hand", y="Monthly_Velocity",
        color=risk_col,
        color_discrete_map=color_map,
        size="Stock Value", size_max=30,
        hover_data=["Stock Item", "Stock Category", "Days_Coverage"],
        labels={"Quantity On Hand": "Stock on Hand",
                "Monthly_Velocity": "Monthly Velocity (units)"},
        category_orders={risk_col: ["LOW", "MEDIUM", "HIGH"]},
    )
    fig.update_layout(**LAYOUT,
                      title=dict(text="Inventory Risk Matrix: Stock vs Velocity",
                                 font=dict(size=15, color="#00D4FF")),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
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
    Plot ALL transactions — Normal as a faint blue cloud, Anomalies as bright
    red dots on top.  Two explicit go.Scatter traces are used instead of
    px.scatter so there is zero ambiguity about colour/opacity mapping:
    px.scatter serialises boolean Is_Anomaly to strings and combines marker
    opacity with any alpha already in the colour string, both of which caused
    the Normal trace to be essentially invisible (effective alpha ~0.2).
    """
    df = txn_df.copy()
    # coerce Is_Anomaly to plain bool regardless of original dtype
    is_anom = df["Is_Anomaly"].astype(bool)
    normal   = df[~is_anom]
    anomalous = df[is_anom]

    fig = go.Figure()

    # ① Normal transactions — small, faint, rendered first (background)
    fig.add_trace(go.Scatter(
        x=normal["Total Including Tax"],
        y=normal["Outstanding Balance"],
        mode="markers",
        name=f"Normal ({len(normal):,})",
        marker=dict(color="#00D4FF", size=4, opacity=0.25),
        hovertemplate=(
            "<b>Normal</b><br>"
            "Total incl. Tax: $%{x:,.0f}<br>"
            "Outstanding Balance: $%{y:,.0f}<extra></extra>"
        ),
    ))

    # ② Anomalous transactions — larger, vivid red, rendered on top
    fig.add_trace(go.Scatter(
        x=anomalous["Total Including Tax"],
        y=anomalous["Outstanding Balance"],
        mode="markers",
        name=f"Anomaly ({len(anomalous):,})",
        marker=dict(color="#FF4C6E", size=7, opacity=0.85,
                    line=dict(color="#FF4C6E", width=0.5)),
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
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rfm_df[rfm_df["Churned"]==0]["Churn_Prob"],
        name="Active", marker_color="#00C49A", opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(
        x=rfm_df[rfm_df["Churned"]==1]["Churn_Prob"],
        name="Churned", marker_color="#FF4C6E", opacity=0.7, nbinsx=30))
    fig.update_layout(**LAYOUT, barmode="overlay",
                      title=dict(text="Churn Probability Distribution",
                                 font=dict(size=15,color="#00D4FF")),
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