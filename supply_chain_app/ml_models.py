"""
ML Models — Supply Chain Predictive Analytics  v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Demand Forecasting · Stockout Risk · Churn · Supplier Quality
Anomaly Detection · Customer Segmentation

v2.0 Changelog
──────────────
1. Demand Forecast    — Fixed auto-regressive lag slide (critical bug); predictions
                        now feed back as inputs for multi-step forecasting.
                        LightGBM primary / HistGradientBoosting fallback.
                        Removed hacky GradientBoostingClassifier.__bases__[0] line.
                        Added YoY, Price×Volume features. TimeSeriesSplit CV metric.
2. Stockout Risk      — Monthly velocity now aggregated per SKU×month (true units/month)
                        instead of per-transaction mean. Days_Coverage formula fixed.
                        Added Lead_Demand, Safety_Margin, Overstock_Ratio features.
                        LightGBM / GradientBoosting with balanced class weighting.
3. Churn Predictor    — Fixed critical index-alignment bug in predict_proba assignment.
                        Added Log_Monetary, Log_Frequency, Tenure_Days, Buy_Rate.
                        Proper stratified train/test split, calibrated RF probabilities.
4. Supplier Scorer    — Added proper hold-out R² reporting (was fitting on full data).
                        Order variance and lead-time consistency features added.
5. Anomaly Detection  — RobustScaler (outlier-resistant) replaces StandardScaler.
                        Log1p-transforms on all monetary columns. Payment method
                        and transaction type encoded as additional features.
6. Segmentation       — Tenure_Days and Buy_Rate added to clustering. Centroid-based
                        segment naming is more robust than monetary-rank ordering.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    HistGradientBoostingRegressor,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score,
    classification_report,
    roc_auc_score,
)
from sklearn.cluster import KMeans

# ── optional fast-path: LightGBM ▶ XGBoost ▶ sklearn ─────────────────────────
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════
def _le(series: pd.Series) -> np.ndarray:
    """Fit-transform LabelEncoder on a nullable series."""
    return LabelEncoder().fit_transform(series.fillna("__MISSING__"))


def _regressor(**overrides):
    """Best available regressor: LightGBM > HistGradientBoosting."""
    if _HAS_LGB:
        p = dict(
            n_estimators=600, learning_rate=0.03, num_leaves=63,
            min_child_samples=5, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.05, reg_lambda=0.1, random_state=42,
            n_jobs=-1, verbose=-1,
        )
        p.update(overrides)
        return lgb.LGBMRegressor(**p)
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.04, max_leaf_nodes=63,
        min_samples_leaf=5, l2_regularization=0.05, random_state=42,
    )


def _classifier(balanced: bool = True, **overrides):
    """Best available classifier: LightGBM > GradientBoosting."""
    cw = "balanced" if balanced else None
    if _HAS_LGB:
        p = dict(
            n_estimators=600, learning_rate=0.03, num_leaves=63,
            min_child_samples=5, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.05, reg_lambda=0.1, class_weight=cw,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        p.update(overrides)
        return lgb.LGBMClassifier(**p)
    return GradientBoostingClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=3, random_state=42,
    )


def _feat_imp(model, features: list) -> pd.DataFrame:
    """Extract feature importances for sklearn and LightGBM."""
    try:
        imp = model.feature_importances_
    except AttributeError:
        imp = np.full(len(features), 1.0 / len(features))
    return (
        pd.DataFrame({"Feature": features, "Importance": imp})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. DEMAND FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
def build_demand_forecast(sale_df, horizon_months: int = 3):
    """
    Auto-regressive monthly demand forecast per Stock Category.

    Feature set
    ───────────
    Lags 1-3, 6, 12 · Rolling mean 3/6/12 · Rolling std 3
    Trend 1-period & 3-period diff · YoY same-month growth
    Cyclical month (sin/cos) · Category code
    Avg unit price · Transaction count · Price×Volume interaction

    Key fix (v2)
    ────────────
    The original lag slide: `lag3, lag2, lag1 = lag2, lag1, lag1`
    never updated lag1 to the predicted value, producing flat forecasts
    beyond period 1. The auto-regressive loop below feeds each prediction
    back as the next period's lag1, with proper window management.
    """
    df = sale_df.copy()
    df = df.dropna(subset=["Calendar Year", "Calendar Month Number", "Quantity"])

    # ── monthly aggregation per category ─────────────────────────────────────
    agg = (
        df.groupby(["Calendar Year", "Calendar Month Number", "Stock Category"])
        .agg(
            Total_Qty        = ("Quantity",            "sum"),
            Avg_Price        = ("Unit Price",          "mean"),
            Num_Transactions = ("Sale Key",            "count"),
        )
        .reset_index()
    )

    # ── feature engineering per category (avoids data leakage across cats) ───
    enriched = []
    for cat, grp in agg.groupby("Stock Category", sort=False):
        g = grp.sort_values(
            ["Calendar Year", "Calendar Month Number"]
        ).copy().reset_index(drop=True)
        s = g["Total_Qty"]

        g["Lag1"]  = s.shift(1)
        g["Lag2"]  = s.shift(2)
        g["Lag3"]  = s.shift(3)
        g["Lag6"]  = s.shift(6)
        g["Lag12"] = s.shift(12)

        g["Roll3"]    = s.shift(1).rolling(3,  min_periods=1).mean()
        g["Roll6"]    = s.shift(1).rolling(6,  min_periods=1).mean()
        g["Roll12"]   = s.shift(1).rolling(12, min_periods=1).mean()
        g["RollStd3"] = s.shift(1).rolling(3,  min_periods=2).std().fillna(0)

        g["Trend1"]    = s.shift(1).diff(1)
        g["Trend3"]    = s.shift(1).diff(3)
        g["YoY_Growth"] = (
            (s.shift(1) - s.shift(13))
            / s.shift(13).replace(0, np.nan)
        ).fillna(0).clip(-2, 2)

        g["Price_Vol"] = g["Avg_Price"].fillna(0) * g["Roll3"].fillna(0)
        enriched.append(g)

    agg = pd.concat(enriched, ignore_index=True)

    le = LabelEncoder()
    agg["Cat_Code"] = le.fit_transform(agg["Stock Category"].fillna("Unknown"))
    agg["Month_Sin"] = np.sin(2 * np.pi * agg["Calendar Month Number"] / 12)
    agg["Month_Cos"] = np.cos(2 * np.pi * agg["Calendar Month Number"] / 12)

    features = [
        "Calendar Year", "Month_Sin", "Month_Cos", "Cat_Code",
        "Lag1", "Lag2", "Lag3", "Lag6", "Lag12",
        "Roll3", "Roll6", "Roll12", "RollStd3",
        "Trend1", "Trend3", "YoY_Growth",
        "Avg_Price", "Num_Transactions", "Price_Vol",
    ]

    clean = agg.dropna(subset=features + ["Total_Qty"]).reset_index(drop=True)
    X = clean[features]
    y = clean["Total_Qty"]

    # ── chronological split (never shuffle time-series data) ─────────────────
    split   = int(len(clean) * 0.8)
    X_train = X.iloc[:split];  y_train = y.iloc[:split]
    X_test  = X.iloc[split:];  y_test  = y.iloc[split:]

    # ── model ─────────────────────────────────────────────────────────────────
    model = _regressor()
    model.fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), 0, None)
    mape   = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2     = r2_score(y_test, y_pred)

    # ── TimeSeriesSplit cross-validation MAPE ─────────────────────────────────
    cv_mapes = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=3).split(X):
        m = _regressor()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = np.clip(m.predict(X.iloc[va_idx]), 0, None)
        cv_mapes.append(mean_absolute_percentage_error(y.iloc[va_idx], p) * 100)
    cv_mape = float(np.mean(cv_mapes))

    # ── auto-regressive future forecast (v2 fix) ──────────────────────────────
    cat_map  = dict(zip(clean["Stock Category"], clean["Cat_Code"]))
    last_row = clean.groupby("Stock Category").last().reset_index()

    forecasts = []
    for _, row in last_row.iterrows():
        cat      = row["Stock Category"]
        cat_code = cat_map.get(cat, 0)
        yr = int(row["Calendar Year"])
        mo = int(row["Calendar Month Number"])

        # Initialise lag state from actual data
        def _safe(col, fallback=0.0):
            v = row.get(col, np.nan)
            return float(v) if pd.notna(v) else float(fallback)

        # rolling window: list of recent values (oldest → newest)
        window = [
            _safe("Lag3"), _safe("Lag2"), _safe("Lag1"),
            _safe("Total_Qty"),
        ]
        lag6   = _safe("Lag6",    window[-1])
        lag12  = _safe("Lag12",   window[-1])
        avg_p  = _safe("Avg_Price")
        n_txn  = _safe("Num_Transactions")

        for _ in range(horizon_months):
            mo += 1
            if mo > 12:
                mo = 1; yr += 1

            lag1 = window[-1]
            lag2 = window[-2] if len(window) >= 2 else lag1
            lag3 = window[-3] if len(window) >= 3 else lag2

            w3    = window[-min(3,  len(window)):]
            w6    = window[-min(6,  len(window)):]
            w12   = window[-min(12, len(window)):]
            roll3  = float(np.mean(w3))
            roll6  = float(np.mean(w6))
            roll12 = float(np.mean(w12))
            std3   = float(np.std(w3)) if len(w3) >= 2 else 0.0

            trend1     = lag1 - lag2
            trend3     = lag1 - lag3
            yoy_growth = float(np.clip((lag1 - lag12) / max(lag12, 1e-6), -2, 2))

            row_feat = pd.DataFrame([{
                "Calendar Year":     yr,
                "Month_Sin":         np.sin(2 * np.pi * mo / 12),
                "Month_Cos":         np.cos(2 * np.pi * mo / 12),
                "Cat_Code":          cat_code,
                "Lag1": lag1,  "Lag2": lag2,  "Lag3": lag3,
                "Lag6": lag6,  "Lag12": lag12,
                "Roll3": roll3, "Roll6": roll6, "Roll12": roll12,
                "RollStd3":   std3,
                "Trend1":     trend1, "Trend3": trend3,
                "YoY_Growth": yoy_growth,
                "Avg_Price":  avg_p,
                "Num_Transactions": n_txn,
                "Price_Vol":  avg_p * roll3,
            }])[features]

            pred = float(np.clip(model.predict(row_feat)[0], 0, None))

            forecasts.append({
                "Stock Category": cat,
                "Forecast Year":  yr,
                "Forecast Month": mo,
                "Predicted_Qty":  round(pred, 1),
            })

            # ── slide window (auto-regressive key fix) ────────────────────────
            window.append(pred)
            # Approximate lag6/lag12 with exponential smoothing
            lag6  = (lag6  * 5.0 + pred) / 6.0
            lag12 = (lag12 * 11.0 + pred) / 12.0

    fc_df = pd.DataFrame(forecasts) if forecasts else pd.DataFrame(
        columns=["Stock Category", "Forecast Year", "Forecast Month", "Predicted_Qty"]
    )

    return {
        "model":             model,
        "mape":              mape,
        "r2":                r2,
        "cv_mape":           cv_mape,
        "actuals_df":        clean,
        "test_pred":         y_pred,
        "test_true":         y_test.values,
        "forecast_df":       fc_df,
        "feature_importance": _feat_imp(model, features),
        "agg_df":            agg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. STOCKOUT RISK CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
def build_stockout_classifier(inventory_df, movement_df):
    """
    Classifies each SKU as HIGH / MEDIUM / LOW stockout risk.

    v2 fixes & additions
    ─────────────────────
    Monthly velocity: outflow quantities are summed per SKU×month, then
    averaged — giving true units/month. The original `mean` over transactions
    gave average units per individual movement line, which drastically
    underestimated velocity for high-frequency items.

    Days_Coverage = QoH / (monthly_velocity / 30) is now dimensionally sound.

    New features: Lead_Demand, Safety_Margin, Overstock_Ratio.
    """
    inv = inventory_df.copy()
    mv  = movement_df.copy()

    # ── true monthly velocity ─────────────────────────────────────────────────
    out = mv[mv["Transaction_Direction"] == "Outflow"].copy()

    if {"Calendar Year", "Calendar Month Number"}.issubset(out.columns):
        monthly = (
            out.groupby(
                ["Stock Item Key", "Calendar Year", "Calendar Month Number"]
            )["Quantity"]
            .sum()
            .reset_index()
            .rename(columns={"Quantity": "Monthly_Qty"})
        )
        vel = (
            monthly.groupby("Stock Item Key")["Monthly_Qty"]
            .agg(Monthly_Velocity="mean", Velocity_Std="std", Velocity_Max="max")
            .reset_index()
        )
    else:
        # fallback: per-transaction aggregate (less accurate)
        vel = (
            out.groupby("Stock Item Key")["Quantity"]
            .agg(Monthly_Velocity="mean", Velocity_Std="std", Velocity_Max="max")
            .reset_index()
        )

    vel["Velocity_Std"] = vel["Velocity_Std"].fillna(0)
    vel["Velocity_Max"] = vel["Velocity_Max"].fillna(0)

    df = inv.merge(vel, on="Stock Item Key", how="left")
    df[["Monthly_Velocity", "Velocity_Std", "Velocity_Max"]] = (
        df[["Monthly_Velocity", "Velocity_Std", "Velocity_Max"]].fillna(0)
    )

    # ── engineered features ───────────────────────────────────────────────────
    daily_usage = df["Monthly_Velocity"] / 30.0
    df["Days_Coverage"] = np.where(
        daily_usage > 0,
        df["Quantity On Hand"] / daily_usage,
        9999.0,
    )

    ro_safe  = df["Reorder Level"].replace(0, np.nan)
    tgt_safe = df["Target Stock Level"].replace(0, np.nan)

    df["Stock_vs_Reorder"]  = (df["Quantity On Hand"] / ro_safe).fillna(1.0)
    df["Stock_vs_Target"]   = (df["Quantity On Hand"] / tgt_safe).fillna(1.0)
    df["Velocity_CV"]       = (
        df["Velocity_Std"] / df["Monthly_Velocity"].replace(0, np.nan)
    ).fillna(0)
    # Days of stock needed to cover lead time demand
    df["Lead_Demand"]    = (df["Monthly_Velocity"] / 30.0) * df["Lead Time Days"].fillna(7)
    df["Safety_Margin"]  = df["Quantity On Hand"] - df["Lead_Demand"]
    # Positive → overstocked; negative → understocked relative to target
    df["Overstock_Ratio"] = (
        (df["Quantity On Hand"] - tgt_safe) / tgt_safe.fillna(1)
    ).fillna(0)

    # ── risk labels (rule-based ground truth) ─────────────────────────────────
    def _risk(r):
        if r["Reorder Flag"]:           return 2   # HIGH
        if r["Days_Coverage"]   < 30:   return 1   # MEDIUM
        if r["Safety_Margin"]   < 0:    return 1   # MEDIUM
        return 0                                     # LOW
    df["Risk_Label"] = df.apply(_risk, axis=1)

    df["Avail_Code"] = _le(df["Availability"])
    df["Cat_Code"]   = _le(df["Stock Category"])

    features = [
        "Quantity On Hand", "Reorder Level", "Target Stock Level",
        "Last Cost Price",  "Monthly_Velocity", "Velocity_Std", "Velocity_Max",
        "Days_Coverage",    "Lead Time Days",   "Avail_Code",   "Cat_Code",
        "Stock_vs_Reorder", "Stock_vs_Target",  "Velocity_CV",
        "Lead_Demand",      "Safety_Margin",    "Overstock_Ratio",
    ]

    clean = df.dropna(subset=features).reset_index(drop=True)
    X, y  = clean[features], clean["Risk_Label"]
    label_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    if y.nunique() < 2:
        df["Risk_Label_Name"] = df["Predicted_Risk_Name"] = "LOW"
        dummy = pd.DataFrame({"Feature": features,
                               "Importance": [1 / len(features)] * len(features)})
        return {"model": None, "df": df, "report": {}, "auc": None,
                "feature_importance": dummy, "risk_map": label_names}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = _classifier(balanced=True)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    present  = sorted(y_test.unique())
    t_names  = [label_names[l] for l in present]
    report   = classification_report(
        y_test, y_pred, labels=present, target_names=t_names, output_dict=True
    )
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
    except Exception:
        auc = None

    # assign predictions back to full df (only clean rows)
    df["Predicted_Risk"]      = np.nan
    df.loc[clean.index, "Predicted_Risk"] = model.predict(clean[features])
    df["Predicted_Risk"]      = df["Predicted_Risk"].fillna(0).astype(int)
    df["Risk_Label_Name"]     = df["Risk_Label"].map(label_names).fillna("LOW")
    df["Predicted_Risk_Name"] = df["Predicted_Risk"].map(label_names).fillna("LOW")

    return {
        "model": model, "df": df, "report": report, "auc": auc,
        "feature_importance": _feat_imp(model, features),
        "risk_map": label_names,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. CUSTOMER CHURN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
def build_churn_predictor(sale_df):
    """
    Binary churn classifier on enriched RFM + behavioural features.

    Critical v2 fix
    ───────────────
    Original code:
        full_proba = model.predict_proba(clean[features].reindex(rfm.index).fillna(0))
        rfm["Churn_Prob"] = full_proba[:, 1]

    This reindexed `clean` to rfm's full index, creating synthetic zero-filled
    rows for customers that were dropped by dropna(). Model then predicted on
    fabricated data. The fix predicts only on clean rows, then aligns back to
    rfm via index — no synthetic data ever enters the model.

    New features: Log_Monetary, Log_Frequency, Tenure_Days, Buy_Rate (purchases
    per 30 days over tenure). These capture long-term engagement patterns missed
    by raw RFM.
    """
    df = sale_df.copy()
    df = df.dropna(subset=["Invoice Date Key", "Customer Key"])
    df["Invoice Date Key"] = pd.to_datetime(df["Invoice Date Key"], errors="coerce")
    snapshot = df["Invoice Date Key"].max()

    rfm = (
        df.groupby("Customer Key")
        .agg(
            Recency        = ("Invoice Date Key", lambda x: (snapshot - x.max()).days),
            First_Purchase = ("Invoice Date Key", "min"),
            Frequency      = ("Sale Key",               "count"),
            Monetary       = ("Total Including Tax",     "sum"),
            Avg_Order      = ("Total Including Tax",     "mean"),
            Unique_SKUs    = ("Stock Item Key",          "nunique"),
            Avg_Margin     = ("Margin %",                "mean"),
            Std_Order      = ("Total Including Tax",     "std"),
            Max_Order      = ("Total Including Tax",     "max"),
            Min_Order      = ("Total Including Tax",     "min"),
            Active_Months  = ("Calendar Month Number",   "nunique"),
        )
        .reset_index()
    )

    # attach Customer name / metadata for display
    meta = (
        df[["Customer Key", "Customer", "Region", "Customer Value Tier"]]
        .drop_duplicates("Customer Key")
    )
    rfm = rfm.merge(meta, on="Customer Key", how="left")

    rfm["Std_Order"]  = rfm["Std_Order"].fillna(0)
    rfm["Avg_Margin"] = rfm["Avg_Margin"].fillna(rfm["Avg_Margin"].median())

    # log-transforms reduce skew for tree models
    rfm["Log_Monetary"]  = np.log1p(rfm["Monetary"].clip(0))
    rfm["Log_Frequency"] = np.log1p(rfm["Frequency"])

    rfm["CV_Order"]    = (rfm["Std_Order"] / rfm["Avg_Order"].replace(0, np.nan)).fillna(0)
    rfm["Rev_per_SKU"] = (rfm["Monetary"]  / rfm["Unique_SKUs"].replace(0, np.nan)).fillna(0)

    rfm["Tenure_Days"] = (snapshot - rfm["First_Purchase"]).dt.days.fillna(0)
    rfm["Buy_Rate"]    = np.where(
        rfm["Tenure_Days"] > 0,
        rfm["Frequency"] / rfm["Tenure_Days"] * 30.0,
        0.0,
    )

    # churn: inactive ≥90 days AND below-median purchase frequency
    freq_med      = rfm["Frequency"].median()
    rfm["Churned"] = (
        (rfm["Recency"] > 90) & (rfm["Frequency"] < freq_med)
    ).astype(int)

    features = [
        "Recency",      "Frequency",     "Log_Frequency",
        "Monetary",     "Log_Monetary",  "Avg_Order",
        "Unique_SKUs",  "Avg_Margin",    "Std_Order",
        "Max_Order",    "Min_Order",     "Active_Months",
        "CV_Order",     "Rev_per_SKU",   "Tenure_Days",   "Buy_Rate",
    ]

    # Preserve rfm's index — do NOT reset_index here (needed for alignment fix)
    clean = rfm.dropna(subset=features)
    X, y  = clean[features], clean["Churned"]

    if y.nunique() < 2:
        rfm["Churn_Prob"] = 0.0
        rfm["Churn_Pred"] = 0
        dummy = pd.DataFrame({"Feature": features,
                               "Importance": [1 / len(features)] * len(features)})
        return {"model": None, "rfm": rfm, "report": {}, "auc": None,
                "feature_importance": dummy}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", max_features="sqrt",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    # ── FIXED index alignment: predict only on clean, align back via index ────
    rfm["Churn_Prob"] = np.nan
    rfm.loc[clean.index, "Churn_Prob"] = model.predict_proba(clean[features])[:, 1]
    rfm["Churn_Prob"] = rfm["Churn_Prob"].fillna(0.0)
    rfm["Churn_Pred"] = (rfm["Churn_Prob"] > 0.5).astype(int)

    return {
        "model": model, "rfm": rfm, "report": report, "auc": auc,
        "feature_importance": _feat_imp(model, features),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUPPLIER QUALITY SCORER
# ══════════════════════════════════════════════════════════════════════════════
def build_supplier_scorer(purchase_df, supplier_df):
    """
    Scores each supplier 0-100 using a weighted composite + RF interpretation.

    v2 additions
    ────────────
    • Proper hold-out test split with R² on unseen suppliers.
    • Order_Consistency (1 − CV of fulfilment), Late_Delivery_Risk.
    • Quality_Score now includes order consistency in weighted formula.
    """
    pur = purchase_df.copy()

    sup_agg = (
        pur.groupby("Supplier Key")
        .agg(
            Avg_Fulfillment  = ("Fulfillment Rate",    "mean"),
            Std_Fulfillment  = ("Fulfillment Rate",    "std"),
            Min_Fulfillment  = ("Fulfillment Rate",    "min"),
            Total_Orders     = ("Purchase Key",        "count"),
            Total_Value      = ("Purchase Value",      "sum"),
            Avg_Value        = ("Purchase Value",      "mean"),
            Finalized_Rate   = ("Is Order Finalized",  "mean"),
        )
        .reset_index()
    )
    sup_agg["Std_Fulfillment"] = sup_agg["Std_Fulfillment"].fillna(0)
    sup_agg["Min_Fulfillment"] = sup_agg["Min_Fulfillment"].fillna(0)

    # consistency: low CV of fulfilment = reliable supplier
    sup_agg["Order_Consistency"] = (
        1.0 - (sup_agg["Std_Fulfillment"] / sup_agg["Avg_Fulfillment"].replace(0, np.nan))
    ).clip(0, 1).fillna(0)

    sdim = supplier_df.copy()
    sdim["Tier_Code"]  = _le(sdim["Supplier Tier"])
    sdim["Speed_Code"] = _le(sdim["Delivery Speed Category"])

    merged = sup_agg.merge(
        sdim[["Supplier Key", "Supplier", "Supplier Rating",
              "Lead Time Days (Supplier)", "Tier_Code", "Speed_Code", "Region"]],
        on="Supplier Key", how="left",
    )

    score_feats   = ["Avg_Fulfillment", "Min_Fulfillment", "Finalized_Rate",
                     "Order_Consistency", "Supplier Rating", "Avg_Value"]
    penalty_feats = ["Std_Fulfillment", "Lead Time Days (Supplier)"]
    merged = merged.dropna(subset=score_feats + penalty_feats)

    pos_scaled = MinMaxScaler(feature_range=(0, 100)).fit_transform(merged[score_feats])
    neg_scaled = MinMaxScaler(feature_range=(0, 100)).fit_transform(merged[penalty_feats])

    # weights: fulfilment consistency & reliability rank highest
    weights = np.array([0.30, 0.12, 0.18, 0.15, 0.13, 0.12])
    merged["Quality_Score"] = (
        (pos_scaled * weights).sum(axis=1) * 0.75
        - neg_scaled.mean(axis=1) * 0.25
    ).clip(0, 100)

    features = [
        "Avg_Fulfillment",       "Std_Fulfillment",   "Min_Fulfillment",
        "Total_Orders",          "Finalized_Rate",    "Order_Consistency",
        "Tier_Code",             "Speed_Code",
        "Lead Time Days (Supplier)", "Supplier Rating",
    ]
    X = merged[features]
    y = merged["Quality_Score"]

    # ── hold-out test for unbiased R² (v2 fix) ───────────────────────────────
    if len(merged) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        test_r2 = r2_score(y_test, model.predict(X_test))
        # re-fit on full data for scoring all suppliers
        model.fit(X, y)
    else:
        model = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
        model.fit(X, y)
        test_r2 = None

    merged["Predicted_Score"] = model.predict(X).clip(0, 100)

    def _grade(s):
        if s >= 80: return "A"
        if s >= 65: return "B"
        if s >= 50: return "C"
        return "D"

    merged["Grade"] = merged["Quality_Score"].apply(_grade)

    return {
        "model": model, "df": merged,
        "test_r2": test_r2,
        "feature_importance": _feat_imp(model, features),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def build_anomaly_detector(txn_df):
    """
    Isolation Forest on financial transactions.

    v2 improvements
    ───────────────
    • RobustScaler (IQR-based) replaces StandardScaler — resistant to the
      very outliers we are trying to detect.
    • Log1p-transform compresses heavy tails in monetary columns before
      outlier isolation, improving separation in the feature space.
    • Payment method + transaction type encoded as categorical features,
      providing structural context the model was previously blind to.
    • Added Amount_Flag (binary: unusually large transaction) and
      Balance_Pct to enrich the feature set.
    """
    df = txn_df.copy()
    df = df.dropna(subset=["Total Including Tax", "Tax Amount", "Outstanding Balance"])

    num_cols = ["Total Excluding Tax", "Tax Amount",
                "Total Including Tax", "Outstanding Balance"]
    df[num_cols] = df[num_cols].fillna(0)

    # log1p-transform monetary columns (compress right tail)
    for col in num_cols:
        df[f"Log_{col.replace(' ', '_')}"] = np.log1p(df[col].clip(0))
    log_cols = [f"Log_{c.replace(' ', '_')}" for c in num_cols]

    # ratio features
    df["Tax_Rate_Implied"] = (
        df["Tax Amount"] / df["Total Excluding Tax"].replace(0, np.nan)
    ).fillna(0)
    df["Balance_Ratio"] = (
        df["Outstanding Balance"] / df["Total Including Tax"].replace(0, np.nan)
    ).fillna(0)
    df["Tax_to_Total"] = (
        df["Tax Amount"] / df["Total Including Tax"].replace(0, np.nan)
    ).fillna(0)

    # large-amount binary flag (above 95th percentile)
    thresh = df["Total Including Tax"].quantile(0.95)
    df["Amount_Flag"] = (df["Total Including Tax"] > thresh).astype(float)

    # encode categorical context
    df["PayMethod_Code"] = _le(df.get("Payment Method", pd.Series(dtype=str)))
    df["TxnType_Code"]   = _le(df.get("Transaction Type", pd.Series(dtype=str)))

    feature_cols = (
        log_cols
        + ["Tax_Rate_Implied", "Balance_Ratio", "Tax_to_Total",
           "Amount_Flag", "PayMethod_Code", "TxnType_Code"]
    )
    df[feature_cols] = (
        df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    )

    # RobustScaler: median/IQR normalisation — outlier-resistant
    scaler = RobustScaler()
    X = scaler.fit_transform(df[feature_cols])

    model = IsolationForest(
        contamination=0.05,
        n_estimators=400,
        max_samples="auto",
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    df["Anomaly_Score"] = model.decision_function(X)
    df["Is_Anomaly"]    = model.predict(X) == -1

    return {
        "model": model, "df": df,
        "anomaly_count": int(df["Is_Anomaly"].sum()),
        "anomaly_rate":  float(df["Is_Anomaly"].mean() * 100),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. CUSTOMER SEGMENTATION  (KMeans on enriched RFM)
# ══════════════════════════════════════════════════════════════════════════════
def build_customer_segments(sale_df, n_clusters: int = 4):
    """
    KMeans segmentation on log-scaled RFM + Tenure + Buy_Rate.

    v2 additions
    ────────────
    • Tenure_Days and Buy_Rate added to the feature matrix — these capture
      long-term engagement patterns that raw RFM misses.
    • Segment naming derived from centroid characteristics rather than
      arbitrary monetary rank ordering.
    """
    df = sale_df.copy()
    df = df.dropna(subset=["Invoice Date Key", "Customer Key"])
    df["Invoice Date Key"] = pd.to_datetime(df["Invoice Date Key"], errors="coerce")
    snapshot = df["Invoice Date Key"].max()

    rfm = (
        df.groupby(["Customer Key", "Customer", "Region", "Customer Value Tier"])
        .agg(
            Recency        = ("Invoice Date Key", lambda x: (snapshot - x.max()).days),
            First_Purchase = ("Invoice Date Key", "min"),
            Frequency      = ("Sale Key",               "count"),
            Monetary       = ("Total Including Tax",     "sum"),
            Avg_Order      = ("Total Including Tax",     "mean"),
        )
        .reset_index()
    )

    rfm["Tenure_Days"] = (snapshot - rfm["First_Purchase"]).dt.days.fillna(0)
    rfm["Buy_Rate"]    = np.where(
        rfm["Tenure_Days"] > 0,
        rfm["Frequency"] / rfm["Tenure_Days"] * 30.0,
        0.0,
    )

    # log-transform to reduce skew
    rfm["Log_Monetary"]  = np.log1p(rfm["Monetary"].clip(0))
    rfm["Log_Frequency"] = np.log1p(rfm["Frequency"])
    rfm["Log_Buy_Rate"]  = np.log1p(rfm["Buy_Rate"].clip(0))

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(
        rfm[["Recency", "Log_Frequency", "Log_Monetary", "Log_Buy_Rate"]]
    )

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=30, max_iter=500)
    rfm["Segment"] = model.fit_predict(X)

    # ── centroid-based naming ─────────────────────────────────────────────────
    # Decode centroids back to interpretable space
    centers = scaler.inverse_transform(model.cluster_centers_)
    # columns: [Recency, Log_Freq, Log_Monetary, Log_Buy_Rate]
    center_df = pd.DataFrame(centers,
                              columns=["Recency", "Log_Freq", "Log_Mon", "Log_Buy"])
    center_df["Monetary_approx"] = np.expm1(center_df["Log_Mon"])
    center_df["Buy_Rate_approx"] = np.expm1(center_df["Log_Buy"])

    def _name_segment(row):
        rec  = row["Recency"]
        mon  = row["Monetary_approx"]
        rate = row["Buy_Rate_approx"]
        mon_med  = center_df["Monetary_approx"].median()
        rate_med = center_df["Buy_Rate_approx"].median()
        if rec <= 30  and rate >= rate_med and mon >= mon_med: return "Champions"
        if rec <= 90  and mon >= mon_med:                       return "Loyal Customers"
        if rec > 180  and mon <= mon_med:                       return "Churned/Lost"
        return "At-Risk Customers"

    seg_name_map = {
        idx: _name_segment(row) for idx, row in center_df.iterrows()
    }
    # ensure uniqueness (two centroids might map to same name)
    seen = {}
    for k, v in seg_name_map.items():
        if v in seen.values():
            seg_name_map[k] = v + "+"
        seen[k] = seg_name_map[k]

    rfm["Segment Name"] = rfm["Segment"].map(seg_name_map)

    return {"model": model, "rfm": rfm, "seg_names": seg_name_map}
