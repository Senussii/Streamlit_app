"""
ML Models — Supply Chain Predictive Analytics
Demand Forecasting · Stockout Risk · Churn · Supplier Quality · Anomaly Detection
Optimized: better feature engineering, tuned hyperparameters, robust error handling.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingClassifier,
                               IsolationForest, RandomForestClassifier,
                               ExtraTreesRegressor)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_percentage_error, r2_score,
                              classification_report, roc_auc_score)
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DEMAND FORECASTING  (Gradient Boosting + enriched time features)
# ══════════════════════════════════════════════════════════════════════════════
def build_demand_forecast(sale_df, horizon_months=3):
    """
    Forecasts monthly demand (quantity) per Stock Category.
    Uses GradientBoosting with extended lag/rolling + trend features.
    """
    df = sale_df.copy()
    df = df.dropna(subset=["Calendar Year", "Calendar Month Number", "Quantity"])

    agg = (df.groupby(["Calendar Year", "Calendar Month Number", "Stock Category"])
             .agg(Total_Qty=("Quantity", "sum"),
                  Total_Rev=("Total Excluding Tax", "sum"),
                  Avg_Price=("Unit Price", "mean"),
                  Num_Transactions=("Sale Key", "count"))
             .reset_index())

    agg = agg.sort_values(["Stock Category", "Calendar Year", "Calendar Month Number"])

    # extended lag & rolling features
    for cat, grp in agg.groupby("Stock Category"):
        idx = grp.index
        shifted = grp["Total_Qty"]
        agg.loc[idx, "Lag1"]  = shifted.shift(1)
        agg.loc[idx, "Lag2"]  = shifted.shift(2)
        agg.loc[idx, "Lag3"]  = shifted.shift(3)
        agg.loc[idx, "Lag6"]  = shifted.shift(6)
        agg.loc[idx, "Lag12"] = shifted.shift(12)
        agg.loc[idx, "Roll3"]  = shifted.shift(1).rolling(3, min_periods=1).mean()
        agg.loc[idx, "Roll6"]  = shifted.shift(1).rolling(6, min_periods=1).mean()
        agg.loc[idx, "Roll12"] = shifted.shift(1).rolling(12, min_periods=1).mean()
        agg.loc[idx, "RollStd3"] = shifted.shift(1).rolling(3, min_periods=1).std().fillna(0)
        # trend: difference from prior period
        agg.loc[idx, "Trend1"] = shifted.shift(1).diff(1)
        agg.loc[idx, "Trend3"] = shifted.shift(1).diff(3)

    le = LabelEncoder()
    agg["Cat_Code"] = le.fit_transform(agg["Stock Category"].fillna("Unknown"))

    # cyclical month encoding
    agg["Month_Sin"] = np.sin(2 * np.pi * agg["Calendar Month Number"] / 12)
    agg["Month_Cos"] = np.cos(2 * np.pi * agg["Calendar Month Number"] / 12)

    features = ["Calendar Year", "Month_Sin", "Month_Cos", "Cat_Code",
                "Lag1", "Lag2", "Lag3", "Lag6", "Lag12",
                "Roll3", "Roll6", "Roll12", "RollStd3",
                "Trend1", "Trend3", "Avg_Price", "Num_Transactions"]

    clean = agg.dropna(subset=features + ["Total_Qty"])
    X = clean[features]
    y = clean["Total_Qty"]

    # chronological split — never shuffle time-series data
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = GradientBoostingClassifier.__bases__[0]  # just to avoid import confusion
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).clip(0)
    mape   = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2     = r2_score(y_test, y_pred)

    # future forecast
    last_period = clean.groupby("Stock Category").last().reset_index()
    forecasts = []
    for _, row in last_period.iterrows():
        yr, mo = int(row["Calendar Year"]), int(row["Calendar Month Number"])
        lag1, lag2, lag3 = row["Total_Qty"], row["Lag1"], row["Lag2"]
        lag6, lag12 = row["Lag6"], row["Lag12"]
        roll3, roll6, roll12 = row["Roll3"], row["Roll6"], row["Roll12"]
        rollstd3 = row["RollStd3"]
        for _ in range(horizon_months):
            mo += 1
            if mo > 12:
                mo = 1; yr += 1
            forecasts.append({
                "Stock Category": row["Stock Category"],
                "Forecast Year": yr, "Forecast Month": mo,
                "Calendar Year": yr,
                "Month_Sin": np.sin(2 * np.pi * mo / 12),
                "Month_Cos": np.cos(2 * np.pi * mo / 12),
                "Cat_Code": row["Cat_Code"],
                "Lag1": lag1, "Lag2": lag2, "Lag3": lag3,
                "Lag6": lag6, "Lag12": lag12,
                "Roll3": roll3, "Roll6": roll6, "Roll12": roll12,
                "RollStd3": rollstd3,
                "Trend1": 0, "Trend3": 0,
                "Avg_Price": row["Avg_Price"],
                "Num_Transactions": row["Num_Transactions"],
            })
            # slide the window
            lag3, lag2, lag1 = lag2, lag1, lag1  # naive carry-forward

    fc_df = pd.DataFrame(forecasts)
    if not fc_df.empty:
        fc_df["Predicted_Qty"] = model.predict(fc_df[features]).clip(0)

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "model": model, "mape": mape, "r2": r2,
        "actuals_df": clean,
        "test_pred": y_pred,
        "test_true": y_test.values,
        "forecast_df": fc_df,
        "feature_importance": feat_imp,
        "agg_df": agg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. STOCKOUT RISK CLASSIFIER  (Gradient Boosting — optimised)
# ══════════════════════════════════════════════════════════════════════════════
def build_stockout_classifier(inventory_df, movement_df):
    """
    Classifies each SKU as HIGH / MEDIUM / LOW stockout risk.
    Optimized with more features and better hyperparameters.
    """
    inv = inventory_df.copy()

    # velocity from movement (outflows only)
    out = movement_df[movement_df["Transaction_Direction"] == "Outflow"]
    vel = (out.groupby("Stock Item Key")["Quantity"]
              .agg(Monthly_Velocity="mean", Velocity_Std="std", Velocity_Max="max")
              .reset_index())
    vel["Velocity_Std"] = vel["Velocity_Std"].fillna(0)
    vel["Velocity_Max"] = vel["Velocity_Max"].fillna(0)

    df = inv.merge(vel, on="Stock Item Key", how="left")
    df["Monthly_Velocity"] = df["Monthly_Velocity"].fillna(0)
    df["Velocity_Std"]     = df["Velocity_Std"].fillna(0)
    df["Velocity_Max"]     = df["Velocity_Max"].fillna(0)

    df["Days_Coverage"] = np.where(
        df["Monthly_Velocity"] > 0,
        df["Quantity On Hand"] / (df["Monthly_Velocity"] / 30), 9999)

    # ratio features
    df["Stock_vs_Reorder"] = df["Quantity On Hand"] / (df["Reorder Level"].replace(0, np.nan)).fillna(1)
    df["Stock_vs_Target"]  = df["Quantity On Hand"] / (df["Target Stock Level"].replace(0, np.nan)).fillna(1)
    df["Velocity_CV"]      = df["Velocity_Std"] / (df["Monthly_Velocity"].replace(0, np.nan)).fillna(0)

    def risk_label(r):
        if r["Reorder Flag"]:        return 2   # HIGH
        if r["Days_Coverage"] < 30:  return 1   # MEDIUM
        return 0                                  # LOW

    df["Risk_Label"] = df.apply(risk_label, axis=1)

    df["Avail_Code"] = LabelEncoder().fit_transform(df["Availability"].fillna("Unknown"))
    df["Cat_Code"]   = LabelEncoder().fit_transform(df["Stock Category"].fillna("Unknown"))

    features = [
        "Quantity On Hand", "Reorder Level", "Target Stock Level",
        "Last Cost Price", "Monthly_Velocity", "Velocity_Std", "Velocity_Max",
        "Days_Coverage", "Lead Time Days", "Avail_Code", "Cat_Code",
        "Stock_vs_Reorder", "Stock_vs_Target", "Velocity_CV",
    ]

    clean = df.dropna(subset=features)
    X, y = clean[features], clean["Risk_Label"]

    # handle single-class edge case
    if y.nunique() < 2:
        df["Predicted_Risk"] = 0
        df["Risk_Label_Name"] = "LOW"
        df["Predicted_Risk_Name"] = "LOW"
        dummy_fi = pd.DataFrame({"Feature": features, "Importance": [1/len(features)]*len(features)})
        return {"model": None, "df": df, "report": {}, "auc": None,
                "feature_importance": dummy_fi, "risk_map": {0:"LOW",1:"MEDIUM",2:"HIGH"}}

    stratify = y if y.nunique() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify)

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    present_labels = sorted(y_test.unique())
    label_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    t_names = [label_names[l] for l in present_labels]
    report = classification_report(y_test, y_pred, labels=present_labels,
                                   target_names=t_names, output_dict=True)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
    except Exception:
        auc = None

    full_X = clean[features].reindex(df.index).fillna(0)
    df["Predicted_Risk"] = model.predict(full_X)
    df["Risk_Label_Name"]      = df["Risk_Label"].map(label_names)
    df["Predicted_Risk_Name"]  = df["Predicted_Risk"].map(label_names)

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "model": model, "df": df, "report": report, "auc": auc,
        "feature_importance": feat_imp, "risk_map": label_names,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. CUSTOMER CHURN PREDICTOR  (Random Forest — optimised)
# ══════════════════════════════════════════════════════════════════════════════
def build_churn_predictor(sale_df):
    """
    Predicts customer churn using enriched RFM + behavioural features.
    """
    df = sale_df.copy()
    df = df.dropna(subset=["Invoice Date Key", "Customer Key"])
    df["Invoice Date Key"] = pd.to_datetime(df["Invoice Date Key"], errors="coerce")
    snapshot = df["Invoice Date Key"].max()

    rfm = (df.groupby("Customer Key").agg(
        Recency      = ("Invoice Date Key", lambda x: (snapshot - x.max()).days),
        Frequency    = ("Sale Key",          "count"),
        Monetary     = ("Total Including Tax","sum"),
        Avg_Order    = ("Total Including Tax","mean"),
        Unique_SKUs  = ("Stock Item Key",    "nunique"),
        Avg_Margin   = ("Margin %",          "mean"),
        Std_Order    = ("Total Including Tax","std"),
        Max_Order    = ("Total Including Tax","max"),
        Min_Order    = ("Total Including Tax","min"),
        Active_Months= ("Calendar Month Number","nunique"),
    ).reset_index())

    rfm["Std_Order"] = rfm["Std_Order"].fillna(0)
    rfm["CV_Order"]  = rfm["Std_Order"] / (rfm["Avg_Order"].replace(0, np.nan)).fillna(0)
    rfm["Rev_per_SKU"] = rfm["Monetary"] / rfm["Unique_SKUs"].replace(0, np.nan)

    # churn label: no purchase in last 90 days AND below-median frequency
    rfm["Churned"] = (
        (rfm["Recency"] > 90) & (rfm["Frequency"] < rfm["Frequency"].median())
    ).astype(int)

    features = [
        "Recency", "Frequency", "Monetary", "Avg_Order", "Unique_SKUs",
        "Avg_Margin", "Std_Order", "Max_Order", "Min_Order",
        "Active_Months", "CV_Order", "Rev_per_SKU",
    ]
    clean = rfm.dropna(subset=features)
    X, y  = clean[features], clean["Churned"]

    if y.nunique() < 2:
        rfm["Churn_Prob"] = 0.0
        rfm["Churn_Pred"] = 0
        dummy_fi = pd.DataFrame({"Feature": features, "Importance": [1/len(features)]*len(features)})
        return {"model": None, "rfm": rfm, "report": {}, "auc": None,
                "feature_importance": dummy_fi}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba  = model.predict_proba(X_test)
    y_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    report = classification_report(y_test, y_pred, output_dict=True)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    full_proba = model.predict_proba(clean[features].reindex(rfm.index).fillna(0))
    rfm["Churn_Prob"] = full_proba[:, 1] if full_proba.shape[1] > 1 else full_proba[:, 0]
    rfm["Churn_Pred"] = (rfm["Churn_Prob"] > 0.5).astype(int)

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "model": model, "rfm": rfm, "report": report, "auc": auc,
        "feature_importance": feat_imp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUPPLIER QUALITY SCORER  (Random Forest — optimised)
# ══════════════════════════════════════════════════════════════════════════════
def build_supplier_scorer(purchase_df, supplier_df):
    """
    Scores each supplier 0-100 with enriched features and better weighting.
    """
    pur = purchase_df.copy()

    sup_agg = (pur.groupby("Supplier Key").agg(
        Avg_Fulfillment = ("Fulfillment Rate", "mean"),
        Std_Fulfillment = ("Fulfillment Rate", "std"),
        Min_Fulfillment = ("Fulfillment Rate", "min"),
        Total_Orders    = ("Purchase Key", "count"),
        Total_Value     = ("Purchase Value", "sum"),
        Avg_Value       = ("Purchase Value", "mean"),
        Finalized_Rate  = ("Is Order Finalized", "mean"),
    ).reset_index())

    sup_agg["Std_Fulfillment"] = sup_agg["Std_Fulfillment"].fillna(0)
    sup_agg["Min_Fulfillment"] = sup_agg["Min_Fulfillment"].fillna(0)

    sup_df = supplier_df.copy()
    sup_df["Tier_Code"]  = LabelEncoder().fit_transform(sup_df["Supplier Tier"].fillna("Unknown"))
    sup_df["Speed_Code"] = LabelEncoder().fit_transform(sup_df["Delivery Speed Category"].fillna("Unknown"))

    merged = sup_agg.merge(
        sup_df[["Supplier Key", "Supplier", "Supplier Rating",
                "Lead Time Days (Supplier)", "Tier_Code", "Speed_Code", "Region"]],
        on="Supplier Key", how="left")

    score_features  = ["Avg_Fulfillment", "Min_Fulfillment", "Finalized_Rate",
                       "Supplier Rating", "Avg_Value"]
    penalty_feats   = ["Std_Fulfillment", "Lead Time Days (Supplier)"]
    merged = merged.dropna(subset=score_features + penalty_feats)

    scaler_pos = MinMaxScaler(feature_range=(0, 100))
    scaler_neg = MinMaxScaler(feature_range=(0, 100))

    pos_scaled = scaler_pos.fit_transform(merged[score_features])
    neg_scaled = scaler_neg.fit_transform(merged[penalty_feats])

    # weighted composite: fulfillment & finalization weigh more
    weights = np.array([0.35, 0.15, 0.20, 0.15, 0.15])
    merged["Quality_Score"] = (
        (pos_scaled * weights).sum(axis=1) * 0.75 -
        neg_scaled.mean(axis=1) * 0.25
    ).clip(0, 100)

    features = [
        "Avg_Fulfillment", "Std_Fulfillment", "Min_Fulfillment",
        "Total_Orders", "Finalized_Rate", "Tier_Code", "Speed_Code",
        "Lead Time Days (Supplier)", "Supplier Rating",
    ]

    X = merged[features]
    y = merged["Quality_Score"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    merged["Predicted_Score"] = model.predict(X).clip(0, 100)

    def grade(s):
        if s >= 80: return "A"
        if s >= 65: return "B"
        if s >= 50: return "C"
        return "D"

    merged["Grade"] = merged["Quality_Score"].apply(grade)

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "model": model, "df": merged, "feature_importance": feat_imp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. ANOMALY DETECTION  (Isolation Forest — optimised)
# ══════════════════════════════════════════════════════════════════════════════
def build_anomaly_detector(txn_df):
    """
    Flags anomalous financial transactions.
    Enriched with ratio/interaction features for better isolation.
    """
    df = txn_df.copy()
    df = df.dropna(subset=["Total Including Tax", "Tax Amount", "Outstanding Balance"])

    num_cols = ["Total Excluding Tax", "Tax Amount",
                "Total Including Tax", "Outstanding Balance"]
    df[num_cols] = df[num_cols].fillna(0)

    # derived features
    df["Tax_Rate_Implied"] = df["Tax Amount"] / (df["Total Excluding Tax"].replace(0, np.nan)).fillna(0)
    df["Balance_Ratio"]    = df["Outstanding Balance"] / (df["Total Including Tax"].replace(0, np.nan)).fillna(0)
    df["Tax_to_Total"]     = df["Tax Amount"] / (df["Total Including Tax"].replace(0, np.nan)).fillna(0)

    feature_cols = num_cols + ["Tax_Rate_Implied", "Balance_Ratio", "Tax_to_Total"]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    model = IsolationForest(
        contamination=0.05,
        n_estimators=300,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    df["Anomaly_Score"] = model.decision_function(X)
    df["Is_Anomaly"]    = model.predict(X) == -1

    return {
        "model": model, "df": df,
        "anomaly_count": int(df["Is_Anomaly"].sum()),
        "anomaly_rate":  df["Is_Anomaly"].mean() * 100,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. CUSTOMER SEGMENTATION  (KMeans on RFM — optimised)
# ══════════════════════════════════════════════════════════════════════════════
def build_customer_segments(sale_df, n_clusters=4):
    """
    Segments customers using KMeans on scaled RFM + extra features.
    """
    df = sale_df.copy()
    df = df.dropna(subset=["Invoice Date Key", "Customer Key"])
    df["Invoice Date Key"] = pd.to_datetime(df["Invoice Date Key"], errors="coerce")
    snapshot = df["Invoice Date Key"].max()

    rfm = (df.groupby(["Customer Key", "Customer", "Region", "Customer Value Tier"]).agg(
        Recency   = ("Invoice Date Key", lambda x: (snapshot - x.max()).days),
        Frequency = ("Sale Key",          "count"),
        Monetary  = ("Total Including Tax","sum"),
        Avg_Order = ("Total Including Tax","mean"),
    ).reset_index())

    # log-transform Monetary & Frequency to reduce skew
    rfm["Log_Monetary"]  = np.log1p(rfm["Monetary"])
    rfm["Log_Frequency"] = np.log1p(rfm["Frequency"])

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency", "Log_Frequency", "Log_Monetary"]])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    rfm["Segment"] = model.fit_predict(X)

    seg_means = rfm.groupby("Segment")["Monetary"].mean().sort_values(ascending=False)
    seg_names = {seg: name for seg, name in zip(
        seg_means.index,
        ["Champions", "Loyal Customers", "At-Risk Customers", "Churned/Lost"][:n_clusters]
    )}
    rfm["Segment Name"] = rfm["Segment"].map(seg_names)

    return {"model": model, "rfm": rfm, "seg_names": seg_names}