"""
ML Models — Supply Chain Predictive Analytics
Demand Forecasting · Stockout Risk · Churn · Supplier Quality · Anomaly Detection
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingClassifier,
                               IsolationForest, RandomForestClassifier)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_percentage_error, r2_score,
                              classification_report, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DEMAND FORECASTING  (Random Forest Regressor + time features)
# ══════════════════════════════════════════════════════════════════════════════
def build_demand_forecast(sale_df, horizon_months=3):
    """
    Forecasts monthly demand (quantity) per Stock Category using
    time-series features fed into a Random Forest.

    Returns
    -------
    model_info : dict with predictions, metrics, feature importance
    """
    df = sale_df.copy()
    df = df.dropna(subset=["Calendar Year","Calendar Month Number","Quantity"])
    df["Period"] = df["Calendar Year"].astype(int) * 100 + df["Calendar Month Number"].astype(int)

    agg = (df.groupby(["Calendar Year","Calendar Month Number","Stock Category"])
             .agg(Total_Qty=("Quantity","sum"),
                  Total_Rev=("Total Excluding Tax","sum"),
                  Avg_Price=("Unit Price","mean"))
             .reset_index())

    agg = agg.sort_values(["Stock Category","Calendar Year","Calendar Month Number"])

    # lag & rolling features
    for cat, grp in agg.groupby("Stock Category"):
        idx = grp.index
        agg.loc[idx,"Lag1"]    = grp["Total_Qty"].shift(1)
        agg.loc[idx,"Lag2"]    = grp["Total_Qty"].shift(2)
        agg.loc[idx,"Lag3"]    = grp["Total_Qty"].shift(3)
        agg.loc[idx,"Roll3"]   = grp["Total_Qty"].shift(1).rolling(3).mean()
        agg.loc[idx,"Roll6"]   = grp["Total_Qty"].shift(1).rolling(6).mean()

    agg["Cat_Code"] = LabelEncoder().fit_transform(agg["Stock Category"].fillna("Unknown"))
    features = ["Calendar Year","Calendar Month Number","Cat_Code",
                "Lag1","Lag2","Lag3","Roll3","Roll6","Avg_Price"]

    clean = agg.dropna(subset=features + ["Total_Qty"])
    X = clean[features]
    y = clean["Total_Qty"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred    = model.predict(X_test)
    mape      = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2        = r2_score(y_test, y_pred)

    # future forecast: last known period × horizon
    last_period = clean.groupby("Stock Category").last().reset_index()
    forecasts = []
    for _, row in last_period.iterrows():
        yr, mo = int(row["Calendar Year"]), int(row["Calendar Month Number"])
        for h in range(1, horizon_months + 1):
            mo += 1
            if mo > 12: mo = 1; yr += 1
            forecasts.append({
                "Stock Category": row["Stock Category"],
                "Forecast Year": yr, "Forecast Month": mo,
                "Lag1": row["Total_Qty"], "Lag2": row["Lag1"], "Lag3": row["Lag2"],
                "Roll3": row["Roll3"], "Roll6": row["Roll6"],
                "Avg_Price": row["Avg_Price"],
                "Cat_Code": row["Cat_Code"],
                "Calendar Year": yr, "Calendar Month Number": mo,
            })

    fc_df = pd.DataFrame(forecasts)
    fc_df["Predicted_Qty"] = model.predict(fc_df[features]).clip(0)

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "model": model, "mape": mape, "r2": r2,
        "actuals_df": clean, "test_pred": y_pred, "test_true": y_test.values,
        "forecast_df": fc_df, "feature_importance": feat_imp,
        "agg_df": agg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. STOCKOUT RISK CLASSIFIER  (Gradient Boosting)
# ══════════════════════════════════════════════════════════════════════════════
def build_stockout_classifier(inventory_df, movement_df):
    """
    Classifies each SKU as HIGH / MEDIUM / LOW stockout risk using
    inventory levels, movement velocity, and product attributes.
    """
    inv = inventory_df.copy()

    # velocity from movement
    vel = (movement_df[movement_df["Transaction_Direction"] == "Outflow"]
           .groupby("Stock Item Key")["Quantity"]
           .agg(Monthly_Velocity="mean", Velocity_Std="std")
           .reset_index())
    vel["Velocity_Std"] = vel["Velocity_Std"].fillna(0)

    df = inv.merge(vel, on="Stock Item Key", how="left")
    df["Monthly_Velocity"] = df["Monthly_Velocity"].fillna(0)
    df["Velocity_Std"]     = df["Velocity_Std"].fillna(0)

    # days of coverage
    df["Days_Coverage"] = np.where(
        df["Monthly_Velocity"] > 0,
        df["Quantity On Hand"] / (df["Monthly_Velocity"] / 30), 9999)

    # label: HIGH=0-7d, MEDIUM=7-30d, LOW=>30d
    def risk_label(r):
        if r["Reorder Flag"]: return 2   # HIGH
        if r["Days_Coverage"] < 30:      return 1   # MEDIUM
        return 0                                     # LOW
    df["Risk_Label"] = df.apply(risk_label, axis=1)

    # encode availability
    df["Avail_Code"] = LabelEncoder().fit_transform(df["Availability"].fillna("Unknown"))
    df["Cat_Code"]   = LabelEncoder().fit_transform(df["Stock Category"].fillna("Unknown"))

    features = ["Quantity On Hand","Reorder Level","Target Stock Level",
                "Last Cost Price","Monthly_Velocity","Velocity_Std",
                "Days_Coverage","Lead Time Days","Avail_Code","Cat_Code"]

    clean = df.dropna(subset=features)
    X, y = clean[features], clean["Risk_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                        learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    present_labels = sorted(y_test.unique())
    label_names = {0:"LOW", 1:"MEDIUM", 2:"HIGH"}
    t_names = [label_names[l] for l in present_labels]
    report = classification_report(y_test, y_pred,
                                    labels=present_labels,
                                    target_names=t_names,
                                    output_dict=True)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
    except Exception:
        auc = None

    df["Predicted_Risk"] = model.predict(clean[features].reindex(df.index, fill_value=0)
                                          .fillna(0))
    risk_map = {0:"LOW", 1:"MEDIUM", 2:"HIGH"}
    df["Risk_Label_Name"] = df["Risk_Label"].map(risk_map)
    df["Predicted_Risk_Name"] = df["Predicted_Risk"].map(risk_map)

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "model": model, "df": df, "report": report, "auc": auc,
        "feature_importance": feat_imp, "risk_map": risk_map,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. CUSTOMER CHURN PREDICTOR  (Random Forest Classifier)
# ══════════════════════════════════════════════════════════════════════════════
def build_churn_predictor(sale_df):
    """
    Predicts which customers are at risk of churning based on
    RFM (Recency-Frequency-Monetary) + order behaviour features.
    """
    df = sale_df.copy()
    df = df.dropna(subset=["Invoice Date Key","Customer Key"])
    df["Invoice Date Key"] = pd.to_datetime(df["Invoice Date Key"], errors="coerce")
    snapshot = df["Invoice Date Key"].max()

    rfm = (df.groupby("Customer Key").agg(
        Recency     = ("Invoice Date Key", lambda x: (snapshot - x.max()).days),
        Frequency   = ("Sale Key",         "count"),
        Monetary    = ("Total Including Tax","sum"),
        Avg_Order   = ("Total Including Tax","mean"),
        Unique_SKUs = ("Stock Item Key",   "nunique"),
        Avg_Margin  = ("Margin %",         "mean"),
    ).reset_index())

    # churn = no purchase in last 90 days AND below median frequency
    rfm["Churned"] = (
        (rfm["Recency"] > 90) & (rfm["Frequency"] < rfm["Frequency"].median())
    ).astype(int)

    features = ["Recency","Frequency","Monetary","Avg_Order",
                "Unique_SKUs","Avg_Margin"]
    clean = rfm.dropna(subset=features)
    X, y = clean[features], clean["Churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                    class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    proba   = model.predict_proba(X_test)
    y_prob  = proba[:,1] if proba.shape[1] > 1 else proba[:,0]
    report  = classification_report(y_test, y_pred, output_dict=True)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    full_proba = model.predict_proba(clean[features].reindex(rfm.index).fillna(0))
    rfm["Churn_Prob"] = full_proba[:,1] if full_proba.shape[1] > 1 else full_proba[:,0]
    rfm["Churn_Pred"]   = (rfm["Churn_Prob"] > 0.5).astype(int)

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "model": model, "rfm": rfm, "report": report, "auc": auc,
        "feature_importance": feat_imp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUPPLIER QUALITY SCORER  (Random Forest Regressor → composite score)
# ══════════════════════════════════════════════════════════════════════════════
def build_supplier_scorer(purchase_df, supplier_df):
    """
    Scores each supplier 0-100 based on fulfillment rate, lead-time
    consistency, and purchase value delivered.
    """
    pur = purchase_df.copy()

    sup_agg = (pur.groupby("Supplier Key").agg(
        Avg_Fulfillment  = ("Fulfillment Rate","mean"),
        Std_Fulfillment  = ("Fulfillment Rate","std"),
        Total_Orders     = ("Purchase Key","count"),
        Total_Value      = ("Purchase Value","sum"),
        Avg_Value        = ("Purchase Value","mean"),
        Finalized_Rate   = ("Is Order Finalized","mean"),
    ).reset_index())

    sup_agg["Std_Fulfillment"] = sup_agg["Std_Fulfillment"].fillna(0)

    sup_df = supplier_df.copy()
    sup_df["Tier_Code"]  = LabelEncoder().fit_transform(sup_df["Supplier Tier"].fillna("Unknown"))
    sup_df["Speed_Code"] = LabelEncoder().fit_transform(sup_df["Delivery Speed Category"].fillna("Unknown"))

    merged = sup_agg.merge(
        sup_df[["Supplier Key","Supplier","Supplier Rating","Lead Time Days (Supplier)",
                "Tier_Code","Speed_Code","Region"]],
        on="Supplier Key", how="left")

    # composite quality score (normalised 0-100)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,100))
    score_features = ["Avg_Fulfillment","Finalized_Rate","Supplier Rating","Avg_Value"]
    penalty_feats  = ["Std_Fulfillment","Lead Time Days (Supplier)"]
    merged = merged.dropna(subset=score_features + penalty_feats)

    pos_scaled = scaler.fit_transform(merged[score_features])
    neg_scaled = scaler.fit_transform(merged[penalty_feats])
    merged["Quality_Score"] = (pos_scaled.mean(axis=1) * 0.7 -
                                neg_scaled.mean(axis=1) * 0.3).clip(0,100)

    features = ["Avg_Fulfillment","Std_Fulfillment","Total_Orders",
                "Finalized_Rate","Tier_Code","Speed_Code",
                "Lead Time Days (Supplier)","Supplier Rating"]

    X = merged[features]
    y = merged["Quality_Score"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    merged["Predicted_Score"] = model.predict(X).clip(0,100)

    def grade(s):
        if s >= 80: return "A"
        if s >= 65: return "B"
        if s >= 50: return "C"
        return "D"
    merged["Grade"] = merged["Quality_Score"].apply(grade)

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "model": model, "df": merged, "feature_importance": feat_imp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. ANOMALY DETECTION  (Isolation Forest on transactions)
# ══════════════════════════════════════════════════════════════════════════════
def build_anomaly_detector(txn_df):
    """
    Flags anomalous financial transactions using Isolation Forest.
    """
    df = txn_df.copy()
    df = df.dropna(subset=["Total Including Tax","Tax Amount","Outstanding Balance"])

    num_cols = ["Total Excluding Tax","Tax Amount","Total Including Tax","Outstanding Balance"]
    df[num_cols] = df[num_cols].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[num_cols])

    model = IsolationForest(contamination=0.05, n_estimators=200,
                             random_state=42, n_jobs=-1)
    model.fit(X)
    df["Anomaly_Score"] = model.decision_function(X)
    df["Is_Anomaly"]    = model.predict(X) == -1

    return {
        "model": model, "df": df,
        "anomaly_count": int(df["Is_Anomaly"].sum()),
        "anomaly_rate": df["Is_Anomaly"].mean() * 100,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. CUSTOMER SEGMENTATION  (KMeans on RFM)
# ══════════════════════════════════════════════════════════════════════════════
def build_customer_segments(sale_df, n_clusters=4):
    """
    Segments customers using KMeans on RFM features.
    """
    df = sale_df.copy()
    df = df.dropna(subset=["Invoice Date Key","Customer Key"])
    df["Invoice Date Key"] = pd.to_datetime(df["Invoice Date Key"], errors="coerce")
    snapshot = df["Invoice Date Key"].max()

    rfm = (df.groupby(["Customer Key","Customer","Region","Customer Value Tier"]).agg(
        Recency   = ("Invoice Date Key", lambda x: (snapshot - x.max()).days),
        Frequency = ("Sale Key","count"),
        Monetary  = ("Total Including Tax","sum"),
    ).reset_index())

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Segment"] = model.fit_predict(X)

    # label segments by monetary value
    seg_means = rfm.groupby("Segment")["Monetary"].mean().sort_values(ascending=False)
    seg_names = {seg: name for seg, name in zip(
        seg_means.index,
        ["Champions","Loyal Customers","At-Risk Customers","Churned/Lost"][:n_clusters])}
    rfm["Segment Name"] = rfm["Segment"].map(seg_names)

    return {"model": model, "rfm": rfm, "seg_names": seg_names}
