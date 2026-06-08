# Supply Chain Intelligence Platform
## End-to-End Predictive Analytics · Galaxy Schema DWH · ML-Powered

---

### 📂 Project Structure
```
supply_chain_app/
├── app.py            # Main Streamlit application (7 modules)
├── data_loader.py    # Galaxy schema data mart builder (cached)
├── ml_models.py      # 6 ML algorithms
├── charts.py         # Plotly dark-theme chart library
├── requirements.txt
└── README.md

data/
├── Dimensions Sheets V2/   # 8 dimension tables
└── Facts Sheets/           # 6 fact tables
```

---

### 🚀 Setup & Run

```bash
# 1. Install dependencies
pip install -r supply_chain_app/requirements.txt

# 2. Run the app (from project root, where /data folder exists)
streamlit run supply_chain_app/app.py
```

The app auto-discovers data from the `/data/` directory relative to the project root.

---

### 🧠 ML Models

| Module | Algorithm | Target | Key Metric |
|--------|-----------|--------|-----------|
| Demand Forecasting | Random Forest Regressor | Monthly units per category | MAPE ~7%, R²=0.53 |
| Stockout Risk | Gradient Boosting Classifier | HIGH/MEDIUM/LOW risk per SKU | ROC-AUC |
| Customer Churn | Random Forest Classifier | Churn probability per customer | ROC-AUC |
| Supplier Quality | Random Forest Regressor → Score | 0-100 composite quality score | R² |
| Anomaly Detection | Isolation Forest | Anomalous transactions | Contamination=5% |
| Customer Segmentation | KMeans | Champions/Loyal/At-Risk/Churned | Cluster compactness |

---

### 🗄️ Galaxy Schema

**Fact Tables (centre)**
- `Fact.Sale` — 228K rows — revenue, profit, taxes per invoice line
- `Fact.Order` — 231K rows — order lines, quantities, pricing
- `Fact.Movement` — 236K rows — stock in/out movements
- `Fact.Transaction` — 99K rows — financial transactions
- `Fact.Purchase` — 8.4K rows — supplier purchase orders
- `Fact.Stock Holding` — 227 rows — point-in-time inventory snapshot

**Dimension Tables (arms)**
- `Dim.Date` · `Dim.Customer` · `Dim.Stock Item` · `Dim.City`
- `Dim.Employee` · `Dim.Supplier` · `Dim.Transaction Type` · `Dim.Payment Method`

---

### 📊 App Modules

1. **Executive Dashboard** — Revenue/profit KPIs, trend charts, territory breakdown
2. **Demand Forecasting** — Category-level ML demand prediction with configurable horizon
3. **Inventory Risk** — Per-SKU stockout risk scoring + reorder alerts
4. **Customer Intelligence** — Churn prediction + RFM 3D segmentation
5. **Supplier Analytics** — Quality scoring, fulfillment heatmap, benchmarking
6. **Anomaly Detection** — Isolation Forest on 99K financial transactions
7. **Data Explorer** — Full Galaxy Schema browser with search + CSV export
