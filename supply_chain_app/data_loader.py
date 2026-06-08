"""
Data Loader — Supply Chain Galaxy Schema
Loads and joins all Dimension + Fact tables with caching.
"""
import os
import pandas as pd
import numpy as np
import streamlit as st

# ── path resolution ────────────────────────────────────────────────────────────
# supply_chain_app/ lives inside the repo root.
# The /data/ folder is a sibling of supply_chain_app/ at the repo root.
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))          # …/supply_chain_app
_REPO_ROOT = os.path.dirname(_APP_DIR)                          # repo root
DIM_DIR   = os.path.join(_REPO_ROOT, "data", "Dimensions Sheets V2")
FACT_DIR  = os.path.join(_REPO_ROOT, "data", "Facts Sheets")


# ── low-level loaders ─────────────────────────────────────────────────────────
def _csv(name):
    return pd.read_csv(os.path.join(DIM_DIR, name))

def _xl(name):
    return pd.read_excel(os.path.join(FACT_DIR, name), engine="openpyxl")


@st.cache_data(show_spinner=False)
def load_dimensions():
    date     = _csv("Dimension.Date.csv")
    customer = _csv("Dimension.Customer.csv")
    stock    = _csv("Dimension.Stock Item.csv")
    employee = _csv("Dimension.Employee.csv")
    city     = _csv("Dimension.City.csv")
    supplier = _csv("Dimension.Supplier.csv")
    txn_type = _csv("Dimension.Transaction Type.csv")
    payment  = _csv("Dimension.Payment Method.csv")

    date["Date"] = pd.to_datetime(date["Date"], dayfirst=True, errors="coerce")

    return {
        "date": date, "customer": customer, "stock": stock,
        "employee": employee, "city": city, "supplier": supplier,
        "txn_type": txn_type, "payment": payment,
    }


@st.cache_data(show_spinner=False)
def load_facts():
    sale     = _xl("Fact.Sale.xlsx")
    order    = _xl("Fact.Order.xlsx")
    purchase = _xl("Fact.Purchase.xlsx")
    stock_h  = _xl("Fact.Stock Holding.xlsx")
    movement = _xl("Fact.Movement.xlsx")
    txn      = pd.read_csv(os.path.join(FACT_DIR, "Fact.Transaction.csv"))
    return {
        "sale": sale, "order": order, "purchase": purchase,
        "stock_holding": stock_h, "movement": movement, "transaction": txn,
    }


# ── enriched / joined mart tables ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def mart_sales():
    dims  = load_dimensions()
    facts = load_facts()

    sale = facts["sale"].copy()
    sale["Invoice Date Key"] = pd.to_datetime(sale["Invoice Date Key"], errors="coerce")

    date_df = dims["date"][["Date","Calendar Year","Calendar Month Number",
                             "Short Month","Fiscal Year","ISO Week Number"]].copy()
    date_df = date_df.rename(columns={"Date": "Invoice Date Key"})

    sale = sale.merge(date_df, on="Invoice Date Key", how="left")
    sale = sale.merge(
        dims["customer"][["Customer Key","Customer","Category",
                          "Buying Group","Region","Customer Value Tier"]],
        on="Customer Key", how="left")
    sale = sale.merge(
        dims["stock"][["Stock Item Key","Stock Item","Brand",
                       "Stock Category","Subcategory","Unit Price",
                       "Cost Price","Lead Time Days","Tax Rate"]],
        on="Stock Item Key", how="left", suffixes=("","_dim"))
    sale = sale.merge(
        dims["city"][["City Key","City","State Province",
                      "Country","Sales Territory"]],
        on="City Key", how="left")

    sale["Margin"] = sale["Profit"]
    sale["Margin %"] = np.where(
        sale["Total Excluding Tax"] > 0,
        sale["Profit"] / sale["Total Excluding Tax"] * 100, np.nan)

    return sale


@st.cache_data(show_spinner=False)
def mart_inventory():
    dims  = load_dimensions()
    facts = load_facts()

    sh = facts["stock_holding"].copy()
    sh = sh.merge(
        dims["stock"][["Stock Item Key","Stock Item","Brand",
                       "Stock Category","Subcategory","Unit Price",
                       "Cost Price","Lead Time Days","Availability"]],
        on="Stock Item Key", how="left")

    sh["Stock Value"]    = sh["Quantity On Hand"] * sh["Cost Price"]
    sh["Days of Supply"] = np.where(sh["Reorder Level"] > 0,
                                    sh["Quantity On Hand"] / sh["Reorder Level"], np.nan)
    sh["Reorder Flag"]   = sh["Quantity On Hand"] <= sh["Reorder Level"]
    sh["Overstock Flag"] = sh["Quantity On Hand"] > sh["Target Stock Level"]
    sh["Stockout Risk"]  = (sh["Quantity On Hand"] / sh["Target Stock Level"].replace(0, np.nan)).fillna(0)

    return sh


@st.cache_data(show_spinner=False)
def mart_purchase():
    dims  = load_dimensions()
    facts = load_facts()

    pur = facts["purchase"].copy()
    pur["Date Key"] = pd.to_datetime(pur["Date Key"], errors="coerce")

    pur = pur.merge(
        dims["supplier"][["Supplier Key","Supplier","Category",
                          "Supplier Rating","Country","Lead Time Days (Supplier)",
                          "Supplier Tier","Delivery Speed Category","Region"]],
        on="Supplier Key", how="left")
    pur = pur.merge(
        dims["stock"][["Stock Item Key","Stock Item","Stock Category","Subcategory","Cost Price"]],
        on="Stock Item Key", how="left")

    pur["Fulfillment Rate"] = np.where(
        pur["Ordered Outers"] > 0,
        pur["Received Outers"] / pur["Ordered Outers"] * 100, np.nan)
    pur["Purchase Value"] = pur["Received Outers"] * pur["Cost Price"].fillna(0)
    pur["Year"]  = pur["Date Key"].dt.year
    pur["Month"] = pur["Date Key"].dt.month

    return pur


@st.cache_data(show_spinner=False)
def mart_movement():
    dims  = load_dimensions()
    facts = load_facts()

    mv = facts["movement"].copy()
    mv["Date Key"] = pd.to_datetime(mv["Date Key"], errors="coerce")

    date_df = dims["date"][["Date","Calendar Year","Calendar Month Number",
                             "Short Month","ISO Week Number"]].copy()
    date_df = date_df.rename(columns={"Date": "Date Key"})

    mv = mv.merge(date_df, on="Date Key", how="left")
    mv = mv.merge(
        dims["stock"][["Stock Item Key","Stock Item","Stock Category",
                       "Subcategory","Lead Time Days"]],
        on="Stock Item Key", how="left")
    mv = mv.merge(
        dims["txn_type"][["Transaction Type Key","Transaction Type",
                          "Transaction_Direction"]],
        on="Transaction Type Key", how="left")

    return mv


@st.cache_data(show_spinner=False)
def mart_transaction():
    dims  = load_dimensions()
    facts = load_facts()

    txn = facts["transaction"].copy()
    txn["Date Key"] = pd.to_datetime(txn["Date Key"], errors="coerce")

    txn = txn.merge(
        dims["txn_type"][["Transaction Type Key","Transaction Type",
                          "Transaction_Direction","Revenue_Impact"]],
        on="Transaction Type Key", how="left")
    txn = txn.merge(
        dims["payment"][["Payment Method Key","Payment Method",
                         "Risk_Level","Digital_Payment_Flag"]],
        on="Payment Method Key", how="left")
    txn = txn.merge(
        dims["customer"][["Customer Key","Customer","Category","Region"]],
        on="Customer Key", how="left")

    txn["Year"]  = txn["Date Key"].dt.year
    txn["Month"] = txn["Date Key"].dt.month

    return txn