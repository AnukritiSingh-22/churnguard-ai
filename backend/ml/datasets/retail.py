"""
UCI Online Retail dataset (531,282 real transactions, Dec 2010-Dec 2011,
4,340 real customers). Source: UCI ML Repository / Daqing Chen et al.,
mirrored at https://github.com/eaintkyawthmu/UCI_Online_Retail_Dataset_Cleaned_Version

IMPORTANT: this dataset has NO churn label -- it is raw transaction logs.
Any "churn" label here is DERIVED, not provided, and this is disclosed
everywhere it's used. We derive it the leakage-safe way:

  1. Pick a cutoff date splitting the year of data into an OBSERVATION
     window (used to build RFM features) and an OUTCOME window (used
     only to define the label).
  2. A customer is included only if they transacted at least once in
     the observation window (i.e. they were an established customer
     as of the cutoff -- we don't try to predict churn for someone
     who hadn't appeared yet).
  3. churn_flag = 1 if that customer made ZERO purchases in the
     outcome window, else 0. This is the standard "did they come
     back" churn definition for non-subscription retail, and it is
     computed the same way industry RFM-churn heuristics are: no
     future information leaks into the features, because the outcome
     window is strictly after the observation window.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np

RAW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "retail", "online_retail_raw.csv")

CUTOFF_DATE = "2011-09-09"  # ~3 months of outcome window before the dataset's real max date (2011-12-09)

NUMERIC_COLS = ["recency_days", "frequency", "monetary", "avg_order_value", "customer_tenure_days"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["line_total"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_customer_frame(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(CUTOFF_DATE)
    max_date = df["InvoiceDate"].max()

    obs = df[df["InvoiceDate"] <= cutoff]
    outcome = df[df["InvoiceDate"] > cutoff]

    established_customers = obs["CustomerID"].unique()

    agg = obs.groupby("CustomerID").agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("line_total", "sum"),
        n_line_items=("line_total", "count"),
    ).reset_index()

    agg["recency_days"] = (cutoff - agg["last_purchase"]).dt.days
    agg["customer_tenure_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days
    agg["avg_order_value"] = (agg["monetary"] / agg["frequency"]).round(2)

    returning_customer_ids = set(outcome["CustomerID"].unique())
    agg["churn_flag"] = (~agg["CustomerID"].isin(returning_customer_ids)).astype(int)

    # dominant country per customer, for a categorical feature
    country = obs.groupby("CustomerID")["Country"].agg(lambda s: s.value_counts().idxmax())
    agg = agg.merge(country.rename("Country"), on="CustomerID")

    agg["customer_id"] = "RETAIL-" + agg["CustomerID"].astype(str)
    agg["CLV"] = agg["monetary"].round(2)  # observed historical spend -- a real measured value, not an approximation

    print(f"[retail] observation window: start .. {CUTOFF_DATE} | outcome window: {CUTOFF_DATE} .. {max_date.date()}")
    print(f"[retail] established customers as of cutoff: {len(agg)} | churn rate: {agg['churn_flag'].mean():.4f}")

    return agg


def build_feature_frame(agg: pd.DataFrame) -> pd.DataFrame:
    feat = agg[NUMERIC_COLS + ["n_line_items", "Country"]].copy()
    # collapse long tail of countries to top 8 + "Other" to avoid a huge sparse one-hot
    top_countries = agg["Country"].value_counts().nlargest(8).index
    feat["Country"] = feat["Country"].where(feat["Country"].isin(top_countries), "Other")
    encoded = pd.get_dummies(feat, columns=["Country"])
    return encoded
