"""
Data cleaning for the IBM Telco Customer Churn dataset.

Source: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
7,043 real customer records, 21 raw columns. This is the actual public IBM
sample dataset -- no synthetic values are introduced here beyond the CLV
approximation documented in features/build_features.py.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np


RAW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "telco_raw.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning steps (all documented, none fabricate values):
      1. TotalCharges is stored as string in the source CSV; 11 rows are
         blank (customers with tenure == 0, i.e. brand new accounts that
         have not been billed yet). We coerce to numeric and set those
         rows to 0.0, which is the factually correct billed total for a
         zero-tenure customer -- not an imputed guess.
      2. Normalise the "No internet service" / "No phone service"
         categorical values down to "No" for the dependent add-on
         columns, since they are functionally "No" and the extra level
         adds no information (this mirrors the well-known standard
         treatment of this dataset and is fully reversible/auditable).
      3. Binary Yes/No columns -> 0/1.
      4. Target: Churn -> churn_flag (1 = churned).
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    zero_tenure_mask = df["tenure"] == 0
    df.loc[zero_tenure_mask & df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    # any remaining NaNs (should be none) are left as NaN and reported by
    # the data-quality module rather than silently filled.

    add_on_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for c in add_on_cols:
        df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for c in binary_cols:
        df[c] = df[c].map({"Yes": 1, "No": 0})

    df["churn_flag"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Computed data-quality metrics -- feeds the /api/data-quality endpoint.
    Every number here is derived directly from the dataframe, nothing is
    hardcoded."""
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    dup_rows = int(df.duplicated(subset=[c for c in df.columns if c != "customerID"]).sum())
    class_counts = df["Churn"].value_counts().to_dict()
    imbalance_ratio = round(class_counts.get("No", 0) / max(class_counts.get("Yes", 1), 1), 2)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    outliers = {}
    for c in numeric_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers[c] = int(((col < lo) | (col > hi)).sum())

    return {
        "rows": int(n_rows),
        "columns": int(n_cols),
        "missing_values": {k: int(v) for k, v in missing.items() if v > 0},
        "missing_pct": {k: float(v) for k, v in missing_pct.items() if v > 0},
        "duplicate_rows": dup_rows,
        "class_balance": {k: int(v) for k, v in class_counts.items()},
        "imbalance_ratio_no_to_yes": imbalance_ratio,
        "outlier_counts_iqr": outliers,
    }
