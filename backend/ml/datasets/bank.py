"""
Bank Customer Churn dataset (10,000 real customer records).
Source: https://github.com/Aslm-Fawzy/Churn_Classification_for_Bank_Customers
(originally https://www.kaggle.com/shrutimechlearn/churn-modelling)

Target: Exited (1 = customer closed their account).
"""
from __future__ import annotations
import os
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "bank", "bank_churn_raw.csv")

CATEGORICAL_COLS = ["Geography", "Gender"]
NUMERIC_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
BINARY_COLS = ["HasCrCard", "IsActiveMember"]

LEAKAGE_EXCLUDE = {
    "RowNumber": "Row identifier, not predictive.",
    "CustomerId": "Row identifier, not predictive.",
    "Surname": "Personal identifier, not a legitimate predictor -- excluded to avoid encoding identity/name-based bias.",
    "Exited": "Raw target label.",
}


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["churn_flag"] = df["Exited"]
    # CustomerId as a stable, human-readable ID for the API/UI
    df["customer_id"] = "BANK-" + df["CustomerId"].astype(str)
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = df[NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS].copy()
    encoded = pd.get_dummies(feat, columns=CATEGORICAL_COLS)
    return encoded


def build_clv(df: pd.DataFrame) -> pd.Series:
    """Approximation: EstimatedSalary is used as a proxy for account value
    since this dataset has no monthly-charge equivalent; CLV here is a
    fraction of salary scaled by tenure, documented as an approximation
    like the Telco CLV formula, not a measured value."""
    return (df["EstimatedSalary"] * 0.02 * (df["Tenure"].clip(lower=1))).round(2)
