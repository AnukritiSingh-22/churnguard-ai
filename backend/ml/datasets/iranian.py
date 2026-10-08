from __future__ import annotations
import os
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "iranian", "iranian_churn_raw.csv")

NUMERIC_COLS = [
    "call_failure", "charge_amount", "seconds_of_use", "frequency_of_use",
    "frequency_of_sms", "distinct_called_numbers", "age", "customer_value",
    "subscription_length",
]
CATEGORICAL_COLS = ["age_group", "tariff_plan", "status"]
BINARY_COLS = ["complains"]

LEAKAGE_EXCLUDE = {
    "churn": "Raw target label.",
}


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["churn_flag"] = df["churn"].astype(int)
    df = df.reset_index(drop=True)
    df["customer_id"] = "IRTEL-" + (df.index + 1).astype(str).str.zfill(5)
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = df[NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS].copy()
    for c in CATEGORICAL_COLS:
        feat[c] = feat[c].astype(int).astype(str)
    encoded = pd.get_dummies(feat, columns=CATEGORICAL_COLS)
    return encoded


def build_clv(df: pd.DataFrame) -> pd.Series:
    return df["customer_value"].round(2)
