from __future__ import annotations
import pandas as pd
import numpy as np

CATEGORICAL_COLS = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

FEATURE_COLUMNS = CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS


def build_clv(df: pd.DataFrame) -> pd.Series:
    """
    Customer Lifetime Value approximation:
        CLV = MonthlyCharges * expected_remaining_months

    expected_remaining_months is estimated from the empirical average
    tenure-at-churn for customers on the SAME contract type in this
    dataset (i.e. "customers on a month-to-month plan who do churn,
    churn after this many months on average, historically") minus the
    customer's current tenure, floored at 3 months. This is an
    ACTUARIAL APPROXIMATION computed from real historical data in this
    dataset, not a black-box guess and not a "true" CLV -- it is labeled
    as such everywhere it is surfaced in the API/UI.
    """
    avg_tenure_at_churn_by_contract = (
        df[df["churn_flag"] == 1].groupby("Contract")["tenure"].mean()
    )
    overall_avg = df[df["churn_flag"] == 1]["tenure"].mean()

    def expected_remaining(row):
        typical = avg_tenure_at_churn_by_contract.get(row["Contract"], overall_avg)
        remaining = max(typical - row["tenure"], 3)
        if row["Contract"] == "One year":
            remaining = max(remaining, 12 - (row["tenure"] % 12))
        elif row["Contract"] == "Two year":
            remaining = max(remaining, 24 - (row["tenure"] % 24))
        return remaining

    remaining_months = df.apply(expected_remaining, axis=1)
    clv = (df["MonthlyCharges"] * remaining_months).round(2)
    return clv


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Returns (features_df, target_series) using ONLY leakage-cleared
    columns. Categorical columns are one-hot encoded; the encoder
    mapping is saved alongside the trained model so inference uses the
    identical schema."""
    feat = df[FEATURE_COLUMNS].copy()
    feat_encoded = pd.get_dummies(feat, columns=CATEGORICAL_COLS, drop_first=False)
    return feat_encoded


def engagement_score(df: pd.DataFrame) -> pd.Series:
    """A composite 0-100 'engagement' proxy built from real service
    columns (how many add-on services the customer actually uses),
    used for driver narratives, not as a hidden model feature that
    duplicates information already in the one-hot columns."""
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    used = df[service_cols].apply(lambda c: (c == "Yes").astype(int)).sum(axis=1)
    max_services = len(service_cols)
    tenure_component = (df["tenure"].clip(upper=48) / 48) * 40
    service_component = (used / max_services) * 40
    paperless_component = df["PaperlessBilling"].fillna(0) * 10
    autopay_component = df["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int) * 10
    score = tenure_component + service_component + paperless_component + autopay_component
    return score.round(1)
