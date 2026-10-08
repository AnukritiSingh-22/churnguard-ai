"""Dataset registry: single source of truth for what each dataset exposes.
Adding a dataset = one entry here + artifacts/<key>/ + a DB table."""
from __future__ import annotations
from app.core.config import ARTIFACT_DIR

TELCO_MEAN_CLV = 597.04  # scale the assumed intervention costs were written for

DATASETS: dict[str, dict] = {
    "telco": {
        "name": "IBM Telco Customer Churn", "domain": "Telecom", "table": "customers",
        "id_col": "customerID", "artifact_dir": ARTIFACT_DIR, "currency": "$",
        "segment_col": "Contract", "segment_label": "Contract type",
        "columns": [("Contract", "Contract", "text"), ("tenure", "Tenure (mo)", "int"),
                    ("MonthlyCharges", "Monthly charges", "money"), ("InternetService", "Internet", "text"),
                    ("churn_risk_6m", "6-mo risk (survival)", "pct")],
        "detail_fields": ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                          "InternetService", "OnlineSecurity", "TechSupport", "StreamingTV", "Contract",
                          "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"],
        "capabilities": {"survival": True, "counterfactual": True},
        "label_note": "Real churn label (Churn column).",
    },
    "bank": {
        "name": "Bank Customer Churn", "domain": "Banking", "table": "bank_customers",
        "id_col": "customer_id", "artifact_dir": ARTIFACT_DIR / "bank", "currency": "",
        "segment_col": "Geography", "segment_label": "Country",
        "columns": [("Geography", "Country", "text"), ("Age", "Age", "int"), ("Tenure", "Tenure (yr)", "int"),
                    ("Balance", "Balance", "money"), ("NumOfProducts", "Products", "int")],
        "detail_fields": ["Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "CreditScore",
                          "EstimatedSalary", "IsActiveMember"],
        "capabilities": {"survival": False, "counterfactual": False},
        "label_note": "Real churn label (Exited).",
    },
    "iranian": {
        "name": "Iranian Telecom Churn", "domain": "Telecom", "table": "iranian_customers",
        "id_col": "customer_id", "artifact_dir": ARTIFACT_DIR / "iranian", "currency": "",
        "segment_col": "status", "segment_label": "Status (1=active, 2=non-active)",
        "columns": [("subscription_length", "Subscription (mo)", "int"), ("seconds_of_use", "Seconds of use", "int"),
                    ("frequency_of_use", "Use freq.", "int"), ("complains", "Complained", "int"),
                    ("status", "Status", "int")],
        "detail_fields": ["call_failure", "complains", "subscription_length", "charge_amount", "seconds_of_use",
                          "frequency_of_use", "frequency_of_sms", "age", "status"],
        "capabilities": {"survival": False, "counterfactual": False},
        "label_note": "Real label (churn by month 12). Very high AUC: see ablation caveat.",
    },
    "retail": {
        "name": "Online Retail (RFM-derived churn)", "domain": "Retail", "table": "retail_customers",
        "id_col": "customer_id", "artifact_dir": ARTIFACT_DIR / "retail", "currency": "£",
        "segment_col": "Country", "segment_label": "Country",
        "columns": [("Country", "Country", "text"), ("recency_days", "Recency (d)", "int"),
                    ("frequency", "Orders", "int"), ("monetary", "Spend", "money"),
                    ("avg_order_value", "Avg order", "money")],
        "detail_fields": ["Country", "recency_days", "frequency", "monetary", "avg_order_value", "customer_tenure_days"],
        "capabilities": {"survival": False, "counterfactual": False},
        "label_note": "DERIVED label (no purchase in the 3 months after cutoff): a heuristic, not a business outcome.",
    },
}


def get(key: str) -> dict:
    if key not in DATASETS:
        raise KeyError(key)
    return DATASETS[key]
