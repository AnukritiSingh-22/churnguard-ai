"""
Trains the Bank Customer Churn and Iranian Telecom Churn datasets
through the same generic pipeline used for Telco, and builds a SQLite
table for each so the API can serve real predictions for all of them.

Run with: python -m ml.models.train_multi   (from backend/)
"""
from __future__ import annotations
import os
import sys
import sqlite3
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.datasets import bank, iranian, retail
from ml.models.train_generic import train_dataset
from ml.preprocessing.leakage_audit import run_leakage_audit
from ml.evaluation.ablation import run_ablation

ARTIFACT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "churnguard.db")


def run_bank():
    print("=" * 70)
    print("Training on Bank Customer Churn dataset (10,000 real records)")
    print("=" * 70)
    df = bank.load_and_clean()
    df["CLV"] = bank.build_clv(df)

    audit_df = df.rename(columns={"churn_flag": "churn_flag"})
    audit = run_leakage_audit(audit_df.drop(columns=["customer_id", "CLV"]), target_col="churn_flag")
    artifact_dir = os.path.join(ARTIFACT_ROOT, "bank")
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "leakage_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[bank] leakage audit: excluded={audit['excluded_structural']}, flagged={audit['flagged_for_review']}")

    X = bank.build_feature_frame(df)
    y = df["churn_flag"]
    ids = df["customer_id"]

    result = train_dataset(
        dataset_key="bank", X=X, y=y, ids=ids, artifact_dir=artifact_dir,
        numeric_cols_for_drift=["CreditScore", "Age", "Balance", "EstimatedSalary"],
        raw_df_for_drift=df,
    )

    prod_model = joblib.load(os.path.join(artifact_dir, "models", "production_calibrated.joblib"))
    all_prob = prod_model.predict_proba(X)[:, 1]
    build_db_table(
        table_name="bank_customers", df=df, prob=all_prob,
        id_col="customer_id", clv_col="CLV",
        extra_cols=["Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
                    "CreditScore", "EstimatedSalary", "IsActiveMember"],
        production_model=result["production_model"],
    )
    return result


def run_iranian():
    print("=" * 70)
    print("Training on Iranian Telecom Churn dataset (3,150 real records)")
    print("=" * 70)
    df = iranian.load_and_clean()
    df["CLV"] = iranian.build_clv(df)

    audit = run_leakage_audit(df.drop(columns=["customer_id", "CLV"]), target_col="churn_flag")
    artifact_dir = os.path.join(ARTIFACT_ROOT, "iranian")
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "leakage_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[iranian] leakage audit: excluded={audit['excluded_structural']}, flagged={audit['flagged_for_review']}")

    X = iranian.build_feature_frame(df)
    y = df["churn_flag"]
    ids = df["customer_id"]

    ablation = run_ablation(X, y, {
        "complains_status": ["complains", "status"],
        "usage_volume": ["seconds_of_use", "frequency_of_use", "frequency_of_sms", "distinct_called_numbers"],
        "complains_status_usage": ["complains", "status", "seconds_of_use", "frequency_of_use",
                                   "frequency_of_sms", "distinct_called_numbers"],
    })
    ablation["interpretation"] = ("Signal survives removal of complains/status, so it is not a single-column leak; "
                                  "but usage features near the label window may reflect already-disengaged customers.")
    with open(os.path.join(artifact_dir, "ablation.json"), "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"[iranian] ablation: {ablation}")

    result = train_dataset(
        dataset_key="iranian", X=X, y=y, ids=ids, artifact_dir=artifact_dir,
        numeric_cols_for_drift=["call_failure", "seconds_of_use", "frequency_of_use", "customer_value"],
        raw_df_for_drift=df,
        small_dataset=True,
    )

    prod_model = joblib.load(os.path.join(artifact_dir, "models", "production_calibrated.joblib"))
    all_prob = prod_model.predict_proba(X)[:, 1]
    build_db_table(
        table_name="iranian_customers", df=df, prob=all_prob,
        id_col="customer_id", clv_col="CLV",
        extra_cols=["call_failure", "complains", "subscription_length", "charge_amount",
                    "seconds_of_use", "frequency_of_use", "frequency_of_sms", "age", "status"],
        production_model=result["production_model"],
    )
    return result


def build_db_table(table_name, df, prob, id_col, clv_col, extra_cols, production_model):
    conn = sqlite3.connect(DB_PATH)
    out = df[[id_col, "churn_flag", clv_col] + extra_cols].copy()
    out["churn_probability"] = np.round(prob, 4)

    def risk_bucket(p):
        if p < 0.25:
            return "Low"
        elif p < 0.5:
            return "Medium"
        elif p < 0.75:
            return "High"
        return "Critical"

    out["risk_level"] = out["churn_probability"].apply(risk_bucket)
    out["production_model"] = production_model
    out = out.rename(columns={id_col: "customer_id", clv_col: "CLV"})
    out.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name}(customer_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_risk ON {table_name}(risk_level)")
    conn.commit()
    conn.close()
    print(f"[{table_name}] SQLite table written with {len(out)} rows.")


def run_retail():
    print("=" * 70)
    print("Training on Online Retail dataset (531K real transactions -> RFM)")
    print("=" * 70)
    tx = retail.load_and_clean()
    agg = retail.build_customer_frame(tx)

    audit = run_leakage_audit(
        agg.drop(columns=["customer_id", "CLV", "CustomerID", "first_purchase", "last_purchase"]),
        target_col="churn_flag",
    )
    artifact_dir = os.path.join(ARTIFACT_ROOT, "retail")
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "leakage_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[retail] leakage audit: excluded={audit['excluded_structural']}, flagged={audit['flagged_for_review']}")

    X = retail.build_feature_frame(agg)
    y = agg["churn_flag"]
    ids = agg["customer_id"]

    result = train_dataset(
        dataset_key="retail", X=X, y=y, ids=ids, artifact_dir=artifact_dir,
        numeric_cols_for_drift=["recency_days", "frequency", "monetary", "avg_order_value"],
        raw_df_for_drift=agg,
    )

    prod_model = joblib.load(os.path.join(artifact_dir, "models", "production_calibrated.joblib"))
    all_prob = prod_model.predict_proba(X)[:, 1]
    build_db_table(
        table_name="retail_customers", df=agg, prob=all_prob,
        id_col="customer_id", clv_col="CLV",
        extra_cols=["recency_days", "frequency", "monetary", "avg_order_value",
                    "customer_tenure_days", "Country"],
        production_model=result["production_model"],
    )
    return result


if __name__ == "__main__":
    results = {}
    results["bank"] = run_bank()
    results["iranian"] = run_iranian()
    results["retail"] = run_retail()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"{k}: production={v['production_model']} test_ROC-AUC={v['test_roc_auc']} "
              f"test_PR-AUC={v['test_pr_auc']} calibrated_Brier={v['calibrated_brier']} "
              f"n={v['n_total']} positive_rate={v['positive_rate']}")
