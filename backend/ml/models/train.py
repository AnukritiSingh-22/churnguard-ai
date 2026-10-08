from __future__ import annotations
import json
import os
import sys
import time
import sqlite3
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ml.preprocessing.clean import load_raw, clean, data_quality_report
from ml.preprocessing.leakage_audit import run_leakage_audit
from ml.preprocessing.split import stratified_split
from ml.features.build_features import build_feature_frame, build_clv, engagement_score, FEATURE_COLUMNS
from ml.evaluation.metrics import compute_all_metrics
from ml.evaluation.selection import cv_compare, select_one_se, threshold_analysis, calibrate
from ml.monitoring.drift import compute_feature_drift, compute_prediction_drift

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "churnguard.db")

os.makedirs(MODEL_DIR, exist_ok=True)


def get_model_zoo():
    return {
        "logistic_regression": LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=2.7, random_state=42, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        ),
    }


def main():
    print("=" * 70)
    print("ChurnGuard AI -- real training run on the IBM Telco dataset")
    print("=" * 70)

    raw = load_raw()
    df = clean(raw)
    print(f"Loaded and cleaned {len(df)} real customer records.")

    dq_report = data_quality_report(raw)
    with open(os.path.join(ARTIFACT_DIR, "data_quality.json"), "w") as f:
        json.dump(dq_report, f, indent=2)
    print(f"Data quality report written. Missing values found in: {list(dq_report['missing_values'].keys())}")

    audit = run_leakage_audit(df)
    with open(os.path.join(ARTIFACT_DIR, "leakage_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Leakage audit: excluded={audit['excluded_structural']}, flagged={audit['flagged_for_review']}")

    df["CLV"] = build_clv(df)
    df["EngagementScore"] = engagement_score(df)

    X = build_feature_frame(df)
    feature_columns = list(X.columns)
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    y = df["churn_flag"]
    ids = df["customerID"]

    splits = stratified_split(X, y, ids)
    X_train, y_train, id_train = splits["train"]
    X_val, y_val, id_val = splits["val"]
    X_test, y_test, id_test = splits["test"]
    print(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)} "
          f"(stratified random split; dataset has no timestamps, see split.py docstring)")

    all_metrics = {}
    trained_models = {}
    zoo = get_model_zoo()

    for name, model in zoo.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        val_prob = model.predict_proba(X_val)[:, 1]
        inference_time_ms = (time.time() - t0) / len(X_val) * 1000

        test_prob = model.predict_proba(X_test)[:, 1]

        val_metrics = compute_all_metrics(y_val.values, val_prob)
        test_metrics = compute_all_metrics(y_test.values, test_prob)

        all_metrics[name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "train_time_seconds": round(train_time, 3),
            "inference_time_ms_per_sample": round(inference_time_ms, 4),
        }
        trained_models[name] = model
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"  [{name}] val ROC-AUC={val_metrics['roc_auc']}  PR-AUC={val_metrics['pr_auc']}  "
              f"Brier={val_metrics['brier_score']}  test ROC-AUC={test_metrics['roc_auc']}")

    cv_results = cv_compare(get_model_zoo(), X_train, y_train)
    production_model_name, selection_reason = select_one_se(cv_results)
    for _n in all_metrics:
        all_metrics[_n]["cv"] = cv_results[_n]
    print(f"\nSelected production model ({selection_reason}): {production_model_name}")

    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "models": all_metrics,
            "production_model": production_model_name,
            "selection_reason": selection_reason,
            "selection_method": "5x3 repeated stratified CV on train split; 1-SE rule, simplest model preferred",
        }, f, indent=2)

    prod_model = trained_models[production_model_name]
    calibrators = {m: calibrate(prod_model, X_val, y_val, m) for m in ("sigmoid", "isotonic")}
    calibrated = calibrators["sigmoid"]
    joblib.dump(calibrated, os.path.join(MODEL_DIR, "production_calibrated.joblib"))
    calibration_comparison = {
        "raw": compute_all_metrics(y_test.values, prod_model.predict_proba(X_test)[:, 1]),
        **{m: compute_all_metrics(y_test.values, c.predict_proba(X_test)[:, 1]) for m, c in calibrators.items()},
    }
    calibration_comparison = {k: {x: v[x] for x in ("roc_auc", "pr_auc", "brier_score", "ece")}
                              for k, v in calibration_comparison.items()}
    calibration_comparison["production_method"] = "sigmoid"

    test_prob_calibrated = calibrated.predict_proba(X_test)[:, 1]
    calibrated_metrics = compute_all_metrics(y_test.values, test_prob_calibrated)
    conformal_q = float(np.quantile(np.abs(y_val.to_numpy() - calibrated.predict_proba(X_val)[:, 1]), 0.9))
    with open(os.path.join(ARTIFACT_DIR, "conformal.json"), "w") as f:
        json.dump({"confidence": 0.9, "absolute_residual_quantile": conformal_q,
                   "calibration_rows": int(len(y_val)),
                   "note": "Split-conformal interval calibrated on validation residuals."}, f, indent=2)
    calibrated_metrics["calibration_comparison"] = calibration_comparison
    _raw_p = prod_model.predict_proba(X_test)[:, 1]
    with open(os.path.join(ARTIFACT_DIR, "threshold_analysis.json"), "w") as f:
        json.dump({
            "raw": threshold_analysis(y_test.values, _raw_p, fn_cost=10.0, fp_cost=1.0),
            "calibrated": threshold_analysis(y_test.values, test_prob_calibrated, fn_cost=10.0, fp_cost=1.0),
            "note": "Cost ratio 10:1 (missed churner : wasted retention offer) is an ASSUMPTION; "
                    "replace with finance-approved figures.",
        }, f, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "calibrated_test_metrics.json"), "w") as f:
        json.dump(calibrated_metrics, f, indent=2)
    print(f"Calibrated production model test Brier={calibrated_metrics['brier_score']} "
          f"ECE={calibrated_metrics['ece']}")

    shap_export = {}
    try:
        import shap
        if production_model_name in ("random_forest", "xgboost", "lightgbm"):
            explainer = shap.TreeExplainer(prod_model)
            shap_values_all = explainer.shap_values(X)
            if isinstance(shap_values_all, list):
                shap_values_all = shap_values_all[1]
            elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
                shap_values_all = shap_values_all[:, :, 1]
        else:
            explainer = shap.LinearExplainer(prod_model, X_train)
            shap_values_all = explainer.shap_values(X)

        for i, cust_id in enumerate(ids.values):
            contribs = dict(zip(feature_columns, shap_values_all[i].tolist()))
            top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            shap_export[cust_id] = [{"feature": k, "contribution": round(float(v), 4)} for k, v in top]
        print(f"SHAP explanations computed for all {len(shap_export)} customers "
              f"(model: {production_model_name}).")
    except Exception as e:
        print(f"SHAP computation skipped: {e}")

    with open(os.path.join(ARTIFACT_DIR, "shap_values.json"), "w") as f:
        json.dump(shap_export, f, indent=2)

    test_df = df.loc[X_test.index].copy()
    half = len(test_df) // 2
    ref_df = test_df.iloc[:half]
    comp_df = test_df.iloc[half:]
    ref_scores = calibrated.predict_proba(X_test.loc[ref_df.index])[:, 1]
    comp_scores = calibrated.predict_proba(X_test.loc[comp_df.index])[:, 1]

    drift_report = {
        "note": "Reference vs comparison are two real halves of the held-out test set "
                "(ordered by original row index as a proxy for 'earlier vs later' cohorts -- "
                "the source dataset has no real timestamps). See ml/monitoring/drift.py.",
        "feature_drift": compute_feature_drift(ref_df, comp_df, ["tenure", "MonthlyCharges", "TotalCharges"]),
        "prediction_drift": compute_prediction_drift(ref_scores, comp_scores),
    }
    with open(os.path.join(ARTIFACT_DIR, "drift_report.json"), "w") as f:
        json.dump(drift_report, f, indent=2)
    print(f"Drift report computed. Prediction PSI={drift_report['prediction_drift']['psi']} "
          f"({drift_report['prediction_drift']['status']})")

    all_prob = calibrated.predict_proba(X)[:, 1]
    build_database(df, all_prob, production_model_name)
    print(f"\nSQLite database written to {DB_PATH}")
    print("Training pipeline complete. All numbers above are computed, not hardcoded.")


def build_database(df: pd.DataFrame, churn_prob: np.ndarray, production_model_name: str):
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    out = df.copy()
    out["churn_probability"] = churn_prob.round(4)

    def risk_bucket(p):
        if p < 0.25:
            return "Low"
        elif p < 0.5:
            return "Medium"
        elif p < 0.75:
            return "High"
        return "Critical"

    out["risk_level"] = out["churn_probability"].apply(risk_bucket)

    out["primary_risk_driver"] = out.apply(_primary_driver, axis=1)
    out["production_model"] = production_model_name

    cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn", "churn_flag", "CLV",
        "EngagementScore", "churn_probability", "risk_level",
        "primary_risk_driver", "production_model",
    ]
    out[cols].to_sql("customers", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_customer_id ON customers(customerID)")
    conn.execute("CREATE INDEX idx_risk_level ON customers(risk_level)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            action TEXT,
            date TEXT,
            predicted_retention_prob REAL,
            actual_outcome TEXT,
            revenue_impact REAL,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()


def _primary_driver(row) -> str:
    if row["Contract"] == "Month-to-month" and row["tenure"] < 12:
        return "Short tenure on month-to-month contract"
    if row["MonthlyCharges"] > 80:
        return "High monthly charges"
    if row["EngagementScore"] < 30:
        return "Low service engagement"
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        return "No tech support add-on"
    if row["PaymentMethod"] == "Electronic check":
        return "Higher-risk payment method"
    return "Mixed low-level factors"


if __name__ == "__main__":
    main()
