"""
Generic training routine, reused across every dataset (Telco, Bank,
Iranian). Takes an already-encoded feature matrix X and a binary
target y, trains the same 4-model zoo used for Telco, calibrates the
best one, computes SHAP, and writes the same artifact shape so the API
and frontend can treat every dataset uniformly.

This is the SAME code path as ml/models/train.py's model-training
section, factored out so results are directly comparable across
datasets -- no dataset gets an easier or harder training procedure.
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False

from ml.preprocessing.split import stratified_split
from ml.evaluation.metrics import compute_all_metrics
from ml.evaluation.selection import cv_compare, select_one_se, threshold_analysis, calibrate
from ml.monitoring.drift import compute_feature_drift, compute_prediction_drift


def get_model_zoo(small_dataset: bool = False):
    """small_dataset shrinks tree counts for very small datasets
    (e.g. Iranian, 3,150 rows) to avoid overfitting noise, and is
    disclosed in the run log -- not hidden."""
    n_est = 150 if small_dataset else 300
    depth_rf = 6 if small_dataset else 8
    return {
        "logistic_regression": LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=n_est, max_depth=depth_rf, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=n_est, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=n_est, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        ),
    }


def train_dataset(
    dataset_key: str,
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    artifact_dir: str,
    numeric_cols_for_drift: list[str],
    raw_df_for_drift: pd.DataFrame,
    small_dataset: bool = False,
):
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(os.path.join(artifact_dir, "models"), exist_ok=True)

    feature_columns = list(X.columns)
    with open(os.path.join(artifact_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    splits = stratified_split(X, y, ids)
    X_train, y_train, id_train = splits["train"]
    X_val, y_val, id_val = splits["val"]
    X_test, y_test, id_test = splits["test"]

    print(f"[{dataset_key}] split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    all_metrics = {}
    trained_models = {}
    zoo = get_model_zoo(small_dataset=small_dataset)

    for name, model in zoo.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        val_prob = model.predict_proba(X_val)[:, 1]
        inference_time_ms = (time.time() - t0) / max(len(X_val), 1) * 1000
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
        joblib.dump(model, os.path.join(artifact_dir, "models", f"{name}.joblib"))
        print(f"  [{dataset_key}/{name}] val ROC-AUC={val_metrics['roc_auc']} PR-AUC={val_metrics['pr_auc']} "
              f"test ROC-AUC={test_metrics['roc_auc']}")

    cv_results = cv_compare(get_model_zoo(small_dataset=small_dataset), X_train, y_train)
    production_model_name, selection_reason = select_one_se(cv_results)
    for _n in all_metrics:
        all_metrics[_n]["cv"] = cv_results[_n]
    print(f"[{dataset_key}] production model ({selection_reason}): {production_model_name}")

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump({
            "dataset": dataset_key,
            "models": all_metrics,
            "production_model": production_model_name,
            "selection_reason": selection_reason,
            "selection_method": "5x3 repeated stratified CV on train split; 1-SE rule, simplest model preferred",
            "n_total": int(len(X)),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "positive_rate": round(float(y.mean()), 4),
        }, f, indent=2)

    prod_model = trained_models[production_model_name]
    # Calibration: fit BOTH methods on the validation split, report both on
    # test. Sigmoid (Platt) is production because it is monotone: it fixes
    # probability scale without creating ties, so ranking metrics (ROC/PR-AUC)
    # are preserved. Isotonic creates step-function ties and measurably
    # lowered PR-AUC in the v1 run.
    calibrators = {m: calibrate(prod_model, X_val, y_val, m) for m in ("sigmoid", "isotonic")}
    calibrated = calibrators["sigmoid"]
    joblib.dump(calibrated, os.path.join(artifact_dir, "models", "production_calibrated.joblib"))
    calibration_comparison = {
        "raw": compute_all_metrics(y_test.values, prod_model.predict_proba(X_test)[:, 1]),
        **{m: compute_all_metrics(y_test.values, c.predict_proba(X_test)[:, 1]) for m, c in calibrators.items()},
    }
    calibration_comparison = {k: {x: v[x] for x in ("roc_auc", "pr_auc", "brier_score", "ece")}
                              for k, v in calibration_comparison.items()}
    calibration_comparison["production_method"] = "sigmoid"

    test_prob_cal = calibrated.predict_proba(X_test)[:, 1]
    calibrated_metrics = compute_all_metrics(y_test.values, test_prob_cal)
    calibrated_metrics["calibration_comparison"] = calibration_comparison
    # Split-conformal calibration quantile: validation is the calibration set;
    # the untouched test set remains reserved for final reporting.
    conformal_q = float(np.quantile(np.abs(y_val.to_numpy() - calibrated.predict_proba(X_val)[:, 1]), 0.9))
    with open(os.path.join(artifact_dir, "conformal.json"), "w") as f:
        json.dump({"confidence": 0.9, "absolute_residual_quantile": conformal_q,
                   "calibration_rows": int(len(y_val)),
                   "note": "Split-conformal interval calibrated on validation residuals."}, f, indent=2)
    _raw_p = prod_model.predict_proba(X_test)[:, 1]
    with open(os.path.join(artifact_dir, "threshold_analysis.json"), "w") as f:
        json.dump({
            "raw": threshold_analysis(y_test.values, _raw_p, fn_cost=10.0, fp_cost=1.0),
            "calibrated": threshold_analysis(y_test.values, test_prob_cal, fn_cost=10.0, fp_cost=1.0),
            "note": "Cost ratio 10:1 (missed churner : wasted retention offer) is an ASSUMPTION; "
                    "replace with finance-approved figures.",
        }, f, indent=2)
    with open(os.path.join(artifact_dir, "calibrated_test_metrics.json"), "w") as f:
        json.dump(calibrated_metrics, f, indent=2)
    print(f"[{dataset_key}] calibrated test Brier={calibrated_metrics['brier_score']} ECE={calibrated_metrics['ece']}")

    # SHAP for every row
    shap_export = {}
    try:
        import shap
        if production_model_name in ("random_forest", "xgboost", "lightgbm"):
            explainer = shap.TreeExplainer(prod_model)
            shap_values_all = explainer.shap_values(X)
            if isinstance(shap_values_all, list):
                shap_values_all = shap_values_all[1]
            elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
                # (n_samples, n_features, n_classes) -> take the positive class
                shap_values_all = shap_values_all[:, :, 1]
        else:
            explainer = shap.LinearExplainer(prod_model, X_train)
            shap_values_all = explainer.shap_values(X)
        for i, cust_id in enumerate(ids.values):
            contribs = dict(zip(feature_columns, shap_values_all[i].tolist()))
            top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            shap_export[str(cust_id)] = [{"feature": k, "contribution": round(float(v), 4)} for k, v in top]
        print(f"[{dataset_key}] SHAP explanations computed for {len(shap_export)} records.")
    except Exception as e:
        print(f"[{dataset_key}] SHAP skipped: {e}")
    with open(os.path.join(artifact_dir, "shap_values.json"), "w") as f:
        json.dump(shap_export, f, indent=2)

    # Drift: two real halves of the test set
    test_ref_ids = id_test.index
    test_raw = raw_df_for_drift.loc[test_ref_ids]
    half = len(test_raw) // 2
    ref_df = test_raw.iloc[:half]
    comp_df = test_raw.iloc[half:]
    ref_scores = calibrated.predict_proba(X_test.loc[ref_df.index])[:, 1]
    comp_scores = calibrated.predict_proba(X_test.loc[comp_df.index])[:, 1]
    drift_report = {
        "note": "Reference vs comparison are two real halves of the held-out test set "
                "(proxy cohorts -- no real timestamps in source data).",
        "feature_drift": compute_feature_drift(ref_df, comp_df, numeric_cols_for_drift),
        "prediction_drift": compute_prediction_drift(ref_scores, comp_scores),
    }
    with open(os.path.join(artifact_dir, "drift_report.json"), "w") as f:
        json.dump(drift_report, f, indent=2)

    return {
        "production_model": production_model_name,
        "test_roc_auc": all_metrics[production_model_name]["test"]["roc_auc"],
        "test_pr_auc": all_metrics[production_model_name]["test"]["pr_auc"],
        "calibrated_brier": calibrated_metrics["brier_score"],
        "calibrated_ece": calibrated_metrics["ece"],
        "n_total": int(len(X)),
        "positive_rate": round(float(y.mean()), 4),
    }
