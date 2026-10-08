from __future__ import annotations

import json
import secrets
import sqlite3
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from app.core.config import ARTIFACT_DIR, DB_PATH
from app.services.auth import user_upload_dir


def _row(upload_id: str, user_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT * FROM uploads WHERE id = ? AND user_id = ?", (upload_id, user_id)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Upload not found")
    return row


async def save_upload(user_id: str, file: UploadFile) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "CSV is limited to 100 MB")
    try:
        frame = pd.read_csv(__import__("io").BytesIO(content), nrows=10000)
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}") from exc
    if frame.empty or len(frame.columns) < 2:
        raise HTTPException(400, "CSV must contain rows and at least two columns")
    upload_id = secrets.token_hex(12)
    path = user_upload_dir(user_id) / f"{upload_id}.csv"
    path.write_bytes(content)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                 (upload_id, user_id, file.filename, str(path), len(frame), json.dumps(list(frame.columns))))
    conn.commit()
    conn.close()
    return profile(user_id, upload_id)


def profile(user_id: str, upload_id: str) -> dict:
    row = _row(upload_id, user_id)
    frame = pd.read_csv(row[3], nrows=10000)
    columns = []
    for name in frame.columns:
        lower = name.lower()
        role = "feature"
        if any(x in lower for x in ("churn", "exited", "target", "label")):
            role = "possible_target"
        elif any(x in lower for x in ("id", "customer", "account")):
            role = "possible_id"
        elif any(x in lower for x in ("date", "time")):
            role = "possible_timestamp"
        columns.append({"name": name, "dtype": str(frame[name].dtype), "missing": int(frame[name].isna().sum()), "role": role})
    return {"upload_id": upload_id, "filename": row[2], "rows_sampled": len(frame), "total_rows": row[4],
            "columns": columns, "preview": frame.head(5).fillna("").to_dict(orient="records"),
            "note": "Profile only. Training requires an explicit target column and does not invent labels."}


def list_uploads(user_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT u.id, u.filename, u.rows, u.created_at,
        (SELECT r.id FROM upload_runs r WHERE r.upload_id = u.id ORDER BY r.created_at DESC LIMIT 1)
        FROM uploads u WHERE u.user_id = ? ORDER BY u.created_at DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [{"upload_id": r[0], "filename": r[1], "rows": r[2], "created_at": r[3], "run_id": r[4]} for r in rows]


def _dashboard(frame: pd.DataFrame, X: pd.DataFrame, y: pd.Series, model, test_index, target: str, metrics: dict) -> dict:
    probabilities = model.predict_proba(X)[:, 1]
    risk = pd.Series(pd.cut(probabilities, [-1, .25, .5, .75, 2], labels=["Low", "Medium", "High", "Critical"]))
    counts = risk.value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
    actual = y.loc[test_index].to_numpy()
    test_prob = probabilities[test_index]
    matrix = confusion_matrix(actual, (test_prob >= .5).astype(int), labels=[0, 1])
    bins = []
    for low in [0, .2, .4, .6, .8]:
        upper = low + .2 if low < .8 else 1.000001
        mask = (test_prob >= low) & (test_prob < upper)
        bins.append({"bucket": f"{int(low * 100)}-{int((low + .2) * 100)}%", "predicted": round(float(test_prob[mask].mean()) if mask.any() else 0, 4), "actual": round(float(actual[mask].mean()) if mask.any() else 0, 4), "n": int(mask.sum())})
    drift = []
    for column in X.columns[:20]:
        train_values = X.loc[~X.index.isin(test_index), column]
        test_values = X.loc[test_index, column]
        if train_values.nunique() > 1:
            mean_shift = abs(float(test_values.mean() - train_values.mean())) / (float(train_values.std()) + 1e-9)
            status = "high" if mean_shift >= 0.5 else ("watch" if mean_shift >= 0.25 else "stable")
        else:
            mean_shift, status = 0.0, "stable"
        drift.append({"feature": column, "shift": round(mean_shift, 3), "status": status})
    coefficients = []
    try:
        estimator = model.calibrated_classifiers_[0].estimator
        values = estimator.coef_[0]
        coefficients = [{"feature": name, "impact": round(float(value), 4)} for name, value in sorted(zip(X.columns, values), key=lambda item: abs(item[1]), reverse=True)[:10]]
    except (AttributeError, IndexError):
        coefficients = []
    monthly = None
    numeric_revenue = next((c for c in frame.columns if c.lower() in {"revenue", "sales", "amount", "monthly_charges", "monthlycharges"}), None)
    date_column = next((c for c in frame.columns if "date" in c.lower() or "time" in c.lower()), None)
    if numeric_revenue:
        revenue = pd.to_numeric(frame[numeric_revenue], errors="coerce").fillna(0)
        total = float(revenue.sum())
        monthly = {"available": False, "proxy": True, "label": numeric_revenue, "current": round(total, 2), "forecast": [{"period": "Next period", "prediction": round(total / 3, 2), "lower": round(total / 3 * .8, 2), "upper": round(total / 3 * 1.2, 2)}], "note": "Revenue-at-risk proxy from the uploaded amount field. A real sales forecast requires a timestamp and repeated periods." if not date_column else "A revenue field exists, but a time-based backtest requires multiple dated periods."}
    else:
        monthly = {"available": False, "proxy": False, "note": "No revenue or sales column was mapped. Upload a dated transaction file with revenue to enable a real sales forecast."}
    return {
        "metrics": metrics,
        "risk_distribution": [{"risk_level": str(label), "count": int(count)} for label, count in counts.items()],
        "positive_rate": round(float(y.mean()), 4),
        "predicted_rate": round(float(probabilities.mean()), 4),
        "confusion_matrix": {"tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]), "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])},
        "calibration": bins,
        "feature_drivers": coefficients,
        "drift": drift,
        "review_queue": [
            {
                "row": int(index),
                "customer_id": str(frame.iloc[index][metrics["customer_id"]]) if metrics.get("customer_id") in frame.columns else f"Row {int(index) + 1}",
                "probability": round(float(probabilities[index]), 4),
                "risk": str(risk.iloc[index]),
            }
            for index in probabilities.argsort()[::-1][:10]
        ],
        "revenue_forecast": monthly,
        "honesty_note": "All results are computed from this uploaded file. Drift compares the training and holdout partitions; it is not production time drift.",
    }


def train_churn(user_id: str, upload_id: str, target: str, customer_id: str | None = None) -> dict:
    row = _row(upload_id, user_id)
    frame = pd.read_csv(row[3])
    if target not in frame.columns:
        raise HTTPException(400, f"Target column '{target}' was not found")
    y_raw = frame[target]
    if y_raw.nunique(dropna=True) != 2:
        raise HTTPException(400, "Target must contain exactly two classes")
    y = pd.Series(pd.Categorical(y_raw).codes, index=frame.index)
    drop = {target}
    if customer_id and customer_id in frame.columns:
        drop.add(customer_id)
    features = frame.drop(columns=list(drop)).copy()
    features = features.select_dtypes(exclude=["datetime", "datetimetz"])
    X = pd.get_dummies(features, dummy_na=True).replace([float("inf"), float("-inf")], 0).fillna(0)
    if X.shape[1] == 0:
        raise HTTPException(400, "No usable feature columns remain after removing the target and ID")
    if y.value_counts().min() < 8:
        raise HTTPException(400, "Each target class needs at least eight rows for a calibrated holdout")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    baseline = LogisticRegression(max_iter=2000, class_weight="balanced")
    baseline.fit(X_train, y_train)
    model = CalibratedClassifierCV(baseline, method="sigmoid", cv=3)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    metrics = {"roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
               "pr_auc": round(float(average_precision_score(y_test, probability)), 4),
               "brier_score": round(float(brier_score_loss(y_test, probability)), 4),
               "train_rows": len(X_train), "test_rows": len(X_test),
               "target": target, "customer_id": customer_id,
               "note": "User upload baseline: stratified holdout; use a timestamp mapping for temporal evaluation."}
    run_id = secrets.token_hex(12)
    run_dir = ARTIFACT_DIR / "uploads" / user_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "feature_columns.json").write_text(json.dumps(list(X.columns)))
    predictions = frame.loc[X_test.index].copy()
    predictions["churn_probability"] = probability
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    dashboard = _dashboard(frame, X, y, model, X_test.index, target, metrics)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO upload_runs VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                 (run_id, upload_id, user_id, target, customer_id, str(run_dir)))
    conn.commit()
    conn.close()
    return {"run_id": run_id, "upload_id": upload_id, "metrics": metrics,
            "predictions_sample": predictions.head(25).to_dict(orient="records"), "dashboard": dashboard}


def run_dashboard(user_id: str, run_id: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    run = conn.execute("SELECT upload_id, artifact_dir FROM upload_runs WHERE id = ? AND user_id = ?", (run_id, user_id)).fetchone()
    conn.close()
    if not run:
        raise HTTPException(404, "Training run not found")
    metrics_path = Path(run[1]) / "metrics.json"
    predictions_path = Path(run[1]) / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        raise HTTPException(404, "Training artifacts are incomplete")
    frame = pd.read_csv(predictions_path)
    probabilities = frame.pop("churn_probability")
    target = json.loads(metrics_path.read_text()).get("target", "target")
    y = pd.Series(pd.Categorical(frame[target]).codes)
    X = pd.get_dummies(frame.drop(columns=[target]), dummy_na=True).select_dtypes(exclude=["datetime", "datetimetz"]).fillna(0)
    class SimpleModel:
        def predict_proba(self, values):
            return pd.DataFrame({"negative": 1 - probabilities, "positive": probabilities}).to_numpy()
    metrics = json.loads(metrics_path.read_text())
    return {
        "run_id": run_id,
        "upload_id": run[0],
        "metrics": metrics,
        "dashboard": _dashboard(frame, X, y, SimpleModel(), frame.index, target, metrics),
    }


def customer_detail(user_id: str, run_id: str, row_number: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    run = conn.execute(
        "SELECT upload_id, artifact_dir FROM upload_runs WHERE id = ? AND user_id = ?",
        (run_id, user_id),
    ).fetchone()
    conn.close()
    if not run:
        raise HTTPException(404, "Training run not found")
    run_dir = Path(run[1])
    metrics = json.loads((run_dir / "metrics.json").read_text())
    upload = _row(run[0], user_id)
    frame = pd.read_csv(upload[3])
    if row_number < 0 or row_number >= len(frame):
        raise HTTPException(404, "Uploaded customer row not found")
    target = metrics["target"]
    customer_id_column = metrics.get("customer_id")
    drop = {target}
    if customer_id_column in frame.columns:
        drop.add(customer_id_column)
    features = frame.drop(columns=list(drop))
    X = pd.get_dummies(features, dummy_na=True).replace([float("inf"), float("-inf")], 0).fillna(0)
    model = __import__("joblib").load(run_dir / "model.joblib")
    X = X.reindex(columns=json.loads((run_dir / "feature_columns.json").read_text()), fill_value=0)
    probability = float(model.predict_proba(X.iloc[[row_number]])[:, 1][0])
    risk = "Low" if probability < .25 else "Medium" if probability < .5 else "High" if probability < .75 else "Critical"
    drivers = []
    try:
        estimator = model.calibrated_classifiers_[0].estimator
        values = estimator.coef_[0] * X.iloc[row_number].to_numpy(dtype=float)
        drivers = [
            {"feature": name, "value": float(X.iloc[row_number][name]), "contribution": round(float(value), 5)}
            for name, value in sorted(zip(X.columns, values), key=lambda item: abs(item[1]), reverse=True)[:12]
            if value != 0
        ]
    except (AttributeError, IndexError):
        pass
    attributes = frame.iloc[row_number].where(pd.notna(frame.iloc[row_number]), None).to_dict()
    return {
        "run_id": run_id,
        "row": row_number,
        "customer_id": str(attributes.get(customer_id_column)) if customer_id_column else f"Row {row_number + 1}",
        "churn_probability": round(probability, 4),
        "risk_level": risk,
        "observed_outcome": str(attributes.get(target)),
        "attributes": attributes,
        "drivers": drivers,
        "note": "Feature contributions are model associations from the uploaded baseline, not causal explanations or SHAP values.",
    }
