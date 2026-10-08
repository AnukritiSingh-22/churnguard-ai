from __future__ import annotations

import json
import secrets
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from app.core.config import ARTIFACT_DIR, DB_PATH
from app.services.auth import user_upload_dir


TARGET_NAME_HINTS = (
    "churn", "exited", "exit", "attrition", "cancel", "left", "target", "label",
)
ID_NAME_HINTS = ("customer", "account", "subscriber", "client", "user", "member", "id")
POSITIVE_LABELS = {"1", "1.0", "true", "yes", "y", "churn", "churned", "exited", "exit", "left", "leaved", "positive"}
NEGATIVE_LABELS = {"0", "0.0", "false", "no", "n", "active", " stayed", "stayed", "retained", "negative"}


def _normalise_name(value: object) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _infer_column(frame: pd.DataFrame, hints: tuple[str, ...], exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    candidates = []
    for column in frame.columns:
        if column in exclude:
            continue
        normalized = _normalise_name(column)
        score = sum(2 if normalized == hint else 1 for hint in hints if hint in normalized)
        if score:
            candidates.append((score, column))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _normalise_binary_target(series: pd.Series, column: str) -> tuple[pd.Series | None, dict]:
    values = series.dropna()
    unique = list(pd.unique(values))
    if len(unique) != 2:
        return None, {"status": "not_binary", "column": column, "unique_values": [str(v) for v in unique[:10]]}

    as_text = {str(value).strip().lower(): value for value in unique}
    positive = next((value for label, value in as_text.items() if label in POSITIVE_LABELS), None)
    negative = next((value for label, value in as_text.items() if label in NEGATIVE_LABELS), None)
    if positive is not None and negative is not None:
        mapping = {negative: 0, positive: 1}
        method = "semantic label mapping"
    else:
        ordered = sorted(unique, key=lambda value: str(value))
        mapping = {ordered[0]: 0, ordered[1]: 1}
        method = "deterministic sorted-class mapping"

    normalized = series.map(mapping)
    if normalized.isna().any():
        return None, {"status": "unmapped_values", "column": column}
    return normalized.astype("int8"), {
        "status": "normalized",
        "column": column,
        "method": method,
        "mapping": {str(key): value for key, value in mapping.items()},
        "positive_class": str(next(key for key, value in mapping.items() if value == 1)),
        "negative_class": str(next(key for key, value in mapping.items() if value == 0)),
    }


def _canonicalize_upload(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, str | None, str | None]:
    frame = frame.copy()
    original_columns = list(frame.columns)
    frame.columns = [str(column).strip() for column in frame.columns]
    dropped_columns = [
        column for column in frame.columns
        if str(column).lower().startswith("unnamed:") or frame[column].isna().all()
    ]
    if dropped_columns:
        frame = frame.drop(columns=dropped_columns)

    target = _infer_column(frame, TARGET_NAME_HINTS)
    customer_id = _infer_column(frame, ID_NAME_HINTS, exclude={target} if target else set())
    normalization = {
        "status": "unchanged",
        "original_columns": original_columns,
        "dropped_columns": dropped_columns,
        "target_column": target,
        "customer_id_column": customer_id,
    }
    if target:
        normalized_target, target_info = _normalise_binary_target(frame[target], target)
        normalization.update(target_info)
        if normalized_target is not None:
            frame[target] = normalized_target
    if dropped_columns or target:
        normalization["status"] = "normalized" if normalization.get("status") != "not_binary" else "needs_target_review"
    return frame, normalization, target, customer_id


def _prepare_features(frame: pd.DataFrame, target: str, customer_id: str | None = None) -> pd.DataFrame:
    drop = {target}
    if customer_id and customer_id in frame.columns:
        drop.add(customer_id)
    features = frame.drop(columns=list(drop)).copy()
    features = features.select_dtypes(exclude=["datetime", "datetimetz"])
    date_like = []
    for column in features.select_dtypes(include=["object", "string"]).columns:
        normalized_name = str(column).lower().replace("_", "")
        if not any(token in normalized_name for token in ("date", "timestamp", "time", "month", "period")):
            continue
        parsed = pd.to_datetime(features[column], errors="coerce")
        if parsed.notna().mean() >= 0.9:
            date_like.append(column)
    if date_like:
        features = features.drop(columns=date_like)
    for column in features.select_dtypes(include=["object", "string"]).columns:
        numeric = pd.to_numeric(features[column].astype(str).str.strip(), errors="coerce")
        if numeric.notna().mean() >= 0.9:
            features[column] = numeric
    return pd.get_dummies(features, dummy_na=True).replace(
        [float("-inf"), float("inf")], 0
    ).fillna(0)


def _uploaded_revenue_forecast(frame: pd.DataFrame, horizon: int = 3) -> dict:
    date_column = next(
        (
            column for column in frame.columns
            if (
                any(token in column.lower() for token in ("date", "timestamp", "time"))
                or column.lower().strip() in {"month", "period", "year"}
            )
            and column.lower().replace("_", "") not in {"monthlycharges"}
        ),
        None,
    )
    revenue_column = next(
        (column for column in frame.columns if column.lower().replace("_", "") in {
            "revenue", "sales", "amount", "totalamount", "totalrevenue", "monthlycharges",
        }),
        None,
    )
    quantity_column = next((column for column in frame.columns if column.lower() in {"quantity", "qty", "units"}), None)
    unit_price_column = next(
        (column for column in frame.columns if column.lower().replace("_", "") in {"unitprice", "price", "unitcost"}),
        None,
    )
    if not date_column or (not revenue_column and not (quantity_column and unit_price_column)):
        return {
            "available": False,
            "proxy": bool(revenue_column),
            "note": "A real sales forecast requires a date/time column and a revenue, sales, amount, or quantity × unit-price field with multiple periods.",
        }

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if revenue_column:
        revenue = pd.to_numeric(frame[revenue_column], errors="coerce")
        revenue_label = revenue_column
    else:
        quantity = pd.to_numeric(frame[quantity_column], errors="coerce").fillna(0)
        unit_price = pd.to_numeric(frame[unit_price_column], errors="coerce").fillna(0)
        revenue = quantity.clip(lower=0) * unit_price.clip(lower=0)
        revenue_label = f"{quantity_column} × {unit_price_column}"
    monthly = pd.DataFrame({"date": dates, "revenue": revenue}).dropna(subset=["date"])
    series = monthly.set_index("date")["revenue"].resample("MS").sum()
    if len(series) < 6:
        return {
            "available": False,
            "proxy": True,
            "note": f"Only {len(series)} monthly periods were found. At least 6 periods are required for a meaningful rolling-origin forecast.",
            "date_column": date_column,
            "revenue_column": revenue_label,
        }

    values = series.to_numpy(dtype=float)
    season = min(3, max(1, len(values) // 3))
    folds = []
    errors = []
    for end in range(max(season, 4), len(values)):
        prediction = float(values[max(0, end - season):end].mean())
        actual = float(values[end])
        errors.append(actual - prediction)
        folds.append({
            "period": series.index[end].strftime("%Y-%m"),
            "actual": round(actual, 2),
            "prediction": round(prediction, 2),
        })
    residual_q = float(np.quantile(np.abs(errors), 0.9)) if errors else 0.0
    last = values[-season:].copy()
    forecast = []
    for period in pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"):
        prediction = float(last.mean())
        forecast.append({
            "period": period.strftime("%Y-%m"),
            "actual": None,
            "prediction": round(prediction, 2),
            "lower": round(max(0.0, prediction - residual_q), 2),
            "upper": round(prediction + residual_q, 2),
            "interval_method": "90% empirical residual interval",
        })
        last = np.append(last[1:], prediction)
    denominator = max(float(np.sum(np.abs(values[-len(errors):]))), 1.0) if errors else 1.0
    return {
        "available": True,
        "proxy": False,
        "target": "monthly_revenue",
        "frequency": "MS",
        "model": "seasonal_naive",
        "date_column": date_column,
        "revenue_column": revenue_label,
        "seasonal_period_months": season,
        "history": [{"period": index.strftime("%Y-%m"), "actual": round(float(value), 2)} for index, value in series.items()],
        "forecast": forecast,
        "backtest": {
            "folds": folds,
            "mape": round(float(np.sum(np.abs(errors)) / denominator * 100), 4) if errors else 0.0,
            "rmse": round(float(np.sqrt(np.mean(np.square(errors)))), 2) if errors else 0.0,
            "mae": round(float(np.mean(np.abs(errors))), 2) if errors else 0.0,
            "honesty_note": "Rolling-origin holdout; each fold uses only earlier months.",
        },
        "interval_note": "The interval is empirical uncertainty from backtest residuals, not a guarantee of future sales.",
    }


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
    _, normalization, target, customer_id = _canonicalize_upload(frame)
    if normalization.get("status") == "unmapped_values":
        raise HTTPException(400, f"Target column '{target}' contains values that could not be normalized")
    upload_id = secrets.token_hex(12)
    upload_dir = user_upload_dir(user_id)
    raw_path = upload_dir / f"{upload_id}.original.csv"
    path = upload_dir / f"{upload_id}.csv"
    raw_path.write_bytes(content)
    full_frame = pd.read_csv(__import__("io").BytesIO(content))
    canonical_frame, _, _, _ = _canonicalize_upload(full_frame)
    canonical_frame.to_csv(path, index=False)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                 (upload_id, user_id, file.filename, str(path), len(canonical_frame), json.dumps(list(canonical_frame.columns))))
    conn.commit()
    conn.close()
    result = profile(user_id, upload_id)
    result["normalization"] = normalization
    result["original_filename"] = file.filename
    result["canonical_filename"] = f"{Path(file.filename).stem}.normalized.csv"
    if target and normalization.get("status") == "normalized":
        try:
            result["auto_run"] = train_churn(user_id, upload_id, target, customer_id)
            result["auto_trained"] = True
        except HTTPException as exc:
            result["auto_trained"] = False
            result["auto_train_error"] = exc.detail
    else:
        result["auto_trained"] = False
        result["auto_train_error"] = "Select a binary target column before training."
    return result


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
            "note": "The upload is normalized automatically when a binary target is detected. Original labels are retained in the upload directory for audit.",
            "normalization": {"status": "not_available_until_upload_response"}}


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
    monthly = _uploaded_revenue_forecast(frame)
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
    y, target_info = _normalise_binary_target(y_raw, target)
    if y is None:
        raise HTTPException(400, "Target must contain exactly two classes")
    valid_rows = y.notna()
    if not valid_rows.all():
        frame = frame.loc[valid_rows].reset_index(drop=True)
        y = y.loc[valid_rows].reset_index(drop=True)
    drop = {target}
    if customer_id and customer_id in frame.columns:
        drop.add(customer_id)
    features = frame.drop(columns=list(drop)).copy()
    features = features.select_dtypes(exclude=["datetime", "datetimetz"])
    X = _prepare_features(frame, target, customer_id)
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
    metrics = json.loads(metrics_path.read_text())
    frame = pd.read_csv(predictions_path)
    probabilities = frame.pop("churn_probability")
    target = metrics.get("target", "target")
    y = pd.Series(pd.Categorical(frame[target]).codes)
    X = _prepare_features(frame, target, metrics.get("customer_id"))
    class SimpleModel:
        def predict_proba(self, values):
            return pd.DataFrame({"negative": 1 - probabilities, "positive": probabilities}).to_numpy()
    return {
        "run_id": run_id,
        "upload_id": run[0],
        "metrics": metrics,
        "dashboard": _dashboard(frame, X, y, SimpleModel(), frame.index, target, metrics),
    }


def customer_rows(
    user_id: str,
    run_id: str,
    page: int = 1,
    page_size: int = 25,
    search: str = "",
    risk: str = "",
) -> dict:
    conn = sqlite3.connect(DB_PATH)
    run = conn.execute(
        "SELECT upload_id, artifact_dir FROM upload_runs WHERE id = ? AND user_id = ?",
        (run_id, user_id),
    ).fetchone()
    conn.close()
    if not run:
        raise HTTPException(404, "Training run not found")
    metrics = json.loads((Path(run[1]) / "metrics.json").read_text())
    predictions_path = Path(run[1]) / "predictions.csv"
    if not predictions_path.exists():
        raise HTTPException(404, "Training predictions are unavailable")
    frame = pd.read_csv(predictions_path).fillna("")
    probability = pd.to_numeric(frame.pop("churn_probability"), errors="coerce").fillna(0)
    frame["_row"] = range(len(frame))
    frame["_probability"] = probability
    frame["_risk"] = pd.cut(
        probability, [-1, .25, .5, .75, 2],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)
    query = search.strip().lower()
    if query:
        mask = frame.astype(str).apply(lambda column: column.str.lower().str.contains(query, regex=False)).any(axis=1)
        frame = frame.loc[mask]
    if risk in {"Low", "Medium", "High", "Critical"}:
        frame = frame.loc[frame["_risk"] == risk]
    frame = frame.sort_values("_probability", ascending=False)
    total = len(frame)
    page_size = max(1, min(page_size, 100))
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, pages))
    selected = frame.iloc[(page - 1) * page_size:page * page_size]
    customer_id_column = metrics.get("customer_id")
    columns = [column for column in frame.columns if column not in {"_row", "_probability", "_risk"}]
    rows = []
    for _, item in selected.iterrows():
        values = {column: item[column] for column in columns}
        rows.append({
            "row": int(item["_row"]),
            "customer_id": str(item.get(customer_id_column, f"Row {int(item['_row']) + 1}")) if customer_id_column else f"Row {int(item['_row']) + 1}",
            "probability": round(float(item["_probability"]), 4),
            "risk": str(item["_risk"]),
            "values": {column: (None if value == "" else value) for column, value in values.items()},
        })
    return {
        "run_id": run_id,
        "customer_id_column": customer_id_column,
        "columns": columns,
        "rows": rows,
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
        "search": search,
        "risk": risk,
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
    X = _prepare_features(frame, target, customer_id_column)
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
