"""
Counterfactual / "what-if" scenarios.

This re-runs the REAL production model on a perturbed copy of the
customer's actual feature row (e.g. "what if this customer had a
two-year contract instead"). The resulting probability is a genuine
model output, not a fabricated number.

What this is NOT: a causal effect estimate. The Telco dataset is
observational -- customers were not randomly assigned to contract
types -- so "predicted probability changes" reflects correlational
model behaviour, not a guarantee that changing the contract would
cause that customer's real-world risk to change. Every response is
labeled accordingly.
"""
import json
import joblib
import pandas as pd
from app.core.config import MODEL_DIR, ARTIFACT_DIR
from app.core.db import get_connection

_model = None
_feature_columns = None


def _load():
    global _model, _feature_columns
    if _model is None:
        _model = joblib.load(MODEL_DIR / "production_calibrated.joblib")
        _feature_columns = json.load(open(ARTIFACT_DIR / "feature_columns.json"))
    return _model, _feature_columns


def _row_to_feature_vector(row: dict, feature_columns: list[str]) -> pd.DataFrame:
    from ml.features.build_features import CATEGORICAL_COLS, NUMERIC_COLS, BINARY_COLS
    base = {}
    for c in NUMERIC_COLS + BINARY_COLS:
        base[c] = row[c]
    for c in CATEGORICAL_COLS:
        base[c] = row[c]
    df = pd.DataFrame([base])
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[feature_columns]
    return encoded


def generate_scenarios(customer_id: str):
    model, feature_columns = _load()
    conn = get_connection()
    row = conn.execute("SELECT * FROM customers WHERE customerID = ?", (customer_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    row = dict(row)

    baseline_vec = _row_to_feature_vector(row, feature_columns)
    baseline_prob = float(model.predict_proba(baseline_vec)[:, 1][0])

    scenarios = [{
        "name": "current_state",
        "description": "No intervention -- current predicted risk.",
        "predicted_churn_probability": round(baseline_prob, 4),
        "delta": 0.0,
    }]

    if row["Contract"] == "Month-to-month":
        alt = dict(row)
        alt["Contract"] = "Two year"
        vec = _row_to_feature_vector(alt, feature_columns)
        prob = float(model.predict_proba(vec)[:, 1][0])
        scenarios.append({
            "name": "switch_to_two_year_contract",
            "description": "Model re-scored assuming this customer moves to a two-year contract.",
            "predicted_churn_probability": round(prob, 4),
            "delta": round(prob - baseline_prob, 4),
        })

    if row["PaymentMethod"] == "Electronic check":
        alt = dict(row)
        alt["PaymentMethod"] = "Bank transfer (automatic)"
        vec = _row_to_feature_vector(alt, feature_columns)
        prob = float(model.predict_proba(vec)[:, 1][0])
        scenarios.append({
            "name": "switch_to_autopay",
            "description": "Model re-scored assuming this customer switches to automatic bank transfer.",
            "predicted_churn_probability": round(prob, 4),
            "delta": round(prob - baseline_prob, 4),
        })

    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        alt = dict(row)
        alt["TechSupport"] = "Yes"
        vec = _row_to_feature_vector(alt, feature_columns)
        prob = float(model.predict_proba(vec)[:, 1][0])
        scenarios.append({
            "name": "add_tech_support",
            "description": "Model re-scored assuming this customer adds the Tech Support add-on.",
            "predicted_churn_probability": round(prob, 4),
            "delta": round(prob - baseline_prob, 4),
        })

    return {
        "customer_id": customer_id,
        "baseline_churn_probability": round(baseline_prob, 4),
        "scenarios": scenarios,
        "disclaimer": "Model-generated counterfactual scenario -- a real re-scoring of the "
                       "production model on perturbed inputs, not a causal guarantee. This "
                       "dataset is observational, so the model may reflect correlation "
                       "(e.g. contract type correlates with other loyalty signals) rather "
                       "than the causal effect of the change itself.",
    }
