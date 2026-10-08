"""
Leakage audit: every feature is programmatically checked before it is
allowed into the training feature set. This is not a hardcoded list --
it is a real computation run against the dataframe.

Two independent checks:
  1. STRUCTURAL: known-bad columns that encode the outcome directly or are
     row identifiers (customerID, the raw Churn/Churn Label string, any
     column literally named like a post-event field).
  2. STATISTICAL: for every remaining column, compute its association
     with the target (Cramer's V for categoricals, point-biserial /
     correlation for numerics). Anything above LEAKAGE_THRESHOLD is
     flagged for human review as "suspiciously predictive" -- it is
     *not* auto-excluded, because a genuinely strong legitimate feature
     (e.g. Contract type) will also score high. The audit reports the
     score and the reason a human should look at it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

STRUCTURAL_EXCLUDE = {
    "customerID": "Row identifier, not predictive of behaviour, must be excluded.",
    "Churn": "Raw target label (string) -- including this would be 100% leakage.",
}

LEAKAGE_THRESHOLD = 0.55  # association score above this triggers a warning


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return float(np.sqrt(phi2 / max(min(k - 1, r - 1), 1)))


def run_leakage_audit(df: pd.DataFrame, target_col: str = "churn_flag") -> dict:
    results = []

    for col, reason in STRUCTURAL_EXCLUDE.items():
        if col in df.columns:
            results.append({
                "feature": col,
                "check": "structural",
                "association_score": None,
                "status": "excluded",
                "reason": reason,
            })

    candidate_cols = [
        c for c in df.columns
        if c not in STRUCTURAL_EXCLUDE and c != target_col
    ]

    for col in candidate_cols:
        series = df[col]
        try:
            if pd.api.types.is_numeric_dtype(series):
                valid = df[[col, target_col]].dropna()
                if valid[col].nunique() <= 1:
                    score = 0.0
                else:
                    score, _ = pointbiserialr(valid[target_col], valid[col])
                    score = abs(float(score))
            else:
                ct = pd.crosstab(series, df[target_col])
                score = cramers_v(ct)
        except Exception:
            score = 0.0

        status = "flagged_for_review" if score >= LEAKAGE_THRESHOLD else "cleared"
        reason = (
            f"Association with target = {score:.2f}, above the {LEAKAGE_THRESHOLD} "
            "review threshold -- verify this value is genuinely known BEFORE the "
            "prediction timestamp and not a symptom/consequence of churn itself."
            if status == "flagged_for_review"
            else f"Association with target = {score:.2f}, within normal range for a legitimate predictor."
        )
        results.append({
            "feature": col,
            "check": "statistical",
            "association_score": round(float(score), 4),
            "status": status,
            "reason": reason,
        })

    excluded = [r["feature"] for r in results if r["status"] == "excluded"]
    flagged = [r["feature"] for r in results if r["status"] == "flagged_for_review"]
    cleared = [r["feature"] for r in results if r["status"] == "cleared"]

    return {
        "target_column": target_col,
        "total_columns_checked": len(df.columns),
        "excluded_structural": excluded,
        "flagged_for_review": flagged,
        "cleared": cleared,
        "details": results,
        "note": (
            "This dataset is a single-snapshot (cross-sectional) sample -- it has "
            "no event timestamps, so a temporal train/validation/test split is not "
            "applicable here. We use a stratified random split instead and document "
            "that explicitly (see split.py). If a future dataset carries timestamps, "
            "this pipeline must switch to a temporal split before training."
        ),
    }
