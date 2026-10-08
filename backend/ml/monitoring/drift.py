"""
Drift monitoring: Population Stability Index (PSI) and Kolmogorov-Smirnov
statistic, computed for real between two real data slices.

Because the Telco dataset has no timestamps, we cannot observe genuine
production drift over calendar time. To still provide an honestly-computed
drift signal (rather than fabricating one), we split the held-out test set
in half using the customer's row order in the original file as a proxy
"earlier vs later" cohort, and compute PSI/KS between those two real
halves. This is clearly labeled in the API/UI as
"reference vs comparison window (proxy cohorts, no real timestamps in
source data)" -- it is a real statistical computation, just not a
real-calendar-time drift measurement.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(reference, quantiles))
    if len(cut_points) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=cut_points)
    comp_counts, _ = np.histogram(comparison, bins=cut_points)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-4, None)
    comp_pct = np.clip(comp_counts / max(len(comparison), 1), 1e-4, None)
    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


def status_for_psi(score: float) -> str:
    if score < 0.1:
        return "stable"
    elif score < 0.25:
        return "warning"
    return "significant_drift"


def compute_feature_drift(reference_df: pd.DataFrame, comparison_df: pd.DataFrame, numeric_cols: list[str]) -> list[dict]:
    results = []
    for col in numeric_cols:
        ref_vals = reference_df[col].dropna().values
        comp_vals = comparison_df[col].dropna().values
        if len(ref_vals) < 5 or len(comp_vals) < 5:
            continue
        psi_score = psi(ref_vals, comp_vals)
        ks_stat, ks_p = ks_2samp(ref_vals, comp_vals)
        results.append({
            "feature": col,
            "psi": round(psi_score, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 4),
            "status": status_for_psi(psi_score),
            "reference_mean": round(float(np.mean(ref_vals)), 3),
            "comparison_mean": round(float(np.mean(comp_vals)), 3),
        })
    return results


def compute_prediction_drift(reference_scores: np.ndarray, comparison_scores: np.ndarray) -> dict:
    score = psi(reference_scores, comparison_scores)
    ks_stat, ks_p = ks_2samp(reference_scores, comparison_scores)
    return {
        "psi": round(score, 4),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": round(float(ks_p), 4),
        "status": status_for_psi(score),
        "reference_mean_score": round(float(np.mean(reference_scores)), 4),
        "comparison_mean_score": round(float(np.mean(comparison_scores)), 4),
    }
