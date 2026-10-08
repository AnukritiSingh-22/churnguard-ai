"""
All evaluation metrics computed for real -- nothing hardcoded.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    precision_score, brier_score_loss, confusion_matrix, roc_curve,
    precision_recall_curve,
)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    points = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        points.append({
            "bin_lo": round(float(lo), 2),
            "bin_hi": round(float(hi), 2),
            "predicted_avg": round(float(y_prob[mask].mean()), 4),
            "observed_freq": round(float(y_true[mask].mean()), 4),
            "count": int(mask.sum()),
        })
    return points


def compute_all_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "ece": round(expected_calibration_error(y_true, y_prob), 4),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//50)]],
            "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//50)]],
        },
        "pr_curve": {
            "precision": [round(float(x), 4) for x in prec[::max(1, len(prec)//50)]],
            "recall": [round(float(x), 4) for x in rec[::max(1, len(rec)//50)]],
        },
        "reliability_curve": reliability_curve(y_true, y_prob),
        "n_samples": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }
