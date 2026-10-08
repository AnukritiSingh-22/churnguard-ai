"""
Robust model selection + threshold analysis (v2).

Why: the earlier pipeline picked the production model by PR-AUC on ONE
validation split (~1,400 rows). Differences of ~0.005 between models are
within noise at that size. Here:

  1. Every model is scored with repeated stratified K-fold CV on the
     TRAIN split only (val is kept for calibration, test is never touched).
  2. The production model is the SIMPLEST model whose mean CV PR-AUC is
     within one standard error of the best ("1-SE rule"). Simplicity
     order: logistic_regression < xgboost < lightgbm < random_forest.
  3. Threshold analysis compares raw vs calibrated probabilities at the
     same business operating points, instead of at a fixed 0.5.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold

SIMPLICITY_ORDER = ["logistic_regression", "xgboost", "lightgbm", "random_forest"]


def cv_compare(zoo: dict, X, y, n_splits: int = 5, n_repeats: int = 3, seed: int = 42) -> dict:
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    out = {}
    for name, model in zoo.items():
        pr, roc, br = [], [], []
        for tr, va in rskf.split(X, y):
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            pr.append(average_precision_score(y.iloc[va], p))
            roc.append(roc_auc_score(y.iloc[va], p))
            br.append(brier_score_loss(y.iloc[va], p))
        k = len(pr)
        out[name] = {
            "pr_auc_mean": round(float(np.mean(pr)), 4),
            "pr_auc_std": round(float(np.std(pr, ddof=1)), 4),
            "pr_auc_se": round(float(np.std(pr, ddof=1) / np.sqrt(k)), 4),
            "roc_auc_mean": round(float(np.mean(roc)), 4),
            "roc_auc_std": round(float(np.std(roc, ddof=1)), 4),
            "brier_mean": round(float(np.mean(br)), 4),
            "n_folds": k,
        }
    return out


def select_one_se(cv: dict) -> tuple[str, str]:
    best = max(cv, key=lambda k: cv[k]["pr_auc_mean"])
    cutoff = cv[best]["pr_auc_mean"] - cv[best]["pr_auc_se"]
    for name in SIMPLICITY_ORDER:
        if name in cv and cv[name]["pr_auc_mean"] >= cutoff:
            reason = (f"Simplest model within 1 SE of best CV PR-AUC "
                      f"(best={best} {cv[best]['pr_auc_mean']}; cutoff={cutoff:.4f}).")
            return name, reason
    return best, "Best CV PR-AUC."


def threshold_analysis(y_true, y_prob, fn_cost: float, fp_cost: float, n_grid: int = 99) -> dict:
    """Expected cost = fn_cost*FN + fp_cost*FP across a threshold grid.
    fn_cost / fp_cost are BUSINESS ASSUMPTIONS passed in, not learned."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    rows = []
    for t in np.linspace(0.01, 0.99, n_grid):
        pred = p >= t
        tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        rows.append({"threshold": round(float(t), 2), "precision": round(prec, 4),
                     "recall": round(rec, 4), "flagged": int(pred.sum()),
                     "cost": float(fn_cost * fn + fp_cost * fp)})
    best = min(rows, key=lambda r: r["cost"])
    # threshold-free comparison: lift in top decile
    order = np.argsort(-p); k = max(len(p) // 10, 1)
    lift = float(y[order[:k]].mean() / max(y.mean(), 1e-9))
    return {"assumptions": {"fn_cost": fn_cost, "fp_cost": fp_cost,
                            "note": "Illustrative business assumptions, not measured."},
            "best_threshold": best, "top_decile_lift": round(lift, 3), "grid": rows[::4]}


def calibrate(prod_model, X_val, y_val, method: str):
    """Calibrate a fitted model on the validation split."""
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        c = CalibratedClassifierCV(FrozenEstimator(prod_model), method=method)
    except ImportError:
        c = CalibratedClassifierCV(prod_model, method=method, cv="prefit")
    return c.fit(X_val, y_val)


def paired_compare(baseline, challengers: dict, X, y, n_splits: int = 5, n_repeats: int = 3, seed: int = 42) -> dict:
    """Paired fold-by-fold comparison of challengers vs one baseline on IDENTICAL
    folds. 'statistically_better' = mean PR-AUC gain > 1 SE of the paired difference;
    'practically_meaningful' = gain >= 0.02 PR-AUC. Folds overlap, so the SE is
    optimistic: treat it as a screen, not a formal test."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    models = {"baseline": baseline, **challengers}
    scores = {k: {"pr": [], "roc": [], "brier": []} for k in models}
    for tr, va in rskf.split(X, y):
        for name, model in models.items():
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            scores[name]["pr"].append(average_precision_score(y.iloc[va], p))
            scores[name]["roc"].append(roc_auc_score(y.iloc[va], p))
            scores[name]["brier"].append(brier_score_loss(y.iloc[va], p))
    base_pr = np.array(scores["baseline"]["pr"])
    out = {"baseline": {"pr_auc_mean": round(float(base_pr.mean()), 4),
                        "roc_auc_mean": round(float(np.mean(scores["baseline"]["roc"])), 4),
                        "brier_mean": round(float(np.mean(scores["baseline"]["brier"])), 4)}}
    for name in challengers:
        pr = np.array(scores[name]["pr"])
        d = pr - base_pr
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        out[name] = {
            "pr_auc_mean": round(float(pr.mean()), 4),
            "roc_auc_mean": round(float(np.mean(scores[name]["roc"])), 4),
            "brier_mean": round(float(np.mean(scores[name]["brier"])), 4),
            "pr_auc_gain_vs_baseline": round(float(d.mean()), 4),
            "gain_se": round(se, 4),
            "statistically_better": bool(d.mean() > se and d.mean() > 0),
            "practically_meaningful": bool(d.mean() >= 0.02),
        }
    return out
