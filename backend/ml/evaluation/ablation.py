"""Feature-group ablation: how much of the AUC survives when suspicious
features are removed? Large survival => signal is broad, not one leaky column.
Reported as 5-fold CV ROC-AUC (mean +/- std) with LightGBM."""
from __future__ import annotations
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def run_ablation(X, y, groups: dict[str, list[str]], seed: int = 42) -> dict:
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def score(cols):
        m = LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, verbose=-1, random_state=seed)
        s = cross_val_score(m, X[cols], y, cv=cv, scoring="roc_auc")
        return {"roc_auc_mean": round(float(s.mean()), 4), "roc_auc_std": round(float(s.std()), 4), "n_features": len(cols)}

    allc = list(X.columns)
    out = {"all_features": score(allc)}
    for label, drop in groups.items():
        drop_cols = [c for c in allc if any(c == d or c.startswith(d + "_") for d in drop)]
        out[f"without_{label}"] = {**score([c for c in allc if c not in drop_cols]), "dropped": drop_cols}
    return out
