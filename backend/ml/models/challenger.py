from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ml.models.train_generic import get_model_zoo
from ml.preprocessing.split import stratified_split
from ml.evaluation.metrics import compute_all_metrics
from ml.evaluation.selection import paired_compare

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")


def load_xy(key: str):
    if key == "telco":
        from ml.preprocessing.clean import load_raw, clean
        from ml.features.build_features import build_feature_frame
        df = clean(load_raw())
        return build_feature_frame(df), df["churn_flag"], df["customerID"]
    if key == "bank":
        from ml.datasets import bank as m
        df = m.load_and_clean()
        return m.build_feature_frame(df), df["churn_flag"], df["customer_id"]
    if key == "iranian":
        from ml.datasets import iranian as m
        df = m.load_and_clean()
        return m.build_feature_frame(df), df["churn_flag"], df["customer_id"]
    if key == "retail":
        from ml.datasets import retail as m
        agg = m.build_customer_frame(m.load_and_clean())
        return m.build_feature_frame(agg), agg["churn_flag"], agg["customer_id"]
    raise ValueError(key)


def challengers(small: bool):
    n = 150 if small else 300
    base = [
        ("lr", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=n, max_depth=6 if small else 8, min_samples_leaf=5,
                                      class_weight="balanced", random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=n, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)),
        ("lgbm", LGBMClassifier(n_estimators=n, max_depth=5, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, class_weight="balanced", random_state=42,
                                n_jobs=-1, verbose=-1)),
    ]
    return {
        "stacking_ensemble": StackingClassifier(estimators=base[:3], final_estimator=LogisticRegression(max_iter=2000),
                                                stack_method="predict_proba", cv=3, n_jobs=1),
        "soft_voting_ensemble": VotingClassifier(estimators=base, voting="soft", n_jobs=1),
    }


def run(key: str) -> dict:
    d = ART if key == "telco" else os.path.join(ART, key)
    metrics = json.load(open(os.path.join(d, "metrics.json")))
    prod = metrics["production_model"]
    X, y, ids = load_xy(key)
    sp = stratified_split(X, y, ids)
    Xtr, ytr, _ = sp["train"]
    Xte, yte, _ = sp["test"]
    base_model = get_model_zoo(small_dataset=(key == "iranian"))[prod]
    ch = challengers(key == "iranian")
    cv = paired_compare(base_model, ch, Xtr, ytr)
    test = {}
    for name, model in {"baseline": base_model, **ch}.items():
        m = model.fit(Xtr, ytr)
        t = compute_all_metrics(yte.values, m.predict_proba(Xte)[:, 1])
        test[name] = {k: t[k] for k in ("roc_auc", "pr_auc", "brier_score")}
    better = [n for n in ch if cv[n]["statistically_better"]]
    promoted = [n for n in better if cv[n]["practically_meaningful"]]
    if promoted:
        decision = f"PROMOTE candidate: {', '.join(promoted)} (gain > 1 SE and >= 0.02 PR-AUC)."
    elif better:
        decision = (f"MARGINAL: {', '.join(better)} beat the baseline by more than 1 SE but by < 0.02 PR-AUC. "
                    "Not promoted: the gain is too small to justify losing simple SHAP explanations and adding latency.")
    else:
        decision = "NOT PROMOTED: no challenger beat the baseline by more than 1 SE. Extra complexity is not justified."
    res = {"dataset": key, "baseline_model": prod, "cv": cv, "test": test,
           "statistically_better": better, "promoted": promoted, "decision": decision,
           "rule": "Paired 5x3 CV on train split. Promote only if gain > 1 SE of paired differences AND >= 0.02 PR-AUC "
                   "(0.02 floor added after seeing the Telco result; disclosed)."}
    json.dump(res, open(os.path.join(d, "challenger.json"), "w"), indent=2)
    return res


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["telco", "bank", "iranian", "retail"]):
        r = run(k)
        print(k, r["decision"])
