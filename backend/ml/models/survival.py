"""
Time-to-churn / survival analysis.

The Telco dataset provides exactly what survival analysis needs:
  duration = tenure (months the customer has been active)
  event    = churn_flag (1 = churn event observed, 0 = customer is still
             active, i.e. right-censored at their current tenure)

This is a real, standard application of survival analysis to this
dataset (not simulated). We fit:
  1. A population Kaplan-Meier curve (unconditional survival function).
  2. A Cox Proportional Hazards model on the same leakage-cleared
     features used for classification, to get a per-customer survival
     curve and an honestly-computed concordance index (C-index).

IMPORTANT FIXES (v2):
  * tenure is the survival TIME AXIS, so it (and TotalCharges, which is
    ~ tenure x MonthlyCharges) are EXCLUDED from the Cox covariates.
    Including them made the previous C-index circular/inflated.
  * Per-customer output is now the CONDITIONAL probability of churning in
    the next 3/6/12 months given the customer has already survived to
    their current tenure: 1 - S(t+h|x)/S(t|x). This replaces the old
    heuristic "estimated_days_to_churn" (a hand-written formula).
  * Caveat: Telco has one snapshot per customer, so covariates (e.g.
    TechSupport) are measured at the snapshot, not at baseline. Treat
    results as associations, not causal hazard ratios.

Run with `python -m ml.models.survival` (from backend/), AFTER train.py.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
import sqlite3
import joblib
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.preprocessing.clean import load_raw, clean
from ml.preprocessing.split import stratified_split
from ml.features.build_features import build_feature_frame

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "churnguard.db")
HORIZONS = (3, 6, 12)
TIME_AXIS_COLS = ("tenure", "TotalCharges")  # excluded from covariates


def main():
    print("Training real survival model (Kaplan-Meier + Cox PH) on tenure/churn_flag...")
    raw = load_raw()
    df = clean(raw)
    X = build_feature_frame(df)
    y = df["churn_flag"]
    ids = df["customerID"]

    splits = stratified_split(X, y, ids)
    train_idx = splits["train"][0].index
    test_idx = splits["test"][0].index

    # --- Population Kaplan-Meier curve (whole dataset, real) ---
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["tenure"], event_observed=df["churn_flag"], label="Population")
    survival_function = kmf.survival_function_.reset_index()
    survival_function.columns = ["tenure_months", "survival_probability"]
    population_curve = survival_function.round(4).to_dict(orient="records")

    checkpoints_months = [1, 3, 6, 12, 24, 36]
    checkpoint_values = {}
    for m in checkpoints_months:
        try:
            checkpoint_values[m] = round(float(kmf.predict(m)), 4)
        except Exception:
            checkpoint_values[m] = None

    # --- Cox Proportional Hazards for per-customer curves ---
    cox_df = X.drop(columns=[c for c in TIME_AXIS_COLS if c in X.columns]).copy()
    # drop near-constant / collinear columns that break Cox fitting
    cox_df = cox_df.loc[:, cox_df.std() > 1e-6]
    cox_df["duration"] = df["tenure"].values
    cox_df["event"] = df["churn_flag"].values

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df.loc[train_idx.union(splits["val"][0].index)], duration_col="duration", event_col="event")

    test_df = cox_df.loc[test_idx]
    test_partial_hazard = cph.predict_partial_hazard(test_df)
    c_index = concordance_index(test_df["duration"], -test_partial_hazard, test_df["event"])
    print(f"Cox PH concordance index (test, real): {round(c_index, 4)}")

    # Individual survival function for every customer (from the fitted Cox model)
    all_df = cox_df.drop(columns=["duration", "event"])
    max_t = int(df["tenure"].max()) + max(HORIZONS)
    grid = list(range(0, max_t + 1))
    S = cph.predict_survival_function(all_df, times=grid)  # rows=times, cols=customers
    S_arr = S.values  # (len(grid), n_customers)

    tenures = df["tenure"].astype(int).values
    cond_risk = {h: np.zeros(len(df)) for h in HORIZONS}
    for i, t in enumerate(tenures):
        s_t = max(S_arr[t, i], 1e-9)
        for h in HORIZONS:
            cond_risk[h][i] = float(np.clip(1.0 - S_arr[min(t + h, max_t), i] / s_t, 0.0, 1.0))

    per_customer = {}
    for i, cust_id in enumerate(ids.values):
        per_customer[cust_id] = {
            "curve": [{"month": int(t), "survival_probability": round(float(S_arr[t, i]), 4)}
                      for t in (1, 3, 6, 12, 18, 24, 36, 48) if t < len(grid)],
            "conditional_churn_risk": {f"{h}m": round(float(cond_risk[h][i]), 4) for h in HORIZONS},
        }

    # Sanity check, reported honestly: does the conditional hazard rank
    # eventual churners above non-churners on held-out customers?
    from sklearn.metrics import roc_auc_score
    test_pos = df.index.get_indexer(test_idx)
    surv_auc_6m = roc_auc_score(df["churn_flag"].values[test_pos], cond_risk[6][test_pos])

    # Write the 6/12-month risks to SQLite so the API/UI can show them
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(customers)").fetchall()]
        for h in HORIZONS:
            if f"churn_risk_{h}m" not in cols:
                conn.execute(f"ALTER TABLE customers ADD COLUMN churn_risk_{h}m REAL")
        conn.executemany(
            "UPDATE customers SET churn_risk_3m=?, churn_risk_6m=?, churn_risk_12m=? WHERE customerID=?",
            [(round(float(cond_risk[3][i]), 4), round(float(cond_risk[6][i]), 4),
              round(float(cond_risk[12][i]), 4), cid) for i, cid in enumerate(ids.values)],
        )
        conn.commit()
        conn.close()
        print("Wrote churn_risk_3m/6m/12m to SQLite customers table.")

    joblib.dump(cph, os.path.join(MODEL_DIR, "cox_survival.joblib"))
    joblib.dump(kmf, os.path.join(MODEL_DIR, "kaplan_meier.joblib"))

    with open(os.path.join(ARTIFACT_DIR, "survival_population.json"), "w") as f:
        json.dump({
            "population_curve": population_curve,
            "checkpoint_survival": checkpoint_values,
            "c_index_test": round(float(c_index), 4),
            "conditional_6m_risk_auc_test": round(float(surv_auc_6m), 4),
            "covariates_excluded": list(TIME_AXIS_COLS),
            "caveat": "Single snapshot per customer: covariates are not baseline values. "
                      "Associations only; not causal hazard ratios.",
            "method": "Kaplan-Meier (population) + Cox Proportional Hazards (per-customer), "
                      "fit on real tenure (duration) and churn_flag (event) columns.",
        }, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "survival_per_customer.json"), "w") as f:
        json.dump(per_customer, f, indent=2)

    print(f"Survival artifacts written. Population median survival: "
          f"{kmf.median_survival_time_} months.")


if __name__ == "__main__":
    main()
