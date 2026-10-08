# ChurnGuard AI: Architecture (as built vs. proposed)

Legend: **BUILT** = runs in this repo today. **PROPOSED** = target Microsoft
deployment, not implemented. Nothing marked PROPOSED should be presented as working.

## 1. Pipeline (conceptual)

PREDICT -> EXPLAIN -> TIME-AWARE RISK -> PRIORITIZE -> (INTERVENE -> OPTIMIZE: simulated) -> MONITOR

| Stage | Status | Implementation |
|---|---|---|
| Predict | BUILT | LR / RF / XGBoost / LightGBM, 5x3 CV, 1-SE selection, sigmoid calibration |
| Explain | BUILT | SHAP per customer (associations, not causal) |
| Time-aware risk | BUILT (Telco only) | Cox PH on tenure/churn; conditional 3/6/12-month risk |
| Prioritize | BUILT | churn probability x CLV approximation; cost-based threshold analysis (assumed 10:1 cost) |
| Intervene / optimize | SIMULATION | Uplift benchmarks are assumed; no treatment/control data exists |
| Monitor | BUILT (proxy) | PSI/KS on two halves of the test set; no true timestamps |

## 2. As-built layout

```
frontend (React/Vite/Tailwind, 9 pages)  --HTTP-->  backend (FastAPI)
                                                     |-- SQLite: customers + predictions + churn_risk_{3,6,12}m
                                                     |-- artifacts/: metrics, SHAP, leakage audit, drift, survival
                                                     `-- ml/: preprocessing, evaluation (selection, ablation), models, monitoring
```

Retrain order: `python -m ml.models.train` -> `python -m ml.models.survival` -> `python -m ml.models.train_multi` -> `python -m ml.models.challenger`
(survival must run after train because it adds columns to the DB that train recreates; challenger reads each dataset's metrics.json).

## 3. Proposed Microsoft mapping (PROPOSED)

| Layer | As built | Microsoft target | Why change |
|---|---|---|---|
| Storage | CSV + SQLite | OneLake / Fabric Lakehouse | multi-user, governed, versioned data |
| Feature + training | scripts | Fabric notebooks / Spark | scale beyond one machine |
| Tracking + registry | JSON artifacts | MLflow in Fabric / Azure ML registry | versioning, lineage, rollback |
| Serving | FastAPI container | Azure Container Apps (or Azure ML managed endpoint) | scaling, auth |
| BI | React dashboard | Power BI (DirectLake on predictions table) | business-user self-service |
| Monitoring | PSI/KS script | scheduled Fabric pipeline + Power BI alerts | real timestamps required first |

Add a service only when its trigger is met: e.g. Spark only if data exceeds
single-node memory; Azure ML endpoints only if the model needs autoscaling.
Telco at 7K rows needs none of them. Say so if a judge asks.

## 4. Gate status (honest)

| Gate | Status |
|---|---|
| 1 Predict churn | PASSED: CV ROC-AUC ~0.84, PR-AUC ~0.66 (Telco) |
| 2 Explain | PASSED: SHAP, labeled as association |
| 3 Business impact | PARTIAL: revenue-at-risk is a proxy (probability x CLV approx.) |
| 4 Model time | PARTIAL: Cox on tenure snapshot; no real event sequences |
| 5 Intervention effect | NOT PASSED: no treatment data -> PROPOSED |
| 6 Economic optimization | SIMULATION only |
| 7 Monitor | PARTIAL: proxy cohorts only |

## 5. Data needed to close the gaps

- Gate 5/6: customer, intervention (offer/call/discount), timestamp, outcome, ideally randomized assignment.
- Gate 4: dated events (usage, billing, tickets, plan changes) per customer.
- Gate 7: scoring dates and delayed churn labels so performance decay can be measured.
- Forecasting: dated revenue series. Telco has none, so no forecast is claimed.

## 6. Multi-dataset serving (BUILT)

One registry (`app/services/registry.py`) drives generic `/api/datasets/{key}/...` routes and a header dropdown. Per-dataset capabilities (survival, counterfactual) are declared in the registry and enforced by the API.
