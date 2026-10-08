# ChurnGuard AI: Pitch & Judge Q&A

## 60-second pitch
Telcos know who left last month. We predict who is at risk next, explain why,
estimate how soon, and rank who to call first, and we show exactly which parts
are measured and which are simulated. Prediction is not prevention; we are
honest that proving prevention needs intervention data.

## Demo flow (4 minutes)
0. **Dataset dropdown:** switch Telco -> Bank -> Retail live; every page follows. Point out Survival shows 'not available' on datasets without time-to-event data instead of faking it.
1. **Overview:** high-risk count, revenue-at-risk *proxy* (say "proxy" out loud).
2. **Risk Explorer:** sort by 6-month churn risk (survival model); open one customer.
3. **Customer 360:** probability, SHAP drivers ("the model associated..."), survival curve and conditional 3/12-month risk.
4. **Model Performance:** CV table; point out the four models are statistically tied and we chose by the 1-SE rule.
5. **Datasets / Data Quality:** leakage audit and the Iranian ablation.
6. **Drift:** say plainly it uses proxy cohorts.

## Numbers you can quote (Telco, from artifacts)
- CV ROC-AUC ~0.84-0.85, CV PR-AUC ~0.66-0.67 across models; test ROC-AUC 0.82 (XGBoost).
- Calibration: Brier 0.166 -> 0.146, ECE 0.118 -> 0.034, ranking unchanged.
- Cox C-index ~0.85 (after fixing a circularity; do not quote 0.93).

## Likely judge questions
**"Dynamics 365 / Salesforce already do churn prediction."** Agreed. We reproduce the
prediction layer; our contribution is combining explanation, survival-based timing, calibrated
uncertainty, cost-aware thresholds, and monitoring in one honest pipeline. Where we are weaker:
no real intervention data, no production integration.

**"Why not deep learning / Transformer / GNN?"** Telco has one row per customer: no
sequences, no graph. Models that need that data would be fabricated. We would add them once
dated events exist, and only if they beat the baseline.

**"Is 0.99 on the Iranian data real?"** Not a single-column leak (ablation: 0.935 even
without complaints, status and usage). But it detects imminent churn, so it is not comparable to Telco.

**"Is the intervention ROI real?"** No, it is simulation with assumed uplift. We label it as such.

**"Which model is best?"** They are statistically tied; we pick the simplest within one standard error.

**"What did you fix?"** A made-up days-to-churn formula (removed), a survival model that used
tenure on both sides (fixed, C-index 0.93 -> 0.85), isotonic calibration that hurt ranking
(switched to sigmoid), and single-split model selection (now repeated CV).

**"Why no Transformer / GNN / deep model?"** We built a gate: stacking and voting ensembles
vs the production model, paired CV, promote only if gain > 1 SE and >= 0.02 PR-AUC. Result: Telco +0.009
(marginal, not promoted), no gain on Bank, Iranian or Retail. Sequence and graph models need data
these datasets do not contain.

**"You changed the promotion rule after seeing results."** Yes: the 0.02 floor was added after the first
Telco run, and it is disclosed in challenger.json and the README. Without it, a +0.009 gain would count as a win.
