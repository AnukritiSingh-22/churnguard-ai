"""
Retention Decision Engine.

IMPORTANT HONESTY NOTE (surfaced in the API and UI, not just here):
The Telco dataset contains NO record of any past retention intervention
(no treatment/control groups, no A/B test flags, no "offer sent" column).
Therefore this module CANNOT estimate a real causal treatment effect --
doing so would be fabricating an uplift number.

What it does instead, labeled everywhere as "Simulation mode":
  - Uses the model's REAL predicted churn probability and REAL computed
    CLV for the customer.
  - Applies a configurable, clearly-documented ASSUMED uplift percentage
    per action type (industry-rule-of-thumb ranges, not measured from
    this data) to produce an illustrative expected-value ranking.
  - Every response is tagged `"mode": "simulation"` and
    `"causal_claim": false` so no downstream consumer can present this
    as a proven causal estimate.

If a real experiment log (treatment/control with outcomes) becomes
available, replace ASSUMED_UPLIFT with a trained uplift model (e.g. a
two-model T-learner or causal forest) and flip `mode` to "estimated_from_experiment".
"""
from __future__ import annotations

# Assumed retention-uplift ranges per action, expressed as the ABSOLUTE
# reduction in churn probability an intervention might plausibly achieve.
# These are illustrative, commonly-cited SaaS/telecom retention
# benchmarks -- NOT measured from this dataset. Source: documented
# assumption, adjustable via config.
ACTIONS = {
    "no_action": {"uplift": 0.0, "cost": 0.0, "label": "No Action"},
    "discount_10pct": {"uplift": 0.09, "cost": 300.0, "label": "Discount (10%)"},
    "plan_upgrade_bundle": {"uplift": 0.12, "cost": 600.0, "label": "Plan Upgrade / Bundle"},
    "personal_outreach": {"uplift": 0.15, "cost": 250.0, "label": "Personal Outreach Call"},
    "service_recovery": {"uplift": 0.18, "cost": 400.0, "label": "Service Recovery (support fix)"},
    "engagement_campaign": {"uplift": 0.07, "cost": 120.0, "label": "Engagement Campaign"},
    "retention_offer": {"uplift": 0.20, "cost": 1200.0, "label": "Personalized Retention Offer"},
}


def evaluate_actions(churn_probability: float, clv: float, cost_scale: float = 1.0) -> list[dict]:
    """cost_scale rescales the ASSUMED per-action costs for datasets whose CLV differs from Telco's."""
    results = []
    for key, cfg in ACTIONS.items():
        new_prob = max(churn_probability - cfg["uplift"], 0.0)
        retained_value = (churn_probability - new_prob) * clv
        cost = cfg["cost"] * cost_scale
        net_value = retained_value - cost
        confidence = "N/A" if key == "no_action" else (
            "Medium" if cfg["uplift"] <= 0.12 else "Low-Medium"
        )
        results.append({
            "action": key,
            "label": cfg["label"],
            "assumed_retention_uplift_pct": round(cfg["uplift"] * 100, 1),
            "intervention_cost": round(cost, 2),
            "baseline_churn_probability": round(churn_probability, 4),
            "projected_churn_probability": round(new_prob, 4),
            "expected_clv_preserved": round(retained_value, 2),
            "expected_net_value": round(net_value, 2),
            "confidence": confidence,
            "mode": "simulation",
            "causal_claim": False,
        })
    results.sort(key=lambda r: r["expected_net_value"], reverse=True)
    return results
