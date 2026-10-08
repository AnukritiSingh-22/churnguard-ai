"""Offline explanation layer.

This intentionally does not call a hosted model. It turns already-computed SHAP
associations into a deterministic narrative, keeping customer data local. A
provider can be plugged in behind this interface after privacy review.
"""
from __future__ import annotations


def explain_from_shap(customer_id: str, probability: float, top_features: list[dict]) -> dict:
    drivers = [f"{x.get('feature')}: {x.get('direction', 'associated')}" for x in top_features[:3]]
    return {
        "customer_id": customer_id,
        "mode": "offline-grounded-template",
        "text": f"Estimated churn risk is {probability:.0%}. The strongest associated signals are "
                 + (", ".join(drivers) if drivers else "not available") + ".",
        "disclaimer": "Association is not causation. This narrative is grounded only in model attributions; it is not an LLM claim.",
    }
