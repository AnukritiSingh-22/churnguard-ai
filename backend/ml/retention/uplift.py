"""Treatment-effect boundary and simulation fallback."""
from __future__ import annotations


def score_contact_budget(customers: list[dict], budget: int) -> dict:
    ranked = sorted(customers, key=lambda x: float(x.get("expected_value", 0)), reverse=True)
    return {
        "mode": "simulation",
        "budget": max(0, int(budget)),
        "selected": ranked[:max(0, int(budget))],
        "note": "No randomized treatment/control data is present. This is expected-value ranking, not a validated uplift model.",
        "validated_uplift_model": False,
    }
