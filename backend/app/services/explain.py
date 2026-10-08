import json
from app.core.config import ARTIFACT_DIR

_shap_cache = None
_survival_cache = None


def _load_shap():
    global _shap_cache
    if _shap_cache is None:
        path = ARTIFACT_DIR / "shap_values.json"
        _shap_cache = json.load(open(path)) if path.exists() else {}
    return _shap_cache


def _load_survival_per_customer():
    global _survival_cache
    if _survival_cache is None:
        path = ARTIFACT_DIR / "survival_per_customer.json"
        _survival_cache = json.load(open(path)) if path.exists() else {}
    return _survival_cache


def get_explanation(customer_id: str):
    shap_data = _load_shap()
    contributions = shap_data.get(customer_id)
    if contributions is None:
        return None
    positive = [c for c in contributions if c["contribution"] > 0]
    negative = [c for c in contributions if c["contribution"] < 0]
    return {
        "customer_id": customer_id,
        "method": "SHAP feature contribution (from the production model)",
        "disclaimer": "These are model feature contributions to THIS PREDICTION, "
                       "not proof that a feature causes churn.",
        "risk_increasing_factors": positive,
        "risk_decreasing_factors": negative,
    }


def get_survival(customer_id: str):
    data = _load_survival_per_customer()
    return data.get(customer_id)
