import json
import functools
from app.core.config import ARTIFACT_DIR


def _load(name: str):
    path = ARTIFACT_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def get_metrics():
    return _load("metrics.json")


@functools.lru_cache(maxsize=1)
def get_calibrated_test_metrics():
    return _load("calibrated_test_metrics.json")


@functools.lru_cache(maxsize=1)
def get_leakage_audit():
    return _load("leakage_audit.json")


@functools.lru_cache(maxsize=1)
def get_data_quality():
    return _load("data_quality.json")


@functools.lru_cache(maxsize=1)
def get_drift_report():
    return _load("drift_report.json")


@functools.lru_cache(maxsize=1)
def get_shap_values():
    return _load("shap_values.json")


@functools.lru_cache(maxsize=1)
def get_feature_columns():
    return _load("feature_columns.json")


def get_production_model_name():
    m = get_metrics()
    return m["production_model"] if m else None
