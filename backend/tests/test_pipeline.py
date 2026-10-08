"""
Real tests -- run against the actual cleaned dataset and trained
artifacts, not mocks. Run with `pytest` from the backend/ directory
after `python -m ml.models.train` has been run at least once.
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing.clean import load_raw, clean, data_quality_report
from ml.preprocessing.leakage_audit import run_leakage_audit
from ml.features.build_features import build_feature_frame, build_clv


@pytest.fixture(scope="module")
def cleaned_df():
    return clean(load_raw())


def test_dataset_has_expected_shape(cleaned_df):
    assert cleaned_df.shape[0] == 7043
    assert "churn_flag" in cleaned_df.columns


def test_no_missing_total_charges_after_cleaning(cleaned_df):
    assert cleaned_df["TotalCharges"].isna().sum() == 0


def test_leakage_audit_excludes_target_and_id(cleaned_df):
    audit = run_leakage_audit(cleaned_df)
    assert "customerID" in audit["excluded_structural"]
    assert "Churn" in audit["excluded_structural"]
    assert "churn_flag" not in audit["excluded_structural"] + audit["flagged_for_review"] + audit["cleared"]


def test_feature_frame_has_no_leaked_columns(cleaned_df):
    X = build_feature_frame(cleaned_df)
    assert "customerID" not in X.columns
    assert not any(c.startswith("Churn_") for c in X.columns)


def test_clv_is_positive_and_finite(cleaned_df):
    clv = build_clv(cleaned_df)
    assert (clv > 0).all()
    assert clv.notna().all()


def test_data_quality_report_shape(cleaned_df):
    raw = load_raw()
    report = data_quality_report(raw)
    assert report["rows"] == 7043
    assert report["columns"] == 21
    assert 0 <= report["imbalance_ratio_no_to_yes"]


class TestTrainedArtifacts:
    """These require the training pipeline to have already been run
    (`python -m ml.models.train`). Skipped automatically if artifacts
    are missing, so a fresh checkout's `pytest` run doesn't fail before
    setup -- see README for the required `make train` step."""

    ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

    def _load(self, name):
        path = os.path.join(self.ARTIFACT_DIR, name)
        if not os.path.exists(path):
            pytest.skip(f"{name} not found -- run `python -m ml.models.train` first")
        return json.load(open(path))

    def test_metrics_report_all_four_models(self):
        metrics = self._load("metrics.json")
        assert set(metrics["models"].keys()) == {
            "logistic_regression", "random_forest", "xgboost", "lightgbm"
        }

    def test_production_model_has_reasonable_roc_auc(self):
        metrics = self._load("metrics.json")
        prod = metrics["production_model"]
        roc_auc = metrics["models"][prod]["test"]["roc_auc"]
        # sanity bound, not a hardcoded expected value -- catches training
        # regressions (e.g. accidental leakage inflating AUC near 1.0, or a
        # broken pipeline collapsing AUC near 0.5)
        assert 0.7 < roc_auc < 0.97

    def test_calibration_ece_is_reasonable(self):
        cal = self._load("calibrated_test_metrics.json")
        assert 0 <= cal["ece"] < 0.2

    def test_drift_report_has_stable_or_flagged_status(self):
        drift = self._load("drift_report.json")
        for f in drift["feature_drift"]:
            assert f["status"] in ("stable", "warning", "significant_drift")


class TestMultiDatasetArtifacts:
    """Covers the Bank, Iranian, and Retail datasets added via
    `python -m ml.models.train_multi`. Skipped if not yet trained."""

    ARTIFACT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

    def _load(self, dataset_key, name):
        path = os.path.join(self.ARTIFACT_ROOT, dataset_key, name)
        if not os.path.exists(path):
            pytest.skip(f"{dataset_key}/{name} not found -- run `python -m ml.models.train_multi` first")
        return json.load(open(path))

    @pytest.mark.parametrize("dataset_key", ["bank", "iranian", "retail"])
    def test_metrics_report_all_four_models(self, dataset_key):
        metrics = self._load(dataset_key, "metrics.json")
        assert set(metrics["models"].keys()) == {
            "logistic_regression", "random_forest", "xgboost", "lightgbm"
        }
        assert metrics["n_total"] > 0

    @pytest.mark.parametrize("dataset_key", ["bank", "iranian", "retail"])
    def test_production_model_roc_auc_in_sane_range(self, dataset_key):
        metrics = self._load(dataset_key, "metrics.json")
        prod = metrics["production_model"]
        roc_auc = metrics["models"][prod]["test"]["roc_auc"]
        # Iranian genuinely scores very high (see README) -- bound is wide but still
        # catches a broken pipeline (e.g. AUC collapsing to ~0.5) or outright leakage (AUC == 1.0).
        assert 0.6 < roc_auc < 1.0

    @pytest.mark.parametrize("dataset_key", ["bank", "iranian", "retail"])
    def test_leakage_audit_excludes_raw_target(self, dataset_key):
        audit = self._load(dataset_key, "leakage_audit.json")
        assert audit["target_column"] == "churn_flag"
        assert "churn_flag" not in audit["flagged_for_review"]

    def test_retail_has_no_provided_label_disclosed_via_derivation(self):
        # Retail's "label" is derived, not provided -- sanity check the
        # positive rate is in a plausible range for a 3-month RFM cutoff.
        metrics = self._load("retail", "metrics.json")
        assert 0.1 < metrics["positive_rate"] < 0.9


# ---------------------------------------------------------------------------
# v2 tests: selection, calibration, threshold analysis, survival, ablation
# ---------------------------------------------------------------------------
import sqlite3
import numpy as np
from ml.evaluation.selection import select_one_se, threshold_analysis

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ART = os.path.join(_BACKEND, "artifacts")


def _art(name):
    path = os.path.join(_ART, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} missing -- run training first")
    return json.load(open(path))


def test_select_one_se_prefers_simpler_model_within_noise():
    cv = {
        "logistic_regression": {"pr_auc_mean": 0.657, "pr_auc_se": 0.010},
        "random_forest": {"pr_auc_mean": 0.668, "pr_auc_se": 0.009},
        "xgboost": {"pr_auc_mean": 0.667, "pr_auc_se": 0.008},
        "lightgbm": {"pr_auc_mean": 0.600, "pr_auc_se": 0.008},
    }
    name, _ = select_one_se(cv)
    assert name in ("logistic_regression", "xgboost")  # never the complex RF/the weak LGBM
    cv["logistic_regression"]["pr_auc_mean"] = 0.60  # clearly worse -> not chosen
    assert select_one_se(cv)[0] == "xgboost"


def test_threshold_analysis_returns_true_cost_minimum():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    res = threshold_analysis(y, p, fn_cost=10, fp_cost=1)
    t = res["best_threshold"]["threshold"]
    for thr in np.linspace(0.01, 0.99, 99):
        pred = p >= thr
        cost = 10 * ((~pred) & (y == 1)).sum() + 1 * (pred & (y == 0)).sum()
        assert res["best_threshold"]["cost"] <= cost + 1e-9
    assert 0 < t < 1


def test_sigmoid_calibration_preserves_ranking_and_improves_brier():
    cmp_ = _art("calibrated_test_metrics.json")["calibration_comparison"]
    assert cmp_["production_method"] == "sigmoid"
    assert cmp_["sigmoid"]["roc_auc"] == cmp_["raw"]["roc_auc"]
    assert cmp_["sigmoid"]["pr_auc"] == cmp_["raw"]["pr_auc"]
    assert cmp_["sigmoid"]["brier_score"] < cmp_["raw"]["brier_score"]


def test_every_model_has_cv_results_with_multiple_folds():
    m = _art("metrics.json")
    for name, v in m["models"].items():
        assert v["cv"]["n_folds"] >= 10, name
        assert v["cv"]["pr_auc_std"] >= 0


def test_survival_excludes_time_axis_covariates():
    surv = _art("survival_population.json")
    assert "tenure" in surv["covariates_excluded"] and "TotalCharges" in surv["covariates_excluded"]
    import joblib
    cph = joblib.load(os.path.join(_ART, "models", "cox_survival.joblib"))
    cols = set(cph.params_.index)
    assert "tenure" not in cols and "TotalCharges" not in cols


def test_survival_c_index_is_not_inflated_by_tenure_leak():
    surv = _art("survival_population.json")
    assert 0.5 < surv["c_index_test"] < 0.92


def test_conditional_churn_risk_is_valid_and_monotone_in_horizon():
    db = os.path.join(_BACKEND, "data", "churnguard.db")
    if not os.path.exists(db):
        pytest.skip("db missing")
    conn = sqlite3.connect(db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(customers)").fetchall()]
    assert "estimated_days_to_churn" not in cols  # fabricated heuristic removed
    if "churn_risk_6m" not in cols:
        pytest.skip("run `python -m ml.models.survival`")
    bad = conn.execute(
        "SELECT COUNT(*) FROM customers WHERE churn_risk_3m < 0 OR churn_risk_12m > 1 "
        "OR churn_risk_3m > churn_risk_6m + 1e-6 OR churn_risk_6m > churn_risk_12m + 1e-6"
    ).fetchone()[0]
    assert bad == 0


def test_iranian_ablation_artifact_documents_signal_breadth():
    ab = _art(os.path.join("iranian", "ablation.json"))
    assert "all_features" in ab and "without_complains_status" in ab
    assert ab["without_complains_status"]["roc_auc_mean"] > 0.9


# ---------------------------------------------------------------------------
# v3 tests: dataset-scoped API, capability gating, challenger gate
# ---------------------------------------------------------------------------
from ml.evaluation.selection import paired_compare

DATASET_KEYS = ["telco", "bank", "iranian", "retail"]


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.mark.parametrize("key", DATASET_KEYS)
def test_every_dataset_serves_all_core_endpoints(client, key):
    for ep in ("schema", "summary", "customers?page_size=3", "models", "calibration", "drift",
               "leakage-audit", "data-quality", "experiments"):
        r = client.get(f"/api/datasets/{key}/{ep}")
        assert r.status_code == 200, (key, ep, r.text[:200])


@pytest.mark.parametrize("key", DATASET_KEYS)
def test_summary_is_internally_consistent(client, key):
    s = client.get(f"/api/datasets/{key}/summary").json()
    assert sum(r["n"] for r in s["risk_distribution"]) == s["total_customers"]
    assert 0 <= s["predicted_churn_rate"] <= 1
    assert s["high_risk_customers"] <= s["total_customers"]
    assert s["expected_loss_proxy"] <= s["total_portfolio_clv"] + 1e-6


@pytest.mark.parametrize("key", DATASET_KEYS)
def test_customer_probabilities_valid_and_detail_roundtrips(client, key):
    rows = client.get(f"/api/datasets/{key}/customers?page_size=5").json()["customers"]
    assert rows and all(0 <= r["churn_probability"] <= 1 for r in rows)
    cid = rows[0]["customer_id"]
    d = client.get(f"/api/datasets/{key}/customers/{cid}").json()
    assert str(d["customer_id"]) == str(cid) and d["attributes"]
    assert client.get(f"/api/datasets/{key}/customers/{cid}/explanation").status_code == 200


def test_unsupported_capabilities_are_refused_not_faked(client):
    cid = client.get("/api/datasets/bank/customers?page_size=1").json()["customers"][0]["customer_id"]
    assert client.get(f"/api/datasets/bank/customers/{cid}/survival").status_code == 404
    assert client.get(f"/api/datasets/bank/customers/{cid}/counterfactual").status_code == 404
    assert client.get("/api/datasets/bank/survival/population").status_code == 404
    tid = client.get("/api/datasets/telco/customers?page_size=1").json()["customers"][0]["customer_id"]
    assert client.get(f"/api/datasets/telco/customers/{tid}/survival").status_code == 200


def test_unknown_dataset_404_and_bad_sort_is_safe(client):
    assert client.get("/api/datasets/nope/summary").status_code == 404
    r = client.get("/api/datasets/bank/customers?sort_by=churn_probability;DROP TABLE x")
    assert r.status_code == 200 and r.json()["total"] == 10000


def test_cost_scaling_changes_costs_only_by_scale():
    from ml.explainability.retention_optimizer import evaluate_actions
    base = {a["action"]: a["intervention_cost"] for a in evaluate_actions(0.5, 1000)}
    scaled = {a["action"]: a["intervention_cost"] for a in evaluate_actions(0.5, 1000, cost_scale=10)}
    assert all(abs(scaled[k] - 10 * base[k]) < 0.01 for k in base)


def test_paired_compare_flags_identical_model_as_not_better():
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(n_samples=600, n_features=8, random_state=0)
    X, y = pd.DataFrame(X), pd.Series(y)
    res = paired_compare(LogisticRegression(max_iter=500), {"same": LogisticRegression(max_iter=500)}, X, y, n_repeats=1)
    assert res["same"]["pr_auc_gain_vs_baseline"] == 0 and res["same"]["statistically_better"] is False


@pytest.mark.parametrize("key", DATASET_KEYS)
def test_challenger_artifact_follows_its_own_rule(key):
    art = os.path.join(_ART, "challenger.json" if key == "telco" else os.path.join(key, "challenger.json"))
    if not os.path.exists(art):
        pytest.skip("run python -m ml.models.challenger")
    c = json.load(open(art))
    for name, v in c["cv"].items():
        if name == "baseline":
            continue
        should = v["pr_auc_gain_vs_baseline"] > v["gain_se"] and v["pr_auc_gain_vs_baseline"] > 0
        assert v["statistically_better"] == should
        assert (name in c["promoted"]) == (should and v["pr_auc_gain_vs_baseline"] >= 0.02)
