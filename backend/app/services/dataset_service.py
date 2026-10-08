from __future__ import annotations
import json
import functools
import pandas as pd
from fastapi import HTTPException
from app.core.db import get_connection
from app.services import registry
from ml.explainability.retention_optimizer import evaluate_actions

HIGH = "('High','Critical')"


def cfg(key: str) -> dict:
    try:
        return registry.get(key)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset '{key}'. Valid: {list(registry.DATASETS)}")


def _json(key: str, name: str):
    p = cfg(key)["artifact_dir"] / name
    return json.load(open(p)) if p.exists() else None


@functools.lru_cache(maxsize=8)
def _shap(key: str) -> dict:
    return _json(key, "shap_values.json") or {}


def table_columns(key: str) -> list[str]:
    conn = get_connection()
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({cfg(key)['table']})").fetchall()]
    conn.close()
    return cols


def top_driver(key: str, cid, row: dict | None = None):
    if row and row.get("primary_risk_driver"):
        return row["primary_risk_driver"]
    contribs = _shap(key).get(str(cid))
    if not contribs:
        return None
    pos = [c for c in contribs if c["contribution"] > 0]
    return pos[0]["feature"] if pos else None


def schema(key: str) -> dict:
    c = cfg(key)
    conn = get_connection()
    seg = conn.execute(f"SELECT {c['segment_col']} v, COUNT(*) n FROM {c['table']} GROUP BY 1 ORDER BY n DESC LIMIT 30").fetchall()
    n = conn.execute(f"SELECT COUNT(*) n FROM {c['table']}").fetchone()["n"]
    conn.close()
    return {"key": key, "name": c["name"], "domain": c["domain"], "currency": c["currency"], "rows": n,
            "id_col": c["id_col"], "segment_col": c["segment_col"], "segment_label": c["segment_label"],
            "segment_values": [str(r["v"]) for r in seg],
            "columns": [{"key": k, "label": l, "type": t} for k, l, t in c["columns"]],
            "capabilities": c["capabilities"], "label_note": c["label_note"]}


def summary(key: str) -> dict:
    c = cfg(key)
    t, seg = c["table"], c["segment_col"]
    conn = get_connection()
    q = lambda sql: conn.execute(sql).fetchall()
    total = q(f"SELECT COUNT(*) n FROM {t}")[0]["n"]
    agg = q(f"SELECT AVG(churn_probability) p, AVG(churn_flag) actual, SUM(CLV) clv, SUM(churn_probability*CLV) exp_loss FROM {t}")[0]
    high = q(f"SELECT COUNT(*) n, SUM(CLV) clv FROM {t} WHERE risk_level IN {HIGH}")[0]
    dist = q(f"SELECT risk_level, COUNT(*) n FROM {t} GROUP BY risk_level")
    by_seg = q(f"SELECT {seg} AS segment, COUNT(*) AS customers, AVG(churn_probability) AS avg_risk, "
               f"SUM(CASE WHEN risk_level IN {HIGH} THEN CLV ELSE 0 END) AS revenue_at_risk "
               f"FROM {t} GROUP BY {seg} ORDER BY customers DESC LIMIT 12")
    has_drv = "primary_risk_driver" in table_columns(key)
    top = [dict(r) for r in q(f"SELECT {c['id_col']} AS customer_id, churn_probability, risk_level, CLV"
                              f"{', primary_risk_driver' if has_drv else ''} FROM {t} "
                              f"WHERE risk_level IN {HIGH} ORDER BY CLV DESC LIMIT 10")]
    conn.close()
    for r in top:
        r["top_driver"] = top_driver(key, r["customer_id"], r)
    return {"dataset": key, "currency": c["currency"], "total_customers": total,
            "high_risk_customers": high["n"], "predicted_churn_rate": round(agg["p"] or 0, 4),
            "observed_churn_rate": round(agg["actual"] or 0, 4), "revenue_at_risk": round(high["clv"] or 0, 2),
            "expected_loss_proxy": round(agg["exp_loss"] or 0, 2), "total_portfolio_clv": round(agg["clv"] or 0, 2),
            "risk_distribution": [dict(r) for r in dist], "by_segment": [dict(r) for r in by_seg],
            "segment_label": c["segment_label"], "top_retention_opportunities": top, "label_note": c["label_note"]}


def list_customers(key, risk_level, segment, search, sort_by, sort_dir, page, page_size) -> dict:
    c = cfg(key)
    t, idc = c["table"], c["id_col"]
    if sort_by not in set(table_columns(key)):
        sort_by = "churn_probability"
    where, params = [], []
    if risk_level:
        lv = [x for x in risk_level.split(",") if x in ("Low", "Medium", "High", "Critical")]
        if lv:
            where.append(f"risk_level IN ({','.join('?' * len(lv))})"); params += lv
    if segment:
        where.append(f"CAST({c['segment_col']} AS TEXT) = ?"); params.append(segment)
    if search:
        where.append(f"CAST({idc} AS TEXT) LIKE ?"); params.append(f"%{search}%")
    w = f"WHERE {' AND '.join(where)}" if where else ""
    d = "DESC" if sort_dir.lower() == "desc" else "ASC"
    page_size = max(1, min(page_size, 100)); page = max(page, 1)
    conn = get_connection()
    total = conn.execute(f"SELECT COUNT(*) n FROM {t} {w}", params).fetchone()["n"]
    rows = conn.execute(f"SELECT * FROM {t} {w} ORDER BY {sort_by} {d} LIMIT ? OFFSET ?",
                        params + [page_size, (page - 1) * page_size]).fetchall()
    conn.close()
    out = []
    for r in rows:
        r = dict(r); r["customer_id"] = r[idc]; r["top_driver"] = top_driver(key, r[idc], r)
        out.append(r)
    return {"dataset": key, "total": total, "page": page, "page_size": page_size, "customers": out}


def get_customer(key: str, cid: str) -> dict:
    c = cfg(key)
    conn = get_connection()
    row = conn.execute(f"SELECT * FROM {c['table']} WHERE CAST({c['id_col']} AS TEXT) = ?", (cid,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, f"Customer {cid} not found in {key}")
    r = dict(row)
    r["customer_id"] = r[c["id_col"]]
    r["top_driver"] = top_driver(key, cid, r)
    r["attributes"] = {k: r.get(k) for k in c["detail_fields"] if k in r}
    r["currency"] = c["currency"]
    return r


def explanation(key: str, cid: str) -> dict:
    contribs = _shap(key).get(str(cid))
    if contribs is None:
        raise HTTPException(404, f"No SHAP explanation for {cid} in {key}")
    return {"customer_id": cid, "method": "SHAP feature contribution (production model)",
            "disclaimer": "The model associated these features with the prediction. This is not proof that a feature causes churn.",
            "risk_increasing_factors": [x for x in contribs if x["contribution"] > 0],
            "risk_decreasing_factors": [x for x in contribs if x["contribution"] < 0]}


@functools.lru_cache(maxsize=8)
def _mean_clv(key: str) -> float:
    conn = get_connection()
    v = conn.execute(f"SELECT AVG(CLV) m FROM {cfg(key)['table']}").fetchone()["m"]
    conn.close()
    return float(v or 1.0)


def recommendations(key: str, cid: str) -> dict:
    cu = get_customer(key, cid)
    scale = _mean_clv(key) / registry.TELCO_MEAN_CLV
    return {"customer_id": cid, "mode": "simulation", "currency": cfg(key)["currency"],
            "note": "Expected values use ASSUMED uplift benchmarks, not a causal model (no treatment/control data exists). "
                    f"Intervention costs are rescaled x{scale:.2f} to this dataset's mean CLV (also an assumption).",
            "actions": evaluate_actions(cu["churn_probability"], cu["CLV"], cost_scale=scale)}


def models(key: str) -> dict:
    m = _json(key, "metrics.json")
    if m is None:
        raise HTTPException(404, f"No trained models for {key}")
    keys = ("roc_auc", "pr_auc", "f1", "brier_score", "ece")
    rows = [{"model": n, "is_production": n == m["production_model"],
             "validation": {k: d["validation"][k] for k in keys}, "test": {k: d["test"][k] for k in keys},
             "cv": d.get("cv"), "train_time_seconds": d["train_time_seconds"],
             "inference_time_ms_per_sample": d["inference_time_ms_per_sample"]} for n, d in m["models"].items()]
    return {"dataset": key, "production_model": m["production_model"], "selection_reason": m["selection_reason"],
            "selection_method": m.get("selection_method"), "n_total": m.get("n_total"),
            "positive_rate": m.get("positive_rate"), "models": rows,
            "challenger": _json(key, "challenger.json"), "threshold_analysis": _json(key, "threshold_analysis.json")}


def _need(key, name, label):
    d = _json(key, name)
    if d is None:
        raise HTTPException(404, f"No {label} for {key}")
    return d


def calibration(key: str) -> dict:
    return _need(key, "calibrated_test_metrics.json", "calibration metrics")


def drift(key: str) -> dict:
    return _need(key, "drift_report.json", "drift report")


def leakage(key: str) -> dict:
    a = _need(key, "leakage_audit.json", "leakage audit")
    a["ablation"] = _json(key, "ablation.json")
    a.setdefault("details", [])
    return a


def data_quality(key: str) -> dict:
    art = _json(key, "data_quality.json")
    if art:
        art["source"] = "raw dataset (artifact)"
        return art
    c = cfg(key)
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {c['table']}", conn)
    conn.close()
    feat = df.drop(columns=[c["id_col"], "churn_probability", "risk_level", "production_model", "CLV"], errors="ignore")
    num = feat.drop(columns=["churn_flag"]).select_dtypes("number")
    out = {}
    for col in num.columns:
        q1, q3 = num[col].quantile([.25, .75]); iqr = q3 - q1
        out[col] = int(((num[col] < q1 - 1.5 * iqr) | (num[col] > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
    pos = int(df["churn_flag"].sum()); neg = int(len(df) - pos)
    return {"source": "computed from the scored customer table (cleaned data, not raw)", "rows": int(len(df)),
            "columns": int(feat.shape[1]), "duplicate_rows": int(feat.duplicated().sum()),
            "imbalance_ratio_no_to_yes": round(neg / max(pos, 1), 2),
            "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v > 0}, "outlier_counts_iqr": out}


def experiments(key: str) -> dict:
    m = _need(key, "metrics.json", "experiments")
    name = cfg(key)["name"]
    rows = [{"experiment_id": f"{key.upper()}-{i:03d}", "dataset": name, "model": mn,
             "split": "Stratified random 60/20/20; model chosen by 5x3 CV on train",
             "pr_auc": d["validation"]["pr_auc"], "roc_auc": d["validation"]["roc_auc"],
             "brier": d["validation"]["brier_score"], "cv_pr_auc": (d.get("cv") or {}).get("pr_auc_mean"),
             "status": "completed", "is_production": mn == m["production_model"]} for i, (mn, d) in enumerate(m["models"].items(), 1)]
    return {"dataset": key, "experiments": rows}
