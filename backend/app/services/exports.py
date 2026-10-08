from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from app.core.config import ARTIFACT_DIR
from app.core.db import get_connection


def predictions_csv() -> str:
    conn = get_connection()
    rows = conn.execute(
        "SELECT customerID, churn_probability, risk_level, churn_flag, MonthlyCharges, CLV, "
        "primary_risk_driver, production_model FROM customers ORDER BY churn_probability DESC"
    ).fetchall()
    columns = [d[0] for d in conn.execute("SELECT customerID, churn_probability, risk_level, churn_flag, MonthlyCharges, CLV, primary_risk_driver, production_model FROM customers LIMIT 1").description]
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerows(rows)
    return output.getvalue()


def powerbi_manifest() -> dict:
    return {
        "tables": {
            "predictions": "predictions.csv",
            "model_metrics": "model_metrics.csv",
            "drift": "drift.csv",
            "revenue_forecast": "revenue_forecast.csv",
        },
        "refresh_note": "Import these files into Power BI Desktop. They are generated from local artifacts; no Power BI tenant is required.",
        "last_generated": None,
    }
