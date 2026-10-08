"""Local-first experiment logging with an optional MLflow adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def log_run(artifact_dir: str | Path, run_name: str = "churnguard") -> dict[str, Any]:
    root = Path(artifact_dir)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    payload = {
        "run_name": run_name,
        "dataset": metrics.get("dataset", "telco"),
        "production_model": metrics.get("production_model"),
        "metrics_artifact": str(metrics_path),
        "tracking": "local-json",
        "mlflow_logged": False,
    }
    try:
        import mlflow
    except ImportError:
        pass
    else:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("production_model", str(metrics.get("production_model", "")))
            production = metrics.get("models", {}).get(metrics.get("production_model"), {})
            for key in ("roc_auc", "pr_auc", "brier_score", "ece"):
                value = production.get("test", {}).get(key)
                if value is not None:
                    mlflow.log_metric(key, float(value))
            mlflow.log_artifact(str(metrics_path))
        payload["tracking"] = "mlflow"
        payload["mlflow_logged"] = True
    (root / "run_log.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    from app.core.config import ARTIFACT_DIR
    print(log_run(ARTIFACT_DIR))
