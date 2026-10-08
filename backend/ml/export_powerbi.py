"""Generate flat files for Power BI Desktop or Fabric Data Factory ingestion."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from app.services.exports import predictions_csv
from app.core.config import ARTIFACT_DIR


def _rows(path: Path, keys: list[str]) -> list[dict]:
    data = json.loads(path.read_text()) if path.exists() else {}
    if path.name == "metrics.json":
        return [{"model": name, **v["test"]} for name, v in data.get("models", {}).items()]
    return [{k: item.get(k) for k in keys} for item in data.get("feature_drift", [])]


def export(output_dir: str | Path = ARTIFACT_DIR / "powerbi") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.csv").write_text(predictions_csv())
    for name, path, keys in [
        ("model_metrics.csv", ARTIFACT_DIR / "metrics.json", ["model"]),
        ("drift.csv", ARTIFACT_DIR / "drift_report.json", ["feature", "psi", "ks", "status"]),
    ]:
        rows = _rows(path, keys)
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=(list(rows[0]) if rows else keys))
            writer.writeheader()
            writer.writerows(rows)
    forecast = ARTIFACT_DIR / "revenue_forecast.json"
    if forecast.exists():
        data = json.loads(forecast.read_text())
        with (out / "revenue_forecast.csv").open("w", newline="") as f:
            rows = data.get("forecast", [])
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["period", "prediction", "lower", "upper"])
            writer.writeheader()
            writer.writerows(rows)
    (out / "README.txt").write_text("Import all CSV files into Power BI Desktop. Refresh after rerunning the local pipelines.\n")
    return out


if __name__ == "__main__":
    print(export())
