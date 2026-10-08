from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _monthly_revenue(raw_path: str | Path) -> pd.Series:
    frame = pd.read_csv(raw_path, usecols=["InvoiceDate", "Quantity", "UnitPrice"])
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["revenue"] = frame["Quantity"].clip(lower=0) * frame["UnitPrice"].clip(lower=0)
    return frame.dropna(subset=["InvoiceDate"]).set_index("InvoiceDate")["revenue"].resample("MS").sum()


def build_revenue_forecast(raw_path: str | Path, artifact_dir: str | Path, horizon: int = 3) -> dict:
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    series = _monthly_revenue(raw_path)
    values = series.to_numpy(dtype=float)
    season = min(3, max(1, len(values) // 3))
    errors: list[float] = []
    fold_rows: list[dict] = []
    for end in range(max(season, 4), len(values)):
        train = values[:end]
        actual = values[end]
        pred = float(train[-season:].mean())
        errors.append(actual - pred)
        fold_rows.append({"period": series.index[end].strftime("%Y-%m"), "actual": round(actual, 2), "prediction": round(pred, 2)})
    mae = float(np.mean(np.abs(errors))) if errors else 0.0
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
    denom = float(np.sum(np.abs(values[-len(errors):]))) if errors else 1.0
    mape = float(np.sum(np.abs(errors)) / max(denom, 1.0) * 100)
    residual_q = float(np.quantile(np.abs(errors), 0.9)) if errors else 0.0
    last = values[-season:]
    future_index = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    forecasts = []
    for period in future_index:
        point = float(np.mean(last))
        forecasts.append({
            "period": period.strftime("%Y-%m"),
            "actual": None,
            "prediction": round(point, 2),
            "lower": round(max(0.0, point - residual_q), 2),
            "upper": round(point + residual_q, 2),
            "interval_method": "90% split-conformal residual interval",
        })
        last = np.append(last[1:], point)
    result = {
        "target": "monthly_revenue",
        "frequency": "MS",
        "model": "seasonal_naive",
        "seasonal_period_months": season,
        "source": "backend/data/retail/online_retail_raw.csv",
        "backtest": {"folds": fold_rows, "mape": round(mape, 4), "rmse": round(rmse, 2), "mae": round(mae, 2),
                     "honesty_note": "Rolling-origin holdout; no future month is used to fit an earlier fold."},
        "history": [{"period": i.strftime("%Y-%m"), "actual": round(float(v), 2)} for i, v in series.items()],
        "forecast": forecasts,
        "interval_note": "Intervals are conformal residual intervals, not a guarantee and not a probabilistic sales model.",
    }
    (out / "revenue_forecast.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    from app.core.config import ARTIFACT_DIR, DATA_DIR
    print(build_revenue_forecast(DATA_DIR / "retail" / "online_retail_raw.csv", ARTIFACT_DIR))
