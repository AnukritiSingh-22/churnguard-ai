from __future__ import annotations

import json
from pathlib import Path


def probability_interval(probability: float, artifact_dir: str | Path, confidence: float = 0.9) -> dict:
    """Return a split-conformal interval when its calibration quantile exists."""
    path = Path(artifact_dir) / "conformal.json"
    if not path.exists():
        return {
            "lower": round(max(0.0, probability - 0.15), 6),
            "upper": round(min(1.0, probability + 0.15), 6),
            "confidence": confidence,
            "method": "unavailable-artifact-fallback",
            "note": "Run the training pipeline to compute a split-conformal quantile; this fallback is not a calibrated interval.",
        }
    q = float(json.loads(path.read_text()).get("absolute_residual_quantile", 0.15))
    return {"lower": round(max(0.0, probability - q), 6), "upper": round(min(1.0, probability + q), 6),
            "confidence": confidence, "method": "split-conformal absolute residual interval"}
