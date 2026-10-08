from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def monitoring_status(artifact_dir: str | Path) -> dict:
    root = Path(artifact_dir)
    drift = root / "drift_report.json"
    report = json.loads(drift.read_text()) if drift.exists() else None
    statuses = [x.get("status") for x in (report or {}).get("feature_drift", [])]
    flagged = sum(s in {"warning", "significant_drift"} for s in statuses)
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "drift_available": report is not None,
        "cohort_type": "proxy halves of held-out data" if report else "unavailable",
        "performance_decay": "not estimable without timestamped labels",
        "status": "red" if flagged >= 3 else ("amber" if flagged else "green"),
        "flagged_features": flagged,
        "review_queue_enabled": True,
        "treatment_effect_drift": "not estimable: no treatment/control log",
    }
