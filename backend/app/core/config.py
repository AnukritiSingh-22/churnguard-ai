import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
DB_PATH = os.environ.get("CHURNGUARD_DB_PATH", str(DATA_DIR / "churnguard.db"))

CORS_ORIGINS = os.environ.get("CHURNGUARD_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
