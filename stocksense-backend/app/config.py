"""Central place for all settings. Override any of them with environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

NEWS_API_KEY =os.getenv("NEWS_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Dataset used by train.py  (copy all-data.csv into the data/ folder)
DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "data" / "all-data.csv"))

# Where the trained pipeline and its metrics are stored
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "sentiment_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", BASE_DIR / "models" / "metrics.json"))

# Origins allowed to call the API from a browser (your React dev server)
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    ).split(",")
    if o.strip()
]

# Limits to keep the API safe
MAX_TEXT_LENGTH = 2000
MAX_BATCH_SIZE = 100
