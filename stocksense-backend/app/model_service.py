"""Loads the trained pipeline once and runs predictions."""
import json
import logging

import joblib

from app.config import METRICS_PATH, MODEL_PATH
from app.preprocess import clean_text

logger = logging.getLogger("stocksense")


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no trained model is available."""


class SentimentModel:
    def __init__(self) -> None:
        self.pipeline = None
        self.metrics: dict | None = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            logger.warning("Model file not found at %s. Run `python train.py` first.", MODEL_PATH)
            return
        self.pipeline = joblib.load(MODEL_PATH)
        if METRICS_PATH.exists():
            self.metrics = json.loads(METRICS_PATH.read_text())
        logger.info("Model loaded from %s", MODEL_PATH)

    @property
    def classes(self) -> list[str]:
        self._require_loaded()
        return [str(c) for c in self.pipeline.classes_]

    def predict(self, texts: list[str]) -> list[dict]:
        self._require_loaded()
        cleaned = [clean_text(t) for t in texts]  # same cleaning as training
        probabilities = self.pipeline.predict_proba(cleaned)
        classes = self.classes

        results = []
        for row in probabilities:
            best = int(row.argmax())
            results.append(
                {
                    "sentiment": classes[best],
                    "confidence": round(float(row[best]), 4),
                    "probabilities": {c: round(float(p), 4) for c, p in zip(classes, row)},
                }
            )
        return results

    def _require_loaded(self) -> None:
        if not self.loaded:
            raise ModelNotLoadedError("Model is not loaded. Run `python train.py` first.")


sentiment_model = SentimentModel()
