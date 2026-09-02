from pathlib import Path

import joblib

from .train import HIGH_PRIORITY, MODEL_PATH, train


def load_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)


class ComplaintClassifier:
    """Thin wrapper so the API never talks to sklearn directly."""

    def __init__(self):
        self.model = load_model()

    def predict(self, text: str) -> dict:
        proba = self.model.predict_proba([text])[0]
        idx = proba.argmax()
        category = self.model.classes_[idx]
        confidence = round(float(proba[idx]), 3)
        priority = "high" if category in HIGH_PRIORITY or confidence < 0.45 else "normal"
        return {"category": category, "confidence": confidence, "priority": priority}
