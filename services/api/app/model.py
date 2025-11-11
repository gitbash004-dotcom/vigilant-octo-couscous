from __future__ import annotations

from pathlib import Path
from typing import Any, List

import joblib
import pandas as pd

from .config import get_settings
from .train_model import train


class PredictionModel:
    def __init__(self, artifact_path: Path) -> None:
        self.artifact_path = artifact_path
        self.pipeline = None

    def ensure_model(self) -> None:
        if self.pipeline is None:
            self.pipeline = self._load_or_train()

    def _load_or_train(self):
        if self.artifact_path.exists():
            data = joblib.load(self.artifact_path)
            return data["model"]
        # Train a fresh model when the artifact is missing
        train()
        data = joblib.load(self.artifact_path)
        return data["model"]

    def predict(self, records: List[dict[str, Any]]) -> List[tuple[float, float]]:
        self.ensure_model()
        df = pd.DataFrame(records)
        probabilities = self.pipeline.predict_proba(df)[:, 1]
        predictions = (probabilities >= 0.5).astype(float)
        return list(zip(predictions.tolist(), probabilities.tolist()))


_model_instance: PredictionModel | None = None


def get_model() -> PredictionModel:
    global _model_instance
    if _model_instance is None:
        settings = get_settings()
        _model_instance = PredictionModel(settings.model_path)
    return _model_instance
