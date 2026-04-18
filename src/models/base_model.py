"""src/models/base_model.py — Abstract base class all classifiers inherit from."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from abc import ABC, abstractmethod
import numpy as np
import joblib


class BaseHeartModel(ABC):
    """Unified interface: build → fit → predict → save/load."""

    def __init__(self, name: str, model_path: Path):
        self.name       = name
        self.model_path = model_path
        self.model      = None

    @abstractmethod
    def build(self):
        """Instantiate the sklearn estimator. Must return self."""

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"  Saved → {self.model_path.name}")

    def load(self):
        self.model = joblib.load(self.model_path)
        return self

    def feature_importances_(self):
        return getattr(self.model, "feature_importances_", np.array([]))
