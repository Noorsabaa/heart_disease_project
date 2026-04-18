"""src/models/logistic_regression_model.py — Logistic Regression (linear baseline)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from src.models.base_model import BaseHeartModel
from src.config import MODEL_PATHS, RANDOM_STATE


class LRModel(BaseHeartModel):
    """
    Logistic Regression — interpretable linear baseline.
    Coefficients map directly to clinical risk factor weights.
    Preferred when model explainability is required (e.g. clinical audit).
    Requires StandardScaler (applied in preprocessing).
    """
    def __init__(self):
        super().__init__(name="Logistic Regression", model_path=MODEL_PATHS["Logistic Regression"])

    def build(self):
        self.model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        return self
