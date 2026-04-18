"""src/models/svm_model.py — Support Vector Machine (RBF kernel)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.svm import SVC
from src.models.base_model import BaseHeartModel
from src.config import MODEL_PATHS, RANDOM_STATE


class SVMModel(BaseHeartModel):
    """
    SVM with RBF kernel — non-linear kernel method.
    Completes the comparison triangle: linear (LR) / ensemble (RF) / kernel (SVM).
    probability=True enables predict_proba for ROC curves.
    Requires StandardScaler (already applied in preprocessing).
    """
    def __init__(self):
        super().__init__(name="SVM", model_path=MODEL_PATHS["SVM"])

    def build(self):
        self.model = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )
        return self
