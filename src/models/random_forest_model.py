"""src/models/random_forest_model.py — Random Forest (primary best model)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseHeartModel
from src.config import MODEL_PATHS, RANDOM_STATE


class RFModel(BaseHeartModel):
    """
    Random Forest — expected best model (AUC ~0.97, Recall ~0.96).
    Handles mixed feature types and non-linear interactions natively.
    class_weight='balanced' compensates for the slight 54/46 imbalance.
    Built-in feature_importances_ answers 'which clinical factors matter most'.
    """
    def __init__(self):
        super().__init__(name="Random Forest", model_path=MODEL_PATHS["Random Forest"])

    def build(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return self
