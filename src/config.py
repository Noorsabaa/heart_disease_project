"""src/config.py — Central configuration: paths, constants, feature metadata."""
from pathlib import Path

ROOT               = Path(__file__).resolve().parent.parent
DATA_RAW_DIR       = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR         = ROOT / "models"
FIGURES_DIR        = ROOT / "figures"
APP_DIR            = ROOT / "app"

RAW_DATA_PATH       = DATA_RAW_DIR       / "heart_disease_cleveland.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "processed_data.csv"
TRAIN_PATH          = DATA_PROCESSED_DIR / "X_train.csv"
TEST_PATH           = DATA_PROCESSED_DIR / "X_test.csv"
Y_TRAIN_PATH        = DATA_PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH         = DATA_PROCESSED_DIR / "y_test.csv"
SCALER_PATH         = MODELS_DIR / "scaler.pkl"
DB_PATH             = APP_DIR    / "heart_app.db"

for d in [DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES     = ["sex", "fbs", "exang"]
ORDINAL_FEATURES    = ["ca"]
NOMINAL_FEATURES    = ["cp", "restecg", "slope", "thal"]
ENGINEERED_FEATURES = ["max_hr_reserve"]
SCALE_FEATURES      = CONTINUOUS_FEATURES + ENGINEERED_FEATURES
TARGET              = "target"

FEATURE_META = {
    "age":      {"label": "Age (years)",                      "min": 29,  "max": 77,  "default": 54,  "type": "slider"},
    "sex":      {"label": "Sex",                              "options": {0: "Female", 1: "Male"},                    "type": "select"},
    "cp":       {"label": "Chest Pain Type",                  "options": {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}, "type": "select"},
    "trestbps": {"label": "Resting Blood Pressure (mm Hg)",   "min": 94,  "max": 200, "default": 131, "type": "slider"},
    "chol":     {"label": "Serum Cholesterol (mg/dl)",        "min": 126, "max": 564, "default": 246, "type": "slider"},
    "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl",  "options": {0: "No", 1: "Yes"},                         "type": "select"},
    "restecg":  {"label": "Resting ECG Results",              "options": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}, "type": "select"},
    "thalach":  {"label": "Max Heart Rate Achieved (bpm)",    "min": 71,  "max": 202, "default": 150, "type": "slider"},
    "exang":    {"label": "Exercise-Induced Angina",          "options": {0: "No", 1: "Yes"},                         "type": "select"},
    "oldpeak":  {"label": "ST Depression (oldpeak)",          "min": 0.0, "max": 6.2, "default": 1.0, "type": "slider", "step": 0.1},
    "slope":    {"label": "Slope of Peak ST Segment",         "options": {1: "Upsloping", 2: "Flat", 3: "Downsloping"}, "type": "select"},
    "ca":       {"label": "Major Vessels Colored (0–3)",      "min": 0,   "max": 3,   "default": 0,   "type": "slider"},
    "thal":     {"label": "Thalassemia Type",                 "options": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}, "type": "select"},
}

MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
    "Random Forest":       MODELS_DIR / "random_forest.pkl",
    "SVM":                 MODELS_DIR / "svm.pkl",
}

BEST_MODEL_NAME = "Random Forest"
RANDOM_STATE    = 42
TEST_SIZE       = 0.20

COLORS = {
    "primary":   "#1D9E75",
    "secondary": "#7F77DD",
    "accent":    "#E8593C",
    "neutral":   "#888780",
    "success":   "#1D9E75",
    "danger":    "#E24B4A",
    "warning":   "#BA7517",
}
