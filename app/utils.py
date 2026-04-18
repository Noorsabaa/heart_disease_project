"""
app/utils.py
------------
Shared helpers for the Streamlit app:
  load_artifacts()      — cached model/scaler loading
  run_prediction()      — single-model inference
  run_all_models()      — all-model comparison
  make_gauge()          — Plotly risk gauge
  render_input_form()   — 13-field clinical input form
  SHARED_CSS            — global style injection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import MODELS_DIR, MODEL_PATHS, SCALE_FEATURES
from src.preprocessing import prepare_single_input


# ── Cached artifacts ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    scaler        = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = json.load(open(MODELS_DIR / "feature_names.json"))
    models        = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    return scaler, feature_names, models


# ── Inference ─────────────────────────────────────────────────────────────────
def run_prediction(inputs: dict, model_name: str) -> dict:
    scaler, feature_names, models = load_artifacts()
    X    = prepare_single_input(inputs, scaler, feature_names)
    pred = int(models[model_name].predict(X)[0])
    prob = float(models[model_name].predict_proba(X)[0][1])

    if prob >= 0.65:
        risk, risk_color = "High Risk",     "#E24B4A"
    elif prob >= 0.35:
        risk, risk_color = "Moderate Risk", "#BA7517"
    else:
        risk, risk_color = "Low Risk",      "#1D9E75"

    return {
        "prediction":  pred,
        "probability": round(prob, 4),
        "label":       "Heart Disease Detected" if pred == 1 else "No Heart Disease",
        "risk_level":  risk,
        "risk_color":  risk_color,
    }


def run_all_models(inputs: dict) -> pd.DataFrame:
    scaler, feature_names, models = load_artifacts()
    rows = []
    for name, model in models.items():
        X    = prepare_single_input(inputs, scaler, feature_names)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        rows.append({
            "Model":      name,
            "Prediction": "⚠️ Disease" if pred == 1 else "✅ No Disease",
            "Probability": prob,
            "Prob %":     f"{prob:.1%}",
            "Risk":       "High" if prob >= 0.65 else ("Moderate" if prob >= 0.35 else "Low"),
        })
    return pd.DataFrame(rows)


# ── Gauge chart ───────────────────────────────────────────────────────────────
def make_gauge(probability: float, risk_level: str, risk_color: str) -> go.Figure:
    pct = round(probability * 100, 1)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = pct,
        title = {
            "text": (f"<b>Risk Score</b><br>"
                     f"<span style='font-size:15px;color:{risk_color}'>{risk_level}</span>"),
            "font": {"size": 18},
        },
        number = {"suffix": "%", "font": {"size": 36, "color": risk_color}},
        gauge  = {
            "axis":      {"range": [0, 100], "tickwidth": 1},
            "bar":       {"color": risk_color, "thickness": 0.3},
            "bgcolor":   "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "#E1F5EE"},
                {"range": [35, 65], "color": "#FAEEDA"},
                {"range": [65,100], "color": "#FCEBEB"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.85,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=70, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Clinical input form ───────────────────────────────────────────────────────
def render_input_form(key_prefix: str = "") -> dict:
    """Render all 13 clinical feature widgets. Returns raw input dict."""
    inputs = {}
    c1, c2 = st.columns(2)

    with c1:
        inputs["age"] = st.slider(
            "Age (years)", 29, 77, 54, key=f"{key_prefix}age")
        inputs["sex"] = st.selectbox(
            "Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
            key=f"{key_prefix}sex")
        inputs["cp"] = st.selectbox(
            "Chest Pain Type", [1, 2, 3, 4],
            format_func=lambda x: {1:"Typical Angina",2:"Atypical Angina",
                                    3:"Non-Anginal Pain",4:"Asymptomatic"}[x],
            key=f"{key_prefix}cp")
        inputs["trestbps"] = st.slider(
            "Resting Blood Pressure (mm Hg)", 90, 200, 131, key=f"{key_prefix}trestbps")
        inputs["chol"] = st.slider(
            "Serum Cholesterol (mg/dl)", 100, 570, 246, key=f"{key_prefix}chol")
        inputs["fbs"] = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key=f"{key_prefix}fbs")
        inputs["restecg"] = st.selectbox(
            "Resting ECG Results", [0, 1, 2],
            format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x],
            key=f"{key_prefix}restecg")

    with c2:
        inputs["thalach"] = st.slider(
            "Max Heart Rate Achieved (bpm)", 60, 210, 150, key=f"{key_prefix}thalach")
        inputs["exang"] = st.selectbox(
            "Exercise-Induced Angina", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key=f"{key_prefix}exang")
        inputs["oldpeak"] = st.slider(
            "ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1, key=f"{key_prefix}oldpeak")
        inputs["slope"] = st.selectbox(
            "Slope of Peak ST Segment", [1, 2, 3],
            format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x],
            key=f"{key_prefix}slope")
        inputs["ca"] = st.slider(
            "Major Vessels Colored (0–3)", 0, 3, 0, key=f"{key_prefix}ca")
        inputs["thal"] = st.selectbox(
            "Thalassemia Type", [3, 6, 7],
            format_func=lambda x: {3:"Normal",6:"Fixed Defect",7:"Reversible Defect"}[x],
            key=f"{key_prefix}thal")

    return inputs


# ── Shared CSS ────────────────────────────────────────────────────────────────
SHARED_CSS = """
<style>
/* Background */
[data-testid="stAppViewContainer"] { background: #F7F8FA; }
[data-testid="stSidebar"]          { background: #16213e; }
[data-testid="stSidebar"] *        { color: #e2e8f0 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #e8ecf0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* Primary button */
.stButton > button {
    background: #1D9E75 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #168a65 !important; }

/* Cards */
.card {
    background: white;
    border-radius: 14px;
    padding: 24px;
    border: 1px solid #e8ecf0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin: 8px 0;
}
.result-positive {
    border-left: 5px solid #E24B4A;
    background: #fff8f8;
}
.result-negative {
    border-left: 5px solid #1D9E75;
    background: #f7fdf9;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
"""
