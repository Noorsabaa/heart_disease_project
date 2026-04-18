"""
app/main.py
-----------
CardioPredict — Heart Disease Prediction App.

Run:  streamlit run app/main.py

Pages (authenticated):
  🔬  New Prediction   — enter clinical data → get risk score + gauge
  📋  My History       — past predictions + trend chart
  ⚖️  Compare Models   — all 3 models on same input, side-by-side
  ℹ️  About            — project info, architecture, academic details
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.database import init_db, register_user, login_user, save_prediction, get_user_predictions
from app.utils    import load_artifacts, run_prediction, run_all_models, make_gauge, render_input_form, SHARED_CSS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(SHARED_CSS, unsafe_allow_html=True)
init_db()

# ── Session defaults ──────────────────────────────────────────────────────────
for k, v in {"logged_in": False, "user": None, "page": "login"}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGES
# ══════════════════════════════════════════════════════════════════════════════
def page_login():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center'>"
            "<p style='font-size:48px;margin:0'>🫀</p>"
            "<h1 style='margin:8px 0 4px'>CardioPredict</h1>"
            "<p style='color:#888;margin-bottom:32px'>AI-Powered Heart Disease Risk Assessment</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.container():
            st.markdown("#### Sign In")
            username = st.text_input("Username", placeholder="your_username", key="li_u")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="li_p")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Sign In", use_container_width=True, key="btn_login"):
                    user = login_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user      = user
                        st.session_state.page      = "predict"
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
            with col_b:
                if st.button("Create Account →", use_container_width=True, key="btn_to_signup"):
                    st.session_state.page = "signup"
                    st.rerun()
        st.markdown(
            "<p style='text-align:center;color:#aaa;font-size:12px;margin-top:24px'>"
            "DS 401 · CS 245 · NUST SEECS Spring 2026</p>",
            unsafe_allow_html=True,
        )


def page_signup():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center'>"
            "<h2 style='margin-bottom:4px'>Create Account</h2>"
            "<p style='color:#888;margin-bottom:28px'>Join CardioPredict</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        name     = st.text_input("Full Name",        placeholder="Ali Hassan",          key="su_n")
        username = st.text_input("Username",          placeholder="ali_hassan",          key="su_u")
        email    = st.text_input("Email Address",     placeholder="ali@example.com",     key="su_e")
        password = st.text_input("Password",          type="password",                   key="su_p")
        confirm  = st.text_input("Confirm Password",  type="password",                   key="su_c")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Create Account", use_container_width=True, key="btn_register"):
                if not all([name, username, email, password, confirm]):
                    st.error("All fields are required.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(username, name, email, password)
                    if ok:
                        st.success("✅ Account created! Please sign in.")
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.error(msg)
        with col_b:
            if st.button("← Back to Login", use_container_width=True, key="btn_back"):
                st.session_state.page = "login"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='padding:20px 0 8px'>"
            "<p style='font-size:22px;font-weight:700;margin:0'>🫀 CardioPredict</p>"
            "<p style='font-size:11px;opacity:0.5;margin:2px 0 0'>Heart Disease Risk Assessment</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        user = st.session_state.user
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.08);border-radius:8px;"
            f"padding:10px 14px;margin:8px 0 20px'>"
            f"<p style='margin:0;font-size:12px;opacity:0.6'>Signed in as</p>"
            f"<p style='margin:2px 0 0;font-size:15px;font-weight:600'>{user['name']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        nav = {
            "predict": "🔬  New Prediction",
            "history": "📋  My History",
            "compare": "⚖️  Compare Models",
            "about":   "ℹ️  About",
        }
        for key, label in nav.items():
            is_active = st.session_state.page == key
            btn_type  = "primary" if is_active else "secondary"
            if st.button(label, use_container_width=True,
                         key=f"nav_{key}", type=btn_type):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        if st.button("🚪  Sign Out", use_container_width=True, key="nav_logout"):
            st.session_state.logged_in = False
            st.session_state.user      = None
            st.session_state.page      = "login"
            st.rerun()

        st.markdown(
            "<p style='font-size:10px;opacity:0.35;text-align:center;margin-top:20px'>"
            "DS 401 · CS 245 · NUST SEECS</p>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: NEW PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown("## 🔬 New Prediction")
    st.caption("Enter the patient's clinical values. The model estimates heart disease probability.")
    st.markdown("---")

    model_name = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "SVM"],
        index=0,
        help="Random Forest achieves the best performance (AUC 97.2%, Recall 96.4%).",
    )

    with st.form("predict_form", clear_on_submit=False):
        st.markdown("#### 🩺 Patient Clinical Data")
        inputs    = render_input_form(key_prefix="pf_")
        submitted = st.form_submit_button("▶  Run Prediction", use_container_width=True)

    if submitted:
        with st.spinner("Analysing clinical data..."):
            result = run_prediction(inputs, model_name)

        save_prediction(
            st.session_state.user["id"],
            inputs, model_name,
            result["prediction"], result["probability"],
        )

        st.markdown("---")
        st.markdown("### 📊 Result")

        col_gauge, col_detail = st.columns([1, 1.4])

        with col_gauge:
            st.plotly_chart(
                make_gauge(result["probability"], result["risk_level"], result["risk_color"]),
                use_container_width=True,
            )

        with col_detail:
            color = result["risk_color"]
            emoji = "⚠️" if result["prediction"] == 1 else "✅"
            card_cls = "result-positive" if result["prediction"] == 1 else "result-negative"

            st.markdown(
                f"<div class='card {card_cls}'>"
                f"<p style='font-size:22px;font-weight:700;color:{color};margin:0 0 10px'>"
                f"{emoji}  {result['label']}</p>"
                f"<table style='width:100%;font-size:15px;border-collapse:collapse'>"
                f"<tr><td style='padding:5px 0;color:#555'>Probability</td>"
                f"<td style='font-weight:700;color:{color}'>{result['probability']:.1%}</td></tr>"
                f"<tr><td style='padding:5px 0;color:#555'>Risk Level</td>"
                f"<td style='font-weight:700;color:{color}'>{result['risk_level']}</td></tr>"
                f"<tr><td style='padding:5px 0;color:#555'>Model Used</td>"
                f"<td>{model_name}</td></tr>"
                f"</table>"
                f"<hr style='margin:14px 0;border-color:#eee'>"
                f"<p style='font-size:11px;color:#999;margin:0'>"
                f"⚠️ Educational use only. Not a substitute for professional medical diagnosis.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Input summary table
        st.markdown("#### 🔑 Input Summary")
        cp_map   = {1:"Typical Angina",2:"Atypical Angina",3:"Non-Anginal Pain",4:"Asymptomatic"}
        thal_map = {3:"Normal",6:"Fixed Defect",7:"Reversible Defect"}
        slope_map= {1:"Upsloping",2:"Flat",3:"Downsloping"}
        summary  = {
            "Age": f"{inputs['age']} yrs",
            "Sex": "Male" if inputs["sex"]==1 else "Female",
            "Chest Pain": cp_map[inputs["cp"]],
            "Blood Pressure": f"{inputs['trestbps']} mm Hg",
            "Cholesterol": f"{inputs['chol']} mg/dl",
            "Max Heart Rate": f"{inputs['thalach']} bpm",
            "ST Depression": f"{inputs['oldpeak']} mm",
            "Major Vessels": str(inputs["ca"]),
            "Thalassemia": thal_map[inputs["thal"]],
            "Exercise Angina": "Yes" if inputs["exang"]==1 else "No",
            "ST Slope": slope_map[inputs["slope"]],
            "Fasting BS": "High" if inputs["fbs"]==1 else "Normal",
        }
        df_sum = pd.DataFrame(list(summary.items()), columns=["Feature", "Value"])
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        st.success("✅ Prediction saved to your history.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MY HISTORY
# ══════════════════════════════════════════════════════════════════════════════
def page_history():
    st.markdown("## 📋 My Prediction History")
    records = get_user_predictions(st.session_state.user["id"])

    if not records:
        st.info("No predictions yet. Head to **New Prediction** to get started.")
        return

    df = pd.DataFrame(records)
    df["created_at"]    = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d  %H:%M")
    df["Result"]        = df["prediction"].map({0: "✅ No Disease", 1: "⚠️ Disease"})
    df["Probability %"] = (df["probability"] * 100).round(1).astype(str) + "%"

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total   = len(df)
    disease = int((df["prediction"] == 1).sum())
    avg_p   = df["probability"].mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Predictions", total)
    k2.metric("Disease Detected",  disease)
    k3.metric("No Disease",        total - disease)
    k4.metric("Avg. Risk Score",   f"{avg_p:.1f}%")

    st.markdown("---")

    # ── Trend line ────────────────────────────────────────────────────────────
    if total >= 2:
        df_chart = df.copy()
        df_chart["prob_pct"] = df_chart["probability"] * 100
        df_chart["#"]        = range(1, len(df_chart) + 1)
        df_chart_sorted      = df_chart.sort_values("#")

        fig = px.line(
            df_chart_sorted, x="#", y="prob_pct",
            markers=True,
            color_discrete_sequence=["#1D9E75"],
            labels={"#": "Prediction Number", "prob_pct": "Disease Probability (%)"},
            title="Risk Score Trend Across Predictions",
        )
        fig.add_hline(y=65, line_dash="dot", line_color="#E24B4A",
                      annotation_text="High Risk  (65%)",
                      annotation_position="bottom right")
        fig.add_hline(y=35, line_dash="dot", line_color="#BA7517",
                      annotation_text="Moderate  (35%)",
                      annotation_position="bottom right")
        fig.update_layout(
            height=300,
            margin=dict(t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f9fafb",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Distribution of risk levels ────────────────────────────────────────────
    col_pie, col_table = st.columns([1, 2])

    with col_pie:
        risk_vc = df["risk_level"].value_counts().reset_index()
        risk_vc.columns = ["Risk Level", "Count"]
        color_map = {"High Risk": "#E24B4A", "Moderate Risk": "#BA7517", "Low Risk": "#1D9E75"}
        fig2 = px.pie(
            risk_vc, names="Risk Level", values="Count",
            color="Risk Level", color_discrete_map=color_map,
            hole=0.5, title="Risk Distribution",
        )
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        fig2.update_layout(
            height=280, showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_table:
        st.markdown("#### All Records")
        show_cols = ["created_at", "Result", "Probability %", "risk_level",
                     "model_used", "age", "thalach", "chol", "trestbps", "ca"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(
            df[show_cols].rename(columns={
                "created_at":  "Time",
                "risk_level":  "Risk",
                "model_used":  "Model",
            }),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: COMPARE MODELS
# ══════════════════════════════════════════════════════════════════════════════
def page_compare():
    st.markdown("## ⚖️ Compare Models")
    st.caption("Submit patient data once — see predictions from all three trained models simultaneously.")
    st.markdown("---")

    with st.form("compare_form"):
        st.markdown("#### 🩺 Patient Clinical Data")
        inputs    = render_input_form(key_prefix="cmp_")
        submitted = st.form_submit_button("▶  Compare All Models", use_container_width=True)

    if submitted:
        with st.spinner("Running all three models..."):
            df_cmp = run_all_models(inputs)

        st.markdown("---")
        st.markdown("### 🏆 Side-by-Side Results")

        # Model cards
        cols    = st.columns(3)
        palette = {"High": "#E24B4A", "Moderate": "#BA7517", "Low": "#1D9E75"}
        for i, row in df_cmp.iterrows():
            color  = palette.get(row["Risk"], "#888")
            border = "3px solid #1D9E75" if row["Model"] == "Random Forest" else "1px solid #e2e8f0"
            star   = "⭐ Best Model" if row["Model"] == "Random Forest" else "&nbsp;"
            with cols[i]:
                st.markdown(
                    f"<div style='border:{border};border-radius:14px;padding:22px;"
                    f"background:white;box-shadow:0 2px 8px rgba(0,0,0,0.07);height:100%'>"
                    f"<p style='font-size:11px;color:#1D9E75;font-weight:600;margin:0'>{star}</p>"
                    f"<p style='font-size:17px;font-weight:700;margin:6px 0 4px'>{row['Model']}</p>"
                    f"<p style='font-size:30px;font-weight:800;color:{color};margin:10px 0 4px'>"
                    f"{row['Prob %']}</p>"
                    f"<p style='font-size:13px;margin:2px 0'>{row['Prediction']}</p>"
                    f"<p style='font-size:12px;color:{color};font-weight:600;margin:4px 0'>"
                    f"Risk: {row['Risk']}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("#### Probability Bar Chart")
        fig = px.bar(
            df_cmp, x="Model", y="Probability",
            color="Model",
            color_discrete_sequence=["#1D9E75", "#7F77DD", "#E8593C"],
            text=df_cmp["Prob %"],
            range_y=[0, 1],
            labels={"Probability": "P(Heart Disease)"},
        )
        fig.add_hline(y=0.65, line_dash="dot", line_color="#E24B4A",
                      annotation_text="High Risk Threshold (65%)")
        fig.add_hline(y=0.35, line_dash="dot", line_color="#BA7517",
                      annotation_text="Moderate Threshold (35%)")
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=380, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f9fafb",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trained performance reference
        st.markdown("#### Trained Model Performance (held-out test set, n=61)")
        perf = pd.DataFrame([
            {"Model": "Random Forest",       "Accuracy": "93.4%", "Recall": "96.4%",
             "F1": "93.1%", "AUC": "97.2%", "Note": "⭐ Best model"},
            {"Model": "Logistic Regression", "Accuracy": "88.5%", "Recall": "92.9%",
             "F1": "88.1%", "AUC": "95.6%", "Note": "Most interpretable"},
            {"Model": "SVM (RBF)",           "Accuracy": "85.2%", "Recall": "92.9%",
             "F1": "85.3%", "AUC": "92.0%", "Note": ""},
        ])
        st.dataframe(perf, use_container_width=True, hide_index=True)
        st.caption("Primary metric: Recall — missing a diseased patient carries higher clinical cost.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown("## ℹ️ About CardioPredict")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### 🎯 Project Overview
**CardioPredict** was developed as the semester project for **DS 401 (Data Science)**
and **CS 245 (Machine Learning)** at **NUST SEECS, Spring 2026**.

The system predicts heart disease risk from 13 clinical features using the
**Cleveland Heart Disease Dataset** (UCI ML Repository, 303 patients).

### 📊 Dataset
| | |
|---|---|
| Source | UCI Machine Learning Repository |
| Patients | 303 |
| Features | 13 clinical variables |
| Target | Binary (0 = no disease, 1 = disease) |
| Class balance | 54.2% disease / 45.8% no disease |

### 🤖 Models
| Model | Accuracy | Recall | AUC |
|---|---|---|---|
| Random Forest ⭐ | 93.4% | 96.4% | 97.2% |
| Logistic Regression | 88.5% | 92.9% | 95.6% |
| SVM (RBF) | 85.2% | 92.9% | 92.0% |

### 🔬 Unsupervised
- **K-Means (k=2):** Silhouette Score = 0.40
- **PCA (2 components):** 89.2% variance explained
""")

    with c2:
        st.markdown("""
### 🏗️ Architecture
```
heart_disease_project/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── models/
│       ├── base_model.py
│       ├── random_forest_model.py
│       ├── logistic_regression_model.py
│       ├── svm_model.py
│       └── unsupervised.py
├── app/
│   ├── main.py         ← this app
│   ├── database.py     ← SQLite auth
│   └── utils.py        ← helpers
├── dashboard/
│   └── dashboard.py    ← EDA dashboard
├── data/raw/           ← Cleveland CSV
├── models/             ← trained .pkl files
├── figures/            ← evaluation plots
└── train_models.py     ← run this first
```

### 🔑 Key Design Decisions
- **Recall** as primary metric — false negatives are clinically dangerous
- **Winsorisation** over deletion for clinical outliers
- **Fit scaler on train only** — prevents data leakage
- **`class_weight='balanced'`** — handles mild class imbalance
- **`models/` directory** over single `models.py` — uses abstract base class for a scalable, modular interface
""")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#aaa;font-size:12px'>"
        "DS 401 · CS 245 · NUST SEECS Spring 2026 · Cleveland Heart Disease Dataset (UCI ML Repo)"
        "<br>⚠️ For educational purposes only — not a medical device.</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        if st.session_state.page == "signup":
            page_signup()
        else:
            page_login()
        return

    render_sidebar()

    dispatch = {
        "predict": page_predict,
        "history": page_history,
        "compare": page_compare,
        "about":   page_about,
    }
    dispatch.get(st.session_state.page, page_predict)()


if __name__ == "__main__":
    main()
