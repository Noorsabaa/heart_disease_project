# CardioPredict — Heart Disease Prediction
### CS-245 Machine Learning · NUST SEECS · Spring 2026

A production-ready machine learning system for binary heart disease classification using the Cleveland Heart Disease Dataset. The project includes a supervised and unsupervised ML pipeline, a user-facing prediction app with authentication, and a separate interactive analytics dashboard.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Results](#model-results)
- [First-Time Setup](#first-time-setup)
- [Daily Usage](#daily-usage)

---

## Project Structure

```
heart_disease_project/
│
├── data/
│   ├── raw/                          # Original Cleveland Heart Disease CSV
│   └── processed/                    # Cleaned train/test splits (auto-generated)
│
├── src/
│   ├── config.py                     # All paths, constants, feature metadata
│   ├── preprocessing.py              # 8-step cleaning and feature engineering pipeline
│   ├── evaluation.py                 # Metrics, plots, cross-validation, reports
│   └── models/
│       ├── base_model.py             # Abstract base class (build → fit → predict → save)
│       ├── logistic_regression_model.py
│       ├── random_forest_model.py
│       ├── svm_model.py
│       └── unsupervised.py           # K-Means + PCA
│
├── app/
│   ├── main.py                       # Streamlit prediction app (auth + history)
│   ├── database.py                   # SQLite user auth and prediction storage
│   └── utils.py                      # Inference helpers, gauge chart, input form
│
├── dashboard/
│   └── dashboard.py                  # 6-page interactive data science dashboard
│
├── notebooks/
│   └── eda.py                        # Standalone EDA script (saves 7 figures)
│
├── models/                           # Trained .pkl files and metrics CSVs
├── figures/                          # All saved evaluation and EDA plots
│
├── train_models.py                   # Master training script — run this first
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Name | Cleveland Heart Disease Dataset |
| Source | UCI Machine Learning Repository |
| Patients | 303 |
| Features | 13 clinical variables |
| Target | Binary — 0 = No Disease, 1 = Disease (binarised from severity 0–4) |
| Class Balance | 54.2% disease / 45.8% no disease |
| Missing Values | `ca` (4 rows), `thal` (2 rows) — handled via imputation |

---

## Machine Learning Pipeline

### Preprocessing (8 Steps)

| Step | Method | Justification |
|---|---|---|
| Binarize target | `(target > 0).astype(int)` | Unify severity levels 1–4 into a single disease class |
| Fix data types | `astype(int)`, `astype('Int64')` | Categorical columns stored as float64 in raw data |
| Remove duplicates | `drop_duplicates()` | Data integrity check |
| Impute missing values | `ca` → median, `thal` → mode | Low missingness — imputation preferred over row deletion |
| Winsorise outliers | IQR × 1.5 capping | Clinical extremes are valid readings, not errors |
| Feature engineering | `max_hr_reserve`, `age_group`, `bp_category` | Domain-driven derived features |
| Encode categoricals | `pd.get_dummies(drop_first=True)` | 4 nominal + 2 engineered categorical columns |
| Scale numerics | `StandardScaler` — fit on train only | Prevents data leakage; required for LR and SVM |

### Models

| Model | Type | Role |
|---|---|---|
| Logistic Regression | Supervised | Interpretable linear baseline |
| Random Forest | Supervised | Primary model — best overall performance |
| SVM (RBF kernel) | Supervised | Non-linear kernel comparison |
| K-Means (k=2) + PCA | Unsupervised | Cluster discovery and PCA visualisation |

---

## Model Results

Evaluated on held-out test set (n=61, 20% stratified split). **Recall is the primary metric** — missing a diseased patient has a higher clinical cost than a false alarm.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Random Forest** | **93.4%** | **90.0%** | **96.4%** | **93.1%** | **97.2%** |
| Logistic Regression | 88.5% | 83.9% | 92.9% | 88.1% | 95.6% |
| SVM (RBF) | 85.2% | 78.8% | 92.9% | 85.3% | 92.0% |

**K-Means:** Silhouette Score = 0.40 · PCA explained variance (PC1 + PC2) = 89.2%

---

## First-Time Setup

Follow these steps in order after cloning or downloading the repository.

> **Note:** These steps are done **once only**. After setup is complete, you only need the two commands listed under [Daily Usage](#daily-usage).

---

### 1 · Open the project in VS Code

Open VS Code, click **File → Open Folder**, and select the `heart_disease_project` folder — the one that contains `train_models.py` directly inside it.

Open the integrated terminal with `` Ctrl + ` ``.

---

### 2 · Enable PowerShell scripts (Windows only — one-time system setting)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Press `Y` when prompted. This allows venv activation scripts to run. Skip this step if you are on macOS or Linux.

---

### 3 · Create a virtual environment

```powershell
python -m venv venv
```

---

### 4 · Activate the virtual environment

```powershell
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Your terminal prompt will show `(venv)` when activated. **Every terminal session requires this activation command.**

---

### 5 · Install dependencies

```powershell
pip install -r requirements.txt
```

This installs all required packages into the isolated venv. Takes 2–4 minutes on first run.

---

### 6 · Train all models

```powershell
python train_models.py
```

This single command executes the full pipeline:

- Cleans and preprocesses the raw data
- Engineers three new clinical features
- Trains Logistic Regression, Random Forest, and SVM via cross-validated grid search
- Fits K-Means (k=2) and PCA on the processed feature space
- Evaluates all models and saves the best
- Generates all evaluation figures (confusion matrices, ROC curves, feature importance)
- Runs the EDA script and saves all exploratory plots

Expected output:

```
HEART DISEASE PREDICTION — TRAINING PIPELINE

PREPROCESSING PIPELINE
  Train: (242, 24)  |  Test: (61, 24)

TRAINING SUPERVISED MODELS
  Training Logistic Regression...
  Training Random Forest...
  Training SVM...

EVALUATION (held-out test set, n=61)
  Model                Accuracy   Recall      F1     AUC
  Random Forest          0.9344   0.9643   0.9310  0.9719
  Logistic Regression    0.8852   0.9286   0.8814  0.9556
  SVM                    0.8525   0.9286   0.8525  0.9199

All artifacts → models/
All figures   → figures/
```

Setup is now complete.

---

### 7 · Select the VS Code Python interpreter (optional but recommended)

Press `Ctrl + Shift + P` → type **Python: Select Interpreter** → choose the entry that shows `venv` in its path, for example:

```
Python 3.x.x ('venv': venv)  .\venv\Scripts\python.exe
```

VS Code will then activate the venv automatically in every new terminal.

---

## Daily Usage

After the first-time setup, two commands are all you need.

**To launch the prediction app** (login, patient input, risk gauge, history):

```powershell
venv\Scripts\activate
streamlit run app/main.py
```

Open your browser to **http://localhost:8501**

---

**To launch the analytics dashboard** (EDA, model performance, cluster analysis):

```powershell
venv\Scripts\activate
streamlit run dashboard/dashboard.py --server.port 8502
```

Open your browser to **http://localhost:8502**

---

Both can run at the same time in separate terminal tabs.

---

*Cleveland Heart Disease Dataset — UCI Machine Learning Repository*
*CS-245 Machine Learning · NUST SEECS · Spring 2026*
