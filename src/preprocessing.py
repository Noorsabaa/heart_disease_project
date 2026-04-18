"""src/preprocessing.py — Full preprocessing pipeline for Cleveland Heart Disease dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAIN_PATH, TEST_PATH,
    Y_TRAIN_PATH, Y_TEST_PATH, SCALER_PATH,
    CONTINUOUS_FEATURES, BINARY_FEATURES, ORDINAL_FEATURES,
    NOMINAL_FEATURES, SCALE_FEATURES, TARGET, RANDOM_STATE, TEST_SIZE,
    DATA_PROCESSED_DIR, MODELS_DIR,
)


def load_raw(path=None):
    return pd.read_csv(path or RAW_DATA_PATH)


def binarize_target(df):
    """0=no disease, 1-4 → 1=disease."""
    df = df.copy()
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df


def fix_dtypes(df):
    df = df.copy()
    df["age"] = df["age"].astype(int)
    for col in BINARY_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES + [TARGET]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


def impute_missing(df):
    """ca → median (ordinal, skewed). thal → mode (nominal). Only 6 rows affected."""
    df = df.copy()
    df["ca"]   = df["ca"].fillna(int(df["ca"].median()))
    df["thal"] = df["thal"].fillna(int(df["thal"].mode()[0]))
    return df


def cap_outliers(df, cols=None, factor=1.5):
    """IQR Winsorisation — cap not drop (clinical extremes are valid readings)."""
    if cols is None:
        cols = ["trestbps", "chol", "thalach", "oldpeak"]
    df = df.copy()
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(Q1 - factor * (Q3 - Q1), Q3 + factor * (Q3 - Q1))
    return df


def engineer_features(df):
    """Three clinically motivated derived features."""
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 45, 55, 65, 120],
        labels=["young_adult", "middle_aged", "senior", "elderly"], right=True,
    )
    df["bp_category"] = pd.cut(
        df["trestbps"], bins=[0, 119, 139, 300],
        labels=["normal", "elevated", "high"], right=True,
    )
    df["max_hr_reserve"] = df["thalach"] - (220 - df["age"])
    return df


def encode_categoricals(df, drop_first=True):
    """OHE nominal + engineered categorical columns."""
    df = df.copy()
    ohe_cols = NOMINAL_FEATURES + ["age_group", "bp_category"]
    for col in ohe_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return pd.get_dummies(df, columns=ohe_cols, drop_first=drop_first, dtype=int)


def split_and_scale(df, save=True):
    """80/20 stratified split + StandardScaler on continuous cols. Returns train/test splits."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    scale_cols = [c for c in SCALE_FEATURES if c in X_train.columns]
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])
    if save:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(TRAIN_PATH, index=False)
        X_test.to_csv(TEST_PATH,   index=False)
        y_train.to_csv(Y_TRAIN_PATH, index=False, header=True)
        y_test.to_csv(Y_TEST_PATH,   index=False, header=True)
        joblib.dump(scaler, SCALER_PATH)
    return X_train, X_test, y_train, y_test, scaler, list(X.columns)


def prepare_single_input(raw_dict, scaler, feature_names):
    """Transform one user-submitted dict into a model-ready DataFrame row."""
    row = pd.DataFrame([raw_dict])
    row["max_hr_reserve"] = row["thalach"] - (220 - row["age"])
    row["age_group"] = pd.cut(
        row["age"], bins=[0, 45, 55, 65, 120],
        labels=["young_adult", "middle_aged", "senior", "elderly"], right=True,
    ).astype(str)
    row["bp_category"] = pd.cut(
        row["trestbps"], bins=[0, 119, 139, 300],
        labels=["normal", "elevated", "high"], right=True,
    ).astype(str)
    ohe_cols = ["cp", "restecg", "slope", "thal", "age_group", "bp_category"]
    for col in ohe_cols:
        row[col] = row[col].astype(str)
    row = pd.get_dummies(row, columns=ohe_cols, drop_first=True, dtype=int)
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names]
    scale_cols = [c for c in SCALE_FEATURES if c in row.columns]
    row[scale_cols] = scaler.transform(row[scale_cols])
    return row


def run_pipeline(save=True):
    """Execute all preprocessing steps end-to-end."""
    print("\n" + "="*55 + "\n  PREPROCESSING PIPELINE\n" + "="*55)
    df = load_raw()
    df = binarize_target(df)
    df = fix_dtypes(df)
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = cap_outliers(df)
    df = engineer_features(df)
    df_enc = encode_categoricals(df)
    if save:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_enc.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"  Saved → {PROCESSED_DATA_PATH}")
    result = split_and_scale(df_enc, save=save)
    print(f"  Train: {result[0].shape}  |  Test: {result[1].shape}")
    print(f"  Features: {len(result[5])}")
    print("="*55 + "\n  Preprocessing complete.\n")
    return result
