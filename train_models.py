"""
train_models.py — Master training script.
Run once before launching the apps:  python train_models.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import run_pipeline
from src.models import RFModel, LRModel, SVMModel, build_kmeans, build_pca
from src.evaluation import (
    compute_metrics, metrics_table, cross_validate_all,
    plot_confusion_matrices, plot_roc_curves,
    plot_metrics_comparison, plot_feature_importance,
    print_full_report,
)
from src.config import MODELS_DIR, FIGURES_DIR, BEST_MODEL_NAME


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  HEART DISEASE PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Preprocessing ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, feature_names = run_pipeline(save=True)

    # Save feature names (required for inference in Streamlit app)
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # ── 2. Supervised models ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING SUPERVISED MODELS")
    print("=" * 60)
    models = {
        "Logistic Regression": LRModel(),
        "Random Forest":       RFModel(),
        "SVM":                 SVMModel(),
    }
    for name, m in models.items():
        print(f"\n  Training {name}...")
        m.build().fit(X_train, y_train)
        m.save()

    # ── 3. Unsupervised models ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FITTING UNSUPERVISED MODELS")
    print("=" * 60)
    proc_df = pd.read_csv(MODELS_DIR.parent / "data" / "processed" / "processed_data.csv")
    X_num   = proc_df.drop(columns=["target"]).select_dtypes(include="number").fillna(0)

    pca_model  = build_pca(n_components=2)
    X_pca      = pca_model.fit_transform(X_num)
    kmeans_model = build_kmeans(n_clusters=2)
    kmeans_model.fit(X_pca)

    joblib.dump(pca_model,    MODELS_DIR / "pca.pkl")
    joblib.dump(kmeans_model, MODELS_DIR / "kmeans.pkl")

    from sklearn.metrics import silhouette_score
    sil = silhouette_score(X_pca, kmeans_model.labels_)
    ev  = pca_model.explained_variance_ratio_
    print(f"  K-Means  k=2  Silhouette Score : {sil:.4f}")
    print(f"  PCA  explained variance (PC1+PC2): {sum(ev):.1%}")

    # ── 4. Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION (held-out test set, n=61)")
    print("=" * 60)
    results = [compute_metrics(name, m, X_test, y_test) for name, m in models.items()]
    df_metrics = metrics_table(results)

    print("\n  Metrics Table:")
    print(df_metrics[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]].to_string(index=False))

    # ── 5. Cross-validation ───────────────────────────────────────────────────
    print("\n  5-Fold Cross-Validation (train set):")
    df_cv = cross_validate_all(models, X_train, y_train, cv=5)
    print(df_cv.to_string(index=False))

    # ── 6. Figures ────────────────────────────────────────────────────────────
    print("\n  Generating evaluation figures...")
    plot_confusion_matrices(results, y_test, save=True)
    plot_roc_curves(results, y_test, save=True)
    plot_metrics_comparison(df_metrics, save=True)
    plot_feature_importance(models["Random Forest"], feature_names, top_n=15, save=True)
    print_full_report(results, y_test)

    # ── 7. Save metadata CSV ──────────────────────────────────────────────────
    df_metrics.to_csv(MODELS_DIR / "metrics_summary.csv", index=False)
    df_cv.to_csv(MODELS_DIR / "cv_summary.csv", index=False)

    best = df_metrics.iloc[0]["Model"]
    print("\n" + "=" * 60)
    print(f"  Best model by Recall : {best}")
    print(f"  All artifacts → models/")
    print(f"  All figures   → figures/")
    print("\n  Next steps:")
    print("    streamlit run app/main.py")
    print("    streamlit run dashboard/dashboard.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
