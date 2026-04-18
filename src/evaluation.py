"""src/evaluation.py — Comprehensive model evaluation: metrics, plots, CV, reports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report,
)
from sklearn.model_selection import cross_val_score

from src.config import FIGURES_DIR, COLORS

_PAL = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
_MODEL_COLORS = {
    "Logistic Regression": COLORS["primary"],
    "Random Forest":       COLORS["secondary"],
    "SVM":                 COLORS["accent"],
}


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),                    4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0),  4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0),     4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0),         4),
        "AUC":       round(roc_auc_score(y_test, y_proba[:, 1]),              4),
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }


def metrics_table(results):
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    df = pd.DataFrame([{k: v for k, v in r.items() if k in cols} for r in results])
    return df.sort_values("Recall", ascending=False).reset_index(drop=True)


def cross_validate_all(models_dict, X_train, y_train, cv=5):
    rows = []
    for name, m in models_dict.items():
        rec = cross_val_score(m.model, X_train, y_train, cv=cv, scoring="recall")
        auc = cross_val_score(m.model, X_train, y_train, cv=cv, scoring="roc_auc")
        rows.append({
            "Model":            name,
            "CV Recall Mean":   round(rec.mean(), 4),
            "CV Recall Std":    round(rec.std(),  4),
            "CV AUC Mean":      round(auc.mean(), 4),
            "CV AUC Std":       round(auc.std(),  4),
        })
    return pd.DataFrame(rows)


# ── Confusion matrices ────────────────────────────────────────────────────────
def plot_confusion_matrices(results, y_test, save=True):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            linewidths=0.5, ax=ax,
        )
        ax.set_title(res["Model"], fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        # Save individual CM too
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"],
                    linewidths=0.5, ax=ax2)
        ax2.set_title(res["Model"], fontsize=11, fontweight="bold")
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
        fig2.tight_layout()
        slug = res["Model"].lower().replace(" ", "_")
        fig2.savefig(FIGURES_DIR / f"cm_{slug}.png", dpi=130, bbox_inches="tight")
        plt.close(fig2)

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── ROC curves ────────────────────────────────────────────────────────────────
def plot_roc_curves(results, y_test, save=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.50)")
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"][:, 1])
        color = _MODEL_COLORS.get(res["Model"], "#888")
        ax.plot(fpr, tpr, lw=2.2, color=color,
                label=f"{res['Model']}  (AUC={res['AUC']:.3f})")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Metrics comparison bar chart ──────────────────────────────────────────────
def plot_metrics_comparison(df_metrics, save=True):
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    df_m = df_metrics.set_index("Model")[metric_cols].T
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_cols))
    width = 0.25
    for i, (model_name, col_data) in enumerate(df_m.items()):
        color = _MODEL_COLORS.get(model_name, "#888")
        bars = ax.bar(x + i * width, col_data.values, width,
                      label=model_name, color=color, alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_cols, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Feature importance ────────────────────────────────────────────────────────
def plot_feature_importance(rf_model, feature_names, top_n=15, save=True):
    imp = pd.Series(rf_model.model.feature_importances_, index=feature_names)
    imp = imp.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(imp.index, imp.values, color=COLORS["secondary"], alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=8.5)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title(f"Random Forest — Top {top_n} Feature Importances",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Classification report ─────────────────────────────────────────────────────
def print_full_report(results, y_test):
    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORTS")
    print("=" * 55)
    for res in results:
        print(f"\n{'─'*40}\n  {res['Model']}\n{'─'*40}")
        print(classification_report(y_test, res["y_pred"],
                                    target_names=["No Disease", "Disease"]))
