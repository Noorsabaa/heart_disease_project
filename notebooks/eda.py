"""
notebooks/eda.py
----------------
Exploratory Data Analysis — Cleveland Heart Disease Dataset.
Run standalone:  python notebooks/eda.py
Or paste each section as cells into Google Colab / Jupyter.
All figures saved to figures/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import RAW_DATA_PATH, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
PAL = {"No Disease": "#1D9E75", "Disease": "#E24B4A"}

# ── Cell 1: Load & binarize ───────────────────────────────────────────────────
df = pd.read_csv(RAW_DATA_PATH)
df["target_bin"]   = (df["target"] > 0).astype(int)
df["target_label"] = df["target_bin"].map({0: "No Disease", 1: "Disease"})
print(f"Shape: {df.shape}")
print(df.head())

# ── Cell 2: Class balance ─────────────────────────────────────────────────────
print("\n=== CLASS BALANCE ===")
print(df["target_bin"].value_counts())
print(f"Disease rate: {df['target_bin'].mean():.1%}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df["target_label"].value_counts().plot.pie(
    ax=ax[0], autopct="%1.1f%%",
    colors=["#1D9E75", "#E24B4A"], startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
ax[0].set_title("Class Distribution", fontsize=13); ax[0].set_ylabel("")
sns.histplot(data=df, x="age", hue="target_label", bins=25,
             alpha=0.72, ax=ax[1], palette=PAL)
ax[1].set_title("Age Distribution by Class", fontsize=13)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "eda_class_balance.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved → eda_class_balance.png")

# ── Cell 3: Descriptive stats ─────────────────────────────────────────────────
CONT = ["age", "trestbps", "chol", "thalach", "oldpeak"]
print("\n=== DESCRIPTIVE STATISTICS ===")
print(df[CONT].describe().round(2))
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# ── Cell 4: Distributions ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, feat in enumerate(CONT):
    ax = axes[i // 3][i % 3]
    sns.histplot(data=df, x=feat, hue="target_label", kde=True,
                 bins=25, alpha=0.65, ax=ax, palette=PAL)
    ax.set_title(feat, fontsize=11); ax.set_xlabel("")
axes[1][2].set_visible(False)
plt.suptitle("Feature Distributions by Target", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "eda_distributions.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved → eda_distributions.png")

# ── Cell 5: Box plots ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, feat in enumerate(CONT):
    sns.boxplot(data=df, x="target_label", y=feat, ax=axes[i],
                palette=PAL, width=0.5, linewidth=1.2)
    axes[i].set_title(feat, fontsize=11); axes[i].set_xlabel("")
plt.suptitle("Box Plots — Continuous Features by Target", fontsize=13)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "eda_boxplots.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved → eda_boxplots.png")

# ── Cell 6: Correlation heatmap ───────────────────────────────────────────────
num_df = df[CONT + ["ca", "sex", "fbs", "exang", "target_bin"]]
corr   = num_df.corr().round(3)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.4, square=True, ax=ax,
            mask=np.triu(np.ones_like(corr, dtype=bool)))
ax.set_title("Pearson Correlation Heatmap", fontsize=14, pad=12)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("\n=== FEATURE CORRELATIONS WITH TARGET ===")
print(corr["target_bin"].drop("target_bin").sort_values(ascending=False))
print("Saved → eda_correlation_heatmap.png")

# ── Cell 7: Categorical features ─────────────────────────────────────────────
cat_cols   = ["cp", "restecg", "slope", "thal", "sex", "exang"]
cat_labels = {
    "cp":      {1.0:"Typical",2.0:"Atypical",3.0:"Non-Anginal",4.0:"Asymp."},
    "restecg": {0.0:"Normal",1.0:"ST-T",2.0:"LVH"},
    "slope":   {1.0:"Up",2.0:"Flat",3.0:"Down"},
    "thal":    {3.0:"Normal",6.0:"Fixed",7.0:"Reversible"},
    "sex":     {0.0:"Female",1.0:"Male"},
    "exang":   {0.0:"No",1.0:"Yes"},
}
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, feat in enumerate(cat_cols):
    ax = axes[i // 3][i % 3]
    temp = df.copy()
    temp[feat] = temp[feat].map(cat_labels.get(feat, {})).fillna(temp[feat].astype(str))
    ct = pd.crosstab(temp[feat], temp["target_label"], normalize="index")
    ct[["No Disease", "Disease"]].plot.bar(
        ax=ax, color=["#1D9E75", "#E24B4A"], edgecolor="none",
        width=0.6, stacked=True,
    )
    ax.set_title(feat, fontsize=11)
    ax.set_ylabel("Proportion"); ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=8)
plt.suptitle("Categorical Features — Disease Proportion", fontsize=13)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "eda_categorical.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved → eda_categorical.png")

# ── Cell 8: Statistical tests ─────────────────────────────────────────────────
print("\n=== T-TESTS: DISEASE vs NO DISEASE ===")
g0, g1 = df[df["target_bin"]==0], df[df["target_bin"]==1]
for feat in CONT:
    t, p = stats.ttest_ind(g0[feat].dropna(), g1[feat].dropna())
    sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {feat:12s}: t={t:7.3f}  p={p:.4f}  {sig}")

# ── Cell 9: Pairplot ──────────────────────────────────────────────────────────
g = sns.pairplot(
    df[CONT + ["target_label"]], hue="target_label",
    diag_kind="kde", palette=PAL,
    plot_kws={"alpha": 0.45, "s": 22},
    diag_kws={"fill": True, "alpha": 0.45},
)
g.figure.suptitle("Pairplot — Continuous Features", y=1.01, fontsize=14)
g.figure.savefig(FIGURES_DIR / "eda_pairplot.png", dpi=100, bbox_inches="tight")
plt.close(g.figure)
print("Saved → eda_pairplot.png")

print(f"\n=== EDA COMPLETE — all figures saved to: {FIGURES_DIR} ===")
