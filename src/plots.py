"""Static evaluation charts; PCA is fit separately for each vector space."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

COLORS = {
    "synonyms": "#1f77b4",
    "abbreviations": "#2ca02c",
    "negation": "#d62728",
    "numerical_difference": "#9467bd",
}


def plot_pca(df: pd.DataFrame, sentence_to_idx: dict[str, int], vectors: np.ndarray,
             model_label: str, target: Path) -> None:
    """Plots A/B endpoints and pair edges. Distances in 2D are illustrative only."""
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(11, 8))
    for _, row in df.iterrows():
        a, b = sentence_to_idx[row.sentence_1], sentence_to_idx[row.sentence_2]
        color = COLORS[row.category]
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
                color=color, alpha=0.30, linewidth=1.2, zorder=1)
        ax.scatter([xy[a, 0]], [xy[a, 1]], color=color, marker="o", s=35, alpha=0.8, zorder=2)
        ax.scatter([xy[b, 0]], [xy[b, 1]], color=color, marker="x", s=45, alpha=0.8, zorder=2)
    category_legend = [Line2D([0], [0], marker="o", color="w", label=c.replace("_", " ").title(),
                              markerfacecolor=col, markersize=9) for c, col in COLORS.items()]
    category_legend += [Line2D([0], [0], marker="o", color="#333333", lw=0, label="Sentence A"),
                        Line2D([0], [0], marker="x", color="#333333", lw=0, label="Sentence B")]
    ax.legend(handles=category_legend, loc="best", fontsize=9, framealpha=0.94)
    ax.set_title(f"PCA of Medical Sentence Embeddings — {model_label}")
    ax.set_xlabel(f"Principal component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"Principal component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.grid(alpha=0.15)
    fig.text(0.5, 0.015, "Each line connects a labeled pair; 2D PCA distance is not the original cosine similarity.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(target, dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(df: pd.DataFrame, target: Path) -> None:
    categories = list(COLORS)
    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = [-0.16, 0.16]
    cols = ("general_cosine", "biomedical_cosine")
    labels = ("General MiniLM", "Biomedical PubMedBERT")
    for i, category in enumerate(categories):
        for j, (col, label) in enumerate(zip(cols, labels)):
            vals = df.loc[df.category == category, col].to_numpy()
            x = i + offsets[j]
            ax.scatter(np.full(len(vals), x), vals, c=COLORS[category],
                       alpha=0.32 if j == 0 else 0.75, marker="o" if j == 0 else "^", s=35)
            ax.plot([x - 0.08, x + 0.08], [np.mean(vals)] * 2,
                    color="#202020", linewidth=2.5)
    ax.set_xticks(range(len(categories)), [c.replace("_", "\n").title() for c in categories])
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine Similarity by Category (horizontal dash = group mean)")
    ax.set_ylim(-1.02, 1.02)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="gray", lw=0, label="General MiniLM"),
                       Line2D([0], [0], marker="^", color="gray", lw=0, label="Biomedical PubMedBERT")])
    fig.tight_layout()
    fig.savefig(target, dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_category_auc(category_df: pd.DataFrame, target: Path) -> None:
    categories = category_df["category"].tolist()
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(10, 5.7))
    ax.bar(x - 0.18, category_df["general_category_auc"], 0.36, label="General MiniLM", color="#4e79a7")
    ax.bar(x + 0.18, category_df["biomedical_category_auc"], 0.36, label="Biomedical PubMedBERT", color="#59a14f")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="Chance-level ranking")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, [c.replace("_", "\n").title() for c in categories])
    ax.set_ylabel("Category-vs-opposite-class ranking AUC")
    ax.set_title("How Reliably Each Model Separates Same vs Different Medical Meaning")
    ax.legend()
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(target, dpi=175, bbox_inches="tight")
    plt.close(fig)
