"""Model-agnostic similarity metrics and dataset validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ("synonyms", "abbreviations", "negation", "numerical_difference")
POSITIVE_CATEGORIES = frozenset(("synonyms", "abbreviations"))
REQUIRED_COLUMNS = ("pair_id", "category", "sentence_1", "sentence_2", "same_meaning")


def validate_dataset(df: pd.DataFrame, min_pairs: int = 25) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if len(df) < min_pairs:
        raise ValueError(f"Expected at least {min_pairs} sentence pairs; got {len(df)}")
    if df["pair_id"].isna().any() or df["pair_id"].duplicated().any():
        raise ValueError("pair_id must be present and unique")
    if df["category"].isna().any() or set(df["category"]) != set(CATEGORIES):
        raise ValueError(f"Dataset must contain exactly these categories: {CATEGORIES}")
    if df[["sentence_1", "sentence_2"]].isna().any().any():
        raise ValueError("All sentences must be present")
    for col in ("sentence_1", "sentence_2"):
        if df[col].map(lambda x: not isinstance(x, str) or not x.strip()).any():
            raise ValueError(f"{col} contains empty or non-text values")
    if not df["same_meaning"].isin([0, 1]).all():
        raise ValueError("same_meaning must be 0 or 1")
    for category in CATEGORIES:
        expected = int(category in POSITIVE_CATEGORIES)
        if not df.loc[df.category == category, "same_meaning"].eq(expected).all():
            raise ValueError(f"Labels conflict with category: {category}")
    return df.copy()


def paired_cosine(vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
    """Cosine similarity for corresponding rows, without assuming normalization."""
    a = np.asarray(vectors_a, dtype=np.float64)
    b = np.asarray(vectors_b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Inputs must be 2D arrays with the same shape")
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Cannot compare zero vectors")
    return np.clip(np.sum(a * b, axis=1) / norms, -1.0, 1.0)


def ranking_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """P(score for equivalent pair > score for non-equivalent pair), ties=0.5.

    Unlike a fixed cosine threshold, this is a within-model rank metric and
    is suitable for comparing how well two embedding models separate labels.
    """
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("AUC needs at least one positive and negative example")
    comparisons = pos[:, None] - neg[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def category_auc(df: pd.DataFrame, score_col: str, category: str) -> float:
    """Category vs all examples with the opposite semantic equivalence label."""
    group = df[df.category == category]
    expected = int(category in POSITIVE_CATEGORIES)
    reference = df[df.same_meaning != expected]
    if expected:
        return ranking_auc(group[score_col].to_numpy(), reference[score_col].to_numpy())
    return ranking_auc(reference[score_col].to_numpy(), group[score_col].to_numpy())


def summary_row(df: pd.DataFrame, model_label: str, model_id: str, dim: int, score_col: str) -> dict:
    pos = df.loc[df.same_meaning == 1, score_col].to_numpy()
    neg = df.loc[df.same_meaning == 0, score_col].to_numpy()
    return {
        "model": model_label,
        "model_id": model_id,
        "embedding_dimensions": dim,
        "pairs": len(df),
        "mean_equivalent_cosine": float(np.mean(pos)),
        "mean_non_equivalent_cosine": float(np.mean(neg)),
        "separation_gap": float(np.mean(pos) - np.mean(neg)),
        "overall_roc_auc": ranking_auc(pos, neg),
    }
