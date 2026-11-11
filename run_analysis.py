"""Compare general vs biomedical sentence embeddings on 40 labeled medical pairs.

Usage:
    python run_analysis.py
    python run_analysis.py --device cpu --batch-size 8

Pretrained weights download on first use. No fabricated/model-free metrics are emitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from src.metrics import (
    CATEGORIES, category_auc, paired_cosine, ranking_auc,
    summary_row, validate_dataset,
)
from src.plots import plot_category_auc, plot_pca, plot_score_distributions

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "medical_sentence_pairs.csv"
DEFAULT_OUTPUT = ROOT / "outputs"
GENERAL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BIOMED_ID = "NeuML/pubmedbert-base-embeddings"
LABELS = {"general": "General MiniLM", "biomedical": "Biomedical PubMedBERT"}


def write_embedding_csv(path: Path, sentences: list[str], vectors: np.ndarray) -> None:
    cols = [f"dim_{i:03d}" for i in range(vectors.shape[1])]
    frame = pd.DataFrame(vectors, columns=cols)
    frame.insert(0, "sentence", sentences)
    frame.insert(0, "text_id", [f"T{i + 1:03d}" for i in range(len(sentences))])
    frame.to_csv(path, index=False)


def encode_model(model_id: str, sentences: list[str], batch_size: int, device: str | None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading: {model_id}", flush=True)
    model = SentenceTransformer(model_id, device=device) if device else SentenceTransformer(model_id)
    print(f"Encoding {len(sentences)} unique sentences ...", flush=True)
    vectors = model.encode(
        sentences, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    if not isinstance(vectors, np.ndarray) or vectors.shape[0] != len(sentences):
        raise RuntimeError(f"Unexpected embedding dimensions from {model_id}: {getattr(vectors, 'shape', None)}")
    if not np.isfinite(vectors).all():
        raise RuntimeError(f"Non-finite embeddings returned by {model_id}")
    return vectors


def make_category_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        group = df[df.category == category]
        g_auc = category_auc(df, "general_cosine", category)
        b_auc = category_auc(df, "biomedical_cosine", category)
        if abs(g_auc - b_auc) < 1e-12:
            winner = "Tie"
        else:
            winner = "Biomedical PubMedBERT" if b_auc > g_auc else "General MiniLM"
        rows.append({
            "category": category,
            "pair_count": len(group),
            "same_meaning": int(group.same_meaning.iloc[0]),
            "general_mean_cosine": float(group.general_cosine.mean()),
            "biomedical_mean_cosine": float(group.biomedical_cosine.mean()),
            "general_category_auc": g_auc,
            "biomedical_category_auc": b_auc,
            "auc_difference_biomed_minus_general": b_auc - g_auc,
            "higher_category_auc": winner,
        })
    return pd.DataFrame(rows)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def write_report(df: pd.DataFrame, summary: pd.DataFrame, categories: pd.DataFrame,
                 general_id: str, biomedical_id: str, output: Path) -> None:
    metrics = {r.model: r for r in summary.itertuples(index=False)}
    g, b = metrics[LABELS["general"]], metrics[LABELS["biomedical"]]
    overall_winner = (
        LABELS["biomedical"] if b.overall_roc_auc > g.overall_roc_auc else
        LABELS["general"] if g.overall_roc_auc > b.overall_roc_auc else "Tie"
    )
    lines = [
        "# Medical Sentence Embedding Models — Experimental Report", "",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}", "",
        "## Objective", "",
        "Evaluate sentence-level biomedical semantic equivalence using cosine similarity,",
        "comparing a general-purpose embedding model with a biomedical domain-trained model.", "",
        "## Models", "",
        f"- **General:** `{general_id}` ({g.embedding_dimensions} dimensions)",
        f"- **Biomedical:** `{biomedical_id}` ({b.embedding_dimensions} dimensions)", "",
        "Both models encode the same unique texts. All vectors are L2-normalized before scoring.",
        "The models are **not** fine-tuned on this evaluation dataset.", "",
        "## Dataset and gold labels", "",
        f"- {len(df)} hand-authored pairs across four balanced categories.",
        "- Synonyms and abbreviations: equivalent (label 1).",
        "- Negation and numerical differences: non-equivalent (label 0).",
        "- Examples are illustrative, synthetic and not patient records.", "",
        "## Evaluation method", "",
        "Cosine similarity is calculated for each original sentence pair.",
        "Within each model, overall AUC measures the fraction of comparisons in which a",
        "positive pair scores above a negative pair (ties count as half).",
        "Per-category AUC compares pairs from that category against **all** pairs of the opposite label.",
        "AUC is preferred to judging raw cosine scores across different models because score scales can differ.",
        "Higher AUC means better separation, but does not prove robust clinical understanding.", "",
        "## Overall results", "",
        "| Measure | General MiniLM | Biomedical PubMedBERT |", "|---|---:|---:|",
        f"| Embedding dimensions | {g.embedding_dimensions} | {b.embedding_dimensions} |",
        f"| Mean cosine, equivalent pairs | {_fmt(g.mean_equivalent_cosine)} | {_fmt(b.mean_equivalent_cosine)} |",
        f"| Mean cosine, non-equivalent pairs | {_fmt(g.mean_non_equivalent_cosine)} | {_fmt(b.mean_non_equivalent_cosine)} |",
        f"| Positive-negative mean gap | {_fmt(g.separation_gap)} | {_fmt(b.separation_gap)} |",
        f"| Overall ranking AUC | {_fmt(g.overall_roc_auc)} | {_fmt(b.overall_roc_auc)} |", "",
        f"**Overall top model by ranking AUC on this small dataset:** {overall_winner}.", "",
        "## Category-level comparison", "",
        "| Category | n | General mean cosine | Biomedical mean cosine | General AUC | Biomedical AUC | Better AUC |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in categories.itertuples(index=False):
        lines.append(
            f"| {r.category.replace('_', ' ').title()} | {r.pair_count} | "
            f"{_fmt(r.general_mean_cosine)} | {_fmt(r.biomedical_mean_cosine)} | "
            f"{_fmt(r.general_category_auc)} | {_fmt(r.biomedical_category_auc)} | {r.higher_category_auc} |"
        )
    lines += ["", "## Conclusion by category", ""]
    for r in categories.itertuples(index=False):
        lines.append(
            f"- **{r.category.replace('_', ' ').title()}:** {r.higher_category_auc} "
            f"(general AUC {_fmt(r.general_category_auc)}; biomedical AUC {_fmt(r.biomedical_category_auc)})."
        )
    lines += [
        "", "## Figures", "",
        "- `figures/pca_general.png` — PCA, general embeddings.",
        "- `figures/pca_biomedical.png` — PCA, biomedical embeddings.",
        "- `figures/cosine_by_category.png` — groupwise similarity distributions.",
        "- `figures/auc_by_category.png` — category-specific AUC comparisons.", "",
        "PCA is fit separately for each model because their embedding dimensions differ.",
        "Its axes/distances cannot be compared directly between plots. Two-dimensional PCA",
        "does not preserve every cosine similarity or prove semantic correctness.", "",
        "## Limitations", "",
        "1. Just 40 hand-authored pairs and fixed binary labels; results are exploratory.",
        "2. Some near-paraphrases and medical shorthand can be ambiguous in real clinical contexts.",
        "3. Embedding cosine alone is not a reliable clinical contradiction or dose-error detector.",
        "4. AUC on these few examples is not evidence of safety or real-world clinical utility.",
        "5. Future work: expert-annotated datasets, repeated paraphrases, confidence intervals,",
        "   contrastive hard negatives and external clinical evaluation.", "",
        "## Reproducibility", "",
        "See `results.csv`, `overall_summary.csv`, `category_comparison.csv`, saved vectors",
        "and `README.md`. Run `python run_analysis.py` on the included dataset.", "",
    ]
    (output / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    df = validate_dataset(pd.read_csv(args.dataset))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    # All sentences go through both models, in identical order.
    texts = list(dict.fromkeys(df.sentence_1.tolist() + df.sentence_2.tolist()))
    indices = {text: index for index, text in enumerate(texts)}
    print(f"Dataset: {len(df)} pairs, {len(texts)} unique sentences")
    index_frame = pd.DataFrame({"text_id": [f"T{i+1:03d}" for i in range(len(texts))], "sentence": texts})
    index_frame.to_csv(out / "text_index.csv", index=False)
    general = encode_model(args.general_model, texts, args.batch_size, args.device)
    biomedical = encode_model(args.biomedical_model, texts, args.batch_size, args.device)
    # Shape is allowed to differ between model spaces, but sentence counts must match.
    assert general.shape[0] == biomedical.shape[0] == len(texts)
    for label, matrix in (("general", general), ("biomedical", biomedical)):
        np.save(out / f"embeddings_{label}.npy", matrix)
        write_embedding_csv(out / f"embeddings_{label}.csv", texts, matrix)
    a_idx = df.sentence_1.map(indices).to_numpy()
    b_idx = df.sentence_2.map(indices).to_numpy()
    df["general_cosine"] = paired_cosine(general[a_idx], general[b_idx])
    df["biomedical_cosine"] = paired_cosine(biomedical[a_idx], biomedical[b_idx])
    df["biomedical_minus_general_raw_cosine"] = df.biomedical_cosine - df.general_cosine
    df.to_csv(out / "results.csv", index=False, float_format="%.6f")
    overall = pd.DataFrame([
        summary_row(df, LABELS["general"], args.general_model, general.shape[1], "general_cosine"),
        summary_row(df, LABELS["biomedical"], args.biomedical_model, biomedical.shape[1], "biomedical_cosine"),
    ])
    overall.to_csv(out / "overall_summary.csv", index=False, float_format="%.6f")
    comparison = make_category_comparison(df)
    comparison.to_csv(out / "category_comparison.csv", index=False, float_format="%.6f")
    plot_pca(df, indices, general, LABELS["general"], out / "figures" / "pca_general.png")
    plot_pca(df, indices, biomedical, LABELS["biomedical"], out / "figures" / "pca_biomedical.png")
    plot_score_distributions(df, out / "figures" / "cosine_by_category.png")
    plot_category_auc(comparison, out / "figures" / "auc_by_category.png")
    write_report(df, overall, comparison, args.general_model, args.biomedical_model, out)
    print("\n=== Overall results ===")
    print(overall.to_string(index=False))
    print("\n=== Per-category ranking AUC ===")
    print(comparison[["category", "general_category_auc", "biomedical_category_auc", "higher_category_auc"]].to_string(index=False))
    print(f"\nFinished. Open: {out.resolve() / 'analysis_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--general-model", default=GENERAL_ID)
    parser.add_argument("--biomedical-model", default=BIOMED_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cpu, cuda or mps; auto-detect if omitted")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
