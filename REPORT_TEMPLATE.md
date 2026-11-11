# Medical Text Embedding Comparison — Submission Guide

## Abstract

This project compares a general sentence embedding model (all-MiniLM-L6-v2) and a biomedical embedding model (NeuML/pubmedbert-base-embeddings) using 40 synthetic medical sentence pairs across synonyms, abbreviations, negation, and numerical differences. Each model maps each input sentence to a numerical vector. Corresponding vectors are compared using cosine similarity, and labeled pair rankings are evaluated using AUC. PCA provides visualizations of the embedding spaces.

## Expected experiment steps

1. Load the labeled dataset and check the 10+10+10+10 group balance.
2. Load both pretrained SentenceTransformer models, without fine-tuning.
3. Generate normalized numerical embeddings of all unique sentences.
4. Calculate a cosine similarity for each of the 40 sentence pairs under both models.
5. Calculate descriptive group means and ranking AUC (overall and per-category).
6. Generate two PCA plots, score distribution and category-AUC charts.
7. Interpret results, including failures to separate clinical contradictions from paraphrases.

## Results

**Do not fill in or submit fabricated numbers.** Run `python run_analysis.py` and use the auto-generated `outputs/analysis_report.md` as your results write-up. The resulting report contains numerical results and which model wins each type.

## Discussion questions

- Do equivalent medical phrases rank above clinically different statements?
- Does biomedical-specific training improve abbreviation recognition?
- Are models overly similar on negated statements or changed medication/laboratory numbers?
- Does PCA show paired synonyms close to each other? Where does its 2D projection obscure relationships?
- What could be improved using real annotated clinical datasets and task-specific contradiction detection?

## Conclusion

The final model recommendation must be based on measured results from the downloaded pretrained models, not on the assumption that the biomedical model is automatically superior. This is an educational comparison, not a diagnostic application.
