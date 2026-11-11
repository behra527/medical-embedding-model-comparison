# Comparing General and Biomedical Text Embedding Models

A complete, reproducible Python assignment. The program converts synthetic medical sentences to numerical vectors using **two actual pretrained transformer embedding models**, scores sentence pairs with cosine similarity, evaluates performance by type and plots PCA visualizations.

## Models

| Model | Hugging Face ID | Vector size | Why selected |
|---|---|---:|---|
| General purpose | `sentence-transformers/all-MiniLM-L6-v2` | 384 | General sentence semantic similarity |
| Biomedical | `NeuML/pubmedbert-base-embeddings` | 768 | Sentence embeddings fine-tuned on PubMed title/abstract data |

Sources: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 and https://huggingface.co/NeuML/pubmedbert-base-embeddings . Check each model's license if you plan to distribute or commercially deploy a derivative.

## Dataset

`data/medical_sentence_pairs.csv` contains **40 pairs**, balanced among:

- Synonyms (10, same meaning): e.g., `hypertension` / `high blood pressure`
- Medical abbreviations (10, same meaning): e.g., `ECG` / `electrocardiogram`
- Negation (10, different meaning): e.g., `pneumonia` / `no pneumonia`
- Numerical differences (10, different meaning): e.g., `90 mg/dL` / `350 mg/dL`

Each row contains ID, category, both sentences, a binary gold label, and a clinical rationale. All examples are synthetic, not real patient data. The labels are hand-authored and may require domain-expert review for serious research.

## Setup (Windows PowerShell)

Install **Python 3.10 or later** and keep an internet connection for the first model download. From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python run_analysis.py --device cpu --batch-size 8
```

If PowerShell blocks activation, run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe run_analysis.py --device cpu` directly instead.

You can omit `--device cpu` to allow automatic device selection; a compatible CUDA GPU can accelerate encoding. First-run model downloads may require **several GB** of free storage (including dependencies and caches), and the biomedical model is larger. The CPU-only run works without a dedicated GPU but is slower.

**Google Colab / Linux**:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_analysis.py
```

## Generated files (after a real inference run)

```text
outputs/
  text_index.csv                  # IDs and order of all unique source sentences
  embeddings_general.npy         # real 384D general vectors
  embeddings_biomedical.npy      # real 768D biomedical vectors
  embeddings_general.csv         # vectors exposed as numerical columns
  embeddings_biomedical.csv
  results.csv                    # each pair, label and both cosine scores
  overall_summary.csv            # dimensions, means, gaps, overall AUC
  category_comparison.csv        # four category AUCs + winner
  analysis_report.md             # automatically generated research report
  figures/
    pca_general.png
    pca_biomedical.png
    cosine_by_category.png
    auc_by_category.png
```

To open an optional visual browser dashboard **after** the model run:

```powershell
streamlit run app.py
```

## How similarity is calculated

For each sentence, the model produces a dense vector. Cosine similarity is:

`cos(A,B) = (A · B) / (||A|| ||B||)`

The general and biomedical vectors are **not** compared directly because they live in different vector spaces and have different dimensions. Only within-model cosine scores on a given sentence pair are meaningful. Higher similarity is generally expected for synonyms and abbreviation expansions; lower similarity is desired for contradictory negation and numerical differences.

### How to judge which model is better

Raw average cosine values alone are **not** a fair cross-model competition: one model can produce higher similarities for *all* inputs. The primary evaluation is **within-model ranking AUC**: given a labeled equivalent pair and a labeled non-equivalent pair, how often does the model assign a higher score to the equivalent one? (Ties count 0.5.) Higher is better; 0.5 is chance-like ranking and 1.0 is perfect on the supplied pairs.

For each category, category AUC compares its 10 pairs against **all 20 pairs with the opposite gold label**. Compare the category AUCs to select a winner on this small test set. `category_comparison.csv` and `analysis_report.md` calculate this automatically. Mean cosine scores are displayed as diagnostic observations, not used to declare a winner.

PCA separately projects each model's normalized vectors to 2 dimensions to explore grouping and how corresponding sentences are placed. PCA positions from one model **cannot** be compared directly with positions of the other model; 2D distance does not equal cosine similarity in the original high-dimensional space.

## Limitations

- Embeddings often score sentences with `has` and `does not have` very similarly despite opposing clinical statements; no high cosine value proves logical agreement.
- Models may not accurately encode lab values, medication doses, abnormal ranges, or arithmetic. These tests do not validate clinical safety.
- 40 synthetic examples are a small pedagogical benchmark, not a clinical assessment.
- No experimental accuracy can be claimed until the two model weights have been downloaded and this program actually runs.

## Testing

`python -m pytest -q` validates the dataset, cosine calculation, zero-vector rejection, AUC ties, and category winner logic, without downloading pretrained model files.

## Command options

```bash
python run_analysis.py --help
python run_analysis.py --dataset data/medical_sentence_pairs.csv --output outputs --batch-size 4 --device cpu
```

The optional notebook in `notebooks/` illustrates the experiment workflow for study/submission. Use `run_analysis.py` as the authoritative reproducible experiment.
