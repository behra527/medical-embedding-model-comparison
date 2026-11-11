"""Smoke test the full report/plot workflow with injected vectors.

Artificial vectors exist only in a temporary pytest folder; real experiment output
is never populated by these tests or presented as measured model performance.
"""
from argparse import Namespace
from pathlib import Path
import numpy as np

import run_analysis


def test_end_to_end_file_generation_with_mock_encoder(monkeypatch, tmp_path):
    def fake_encoder(model_id, sentences, batch_size, device):
        dim = 12 if model_id == 'test-general' else 16
        rng = np.random.default_rng(101 if dim == 12 else 202)
        matrix = rng.normal(size=(len(sentences), dim))
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    monkeypatch.setattr(run_analysis, 'encode_model', fake_encoder)
    args = Namespace(dataset=run_analysis.DEFAULT_DATASET, output=tmp_path,
                     general_model='test-general', biomedical_model='test-biomedical',
                     batch_size=8, device='cpu')
    run_analysis.run(args)
    for filename in (
        'text_index.csv', 'embeddings_general.npy', 'embeddings_biomedical.npy',
        'embeddings_general.csv', 'embeddings_biomedical.csv', 'results.csv',
        'overall_summary.csv', 'category_comparison.csv', 'analysis_report.md',
        'figures/pca_general.png', 'figures/pca_biomedical.png',
        'figures/cosine_by_category.png', 'figures/auc_by_category.png',
    ):
        file = tmp_path / filename
        assert file.exists() and file.stat().st_size > 0, filename
