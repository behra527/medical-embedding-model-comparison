from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.metrics import paired_cosine, ranking_auc, category_auc, validate_dataset
from run_analysis import make_category_comparison

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_has_40_balanced_pairs():
    df = validate_dataset(pd.read_csv(ROOT / 'data' / 'medical_sentence_pairs.csv'))
    assert len(df) == 40
    assert df.groupby('category').size().to_dict() == {
        'synonyms': 10, 'abbreviations': 10, 'negation': 10, 'numerical_difference': 10,
    }
    assert df['pair_id'].nunique() == len(df)


def test_cosine_uses_correct_rows_and_handles_negatives():
    result = paired_cosine(np.array([[1, 0], [1, 0], [1, 0]]),
                           np.array([[1, 0], [0, 1], [-1, 0]]))
    np.testing.assert_allclose(result, [1, 0, -1])


def test_cosine_rejects_zero_vector():
    with pytest.raises(ValueError):
        paired_cosine(np.array([[0, 0]]), np.array([[1, 0]]))


def test_auc_perfect_reverse_and_tie():
    assert ranking_auc(np.array([0.8, 0.9]), np.array([0.1, 0.2])) == 1.0
    assert ranking_auc(np.array([0.1]), np.array([0.9])) == 0.0
    assert ranking_auc(np.array([0.5]), np.array([0.5])) == 0.5


def test_auc_ranks_category_against_opposite_class():
    df = validate_dataset(pd.read_csv(ROOT / 'data' / 'medical_sentence_pairs.csv'))
    df['general_cosine'] = df.same_meaning.astype(float) * 0.6 + 0.2
    df['biomedical_cosine'] = 0.5
    for category in ('synonyms', 'abbreviations', 'negation', 'numerical_difference'):
        assert category_auc(df, 'general_cosine', category) == 1.0
        assert category_auc(df, 'biomedical_cosine', category) == 0.5
    result = make_category_comparison(df)
    assert result['higher_category_auc'].eq('General MiniLM').all()
