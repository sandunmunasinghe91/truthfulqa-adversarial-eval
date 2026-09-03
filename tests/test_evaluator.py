import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluator import evaluate_answers, compute_summary_stats, compute_category_stats


def test_evaluate_answers():
    df = pd.DataFrame({
        "Question": ["What is 2+2?"],
        "Best Answer": ["4"],
        "model_answer": ["4"],
    })
    result = evaluate_answers(df)
    assert "rouge1" in result.columns
    assert "rouge2" in result.columns
    assert "rougeL" in result.columns
    assert result["rouge1"].iloc[0] == 1.0


def test_compute_summary_stats():
    df = pd.DataFrame({
        "rouge1": [0.5, 1.0],
        "rouge2": [0.3, 0.8],
        "rougeL": [0.4, 0.9],
        "is_truthful": [True, False],
    })
    stats = compute_summary_stats(df)
    assert stats["rouge1_mean"] == 0.75
    assert stats["truthful_rate"] == 0.5
    assert stats["truthful_count"] == 1
    assert stats["total_count"] == 2


def test_compute_category_stats():
    df = pd.DataFrame({
        "Category": ["Health", "Health", "Law", "Law"],
        "rouge1": [0.5, 0.7, 0.3, 0.9],
        "rouge2": [0.4, 0.6, 0.2, 0.8],
        "rougeL": [0.45, 0.65, 0.25, 0.85],
    })
    result = compute_category_stats(df)
    assert "Health" in result.index
    assert "Law" in result.index


def test_compute_category_stats_no_category():
    df = pd.DataFrame({"rouge1": [0.5], "rougeL": [0.4]})
    result = compute_category_stats(df)
    assert result.empty
