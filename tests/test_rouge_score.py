import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rouge_utils import compute_rouge_scores


def test_identical_strings():
    scores = compute_rouge_scores("the cat sat on the mat", "the cat sat on the mat")
    assert scores["rouge1"] == 1.0
    assert scores["rouge2"] == 1.0
    assert scores["rougeL"] == 1.0


def test_different_strings():
    scores = compute_rouge_scores("the cat sat on the mat", "dogs run in the park")
    assert scores["rouge1"] < 1.0
    assert scores["rougeL"] < 1.0


def test_empty_or_non_string():
    scores = compute_rouge_scores(None, "hello")
    assert scores["rouge1"] == 0.0
    assert scores["rouge2"] == 0.0
    assert scores["rougeL"] == 0.0


def test_partial_overlap():
    scores = compute_rouge_scores("the cat sat", "the cat ran")
    assert 0 < scores["rouge1"] < 1.0
