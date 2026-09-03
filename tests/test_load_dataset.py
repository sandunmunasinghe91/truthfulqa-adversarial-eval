import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from load_dataset import load_truthfulqa

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "TruthfulQA.csv")


def test_load_dataset():
    df = load_truthfulqa(DATA_PATH)
    assert len(df) > 0
    assert "Question" in df.columns
    assert "Best Answer" in df.columns
    assert "Category" in df.columns


def test_dataset_has_expected_columns():
    df = load_truthfulqa(DATA_PATH)
    expected = ["Type", "Category", "Question", "Best Answer", "Correct Answers", "Incorrect Answers"]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


def test_no_empty_questions():
    df = load_truthfulqa(DATA_PATH)
    assert df["Question"].notna().all()
    assert (df["Question"].str.strip() != "").all()
