import pandas as pd


def load_truthfulqa(filepath="data/TruthfulQA.csv"):
    df = pd.read_csv(filepath)
    return df


def load_validation(filepath="data/generation_validation.csv"):
    df = pd.read_csv(filepath)
    return df
