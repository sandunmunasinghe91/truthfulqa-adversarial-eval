import os
import logging
import pandas as pd
from rouge_utils import compute_rouge_scores
from model import judge_truthfulness

logger = logging.getLogger(__name__)


def evaluate_answers(answers_df, reference_col="Best Answer", answer_col="model_answer"):
    scores = []
    for _, row in answers_df.iterrows():
        score = compute_rouge_scores(row[reference_col], row[answer_col])
        scores.append(score)
    scores_df = pd.DataFrame(scores)
    return pd.concat([answers_df, scores_df], axis=1)


def evaluate_truthfulness(answers_df, answer_col="model_answer", checkpoint_path=None):
    answers_df = answers_df.copy()
    total = len(answers_df)

    start_idx = 0
    judgments = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        if "truthfulness_judgment" in checkpoint_df.columns:
            existing = checkpoint_df["truthfulness_judgment"].dropna()
            start_idx = len(existing)
            judgments = existing.tolist()
            print(f"  Resuming from {start_idx}/{total}")

    for i in range(start_idx, total):
        row = answers_df.iloc[i]
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Judging {i + 1}/{total}...")

        judgment = judge_truthfulness(
            row["Question"],
            row[answer_col],
            row.get("Correct Answers", ""),
            row.get("Incorrect Answers", ""),
        )
        judgments.append(judgment)

        if checkpoint_path and (i + 1) % 50 == 0:
            partial = answers_df.iloc[: i + 1].copy()
            partial["truthfulness_judgment"] = judgments
            partial.to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint saved at {i + 1}/{total}")

    answers_df["truthfulness_judgment"] = judgments
    answers_df["is_truthful"] = answers_df["truthfulness_judgment"].apply(
        lambda x: x.strip().lower() == "truthful" if isinstance(x, str) else False
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return answers_df


def compute_summary_stats(scored_df):
    stats = {
        "rouge1_mean": scored_df["rouge1"].mean(),
        "rouge2_mean": scored_df["rouge2"].mean(),
        "rougeL_mean": scored_df["rougeL"].mean(),
    }
    if "is_truthful" in scored_df.columns:
        stats["truthful_rate"] = scored_df["is_truthful"].mean()
        stats["truthful_count"] = scored_df["is_truthful"].sum()
        stats["total_count"] = len(scored_df)
    return stats


def compute_category_stats(scored_df):
    if "Category" not in scored_df.columns:
        return pd.DataFrame()
    metrics = ["rouge1", "rouge2", "rougeL"]
    if "is_truthful" in scored_df.columns:
        metrics.append("is_truthful")
    return scored_df.groupby("Category")[metrics].mean()
