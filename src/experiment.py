import os
import logging
import pandas as pd
from load_dataset import load_truthfulqa
from model import query_model
from prompts import load_prompt_template, format_prompt
from evaluator import evaluate_answers, evaluate_truthfulness, compute_summary_stats
from utils import ensure_dir, get_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROMPT_CONFIGS = {
    "baseline": "prompts/baseline.txt",
    "adversarial_force_wrong": "prompts/adversarial_force_wrong.txt",
    "adversarial_popular_myth": "prompts/adversarial_popular_myth.txt",
    "adversarial_conspiracy": "prompts/adversarial_conspiracy.txt",
}


def run_experiment(df, prompt_name, prompt_path, output_dir, judge=False):
    template = load_prompt_template(prompt_path)
    answers = []
    total = len(df)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{prompt_name}.csv")

    start_idx = 0
    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        start_idx = len(checkpoint_df)
        answers = checkpoint_df["model_answer"].tolist()
        logger.info(f"Resuming {prompt_name} from question {start_idx}/{total}")

    for i in range(start_idx, total):
        row = df.iloc[i]
        prompt = format_prompt(template, row["Question"])
        logger.info(f"[{prompt_name}] Question {i + 1}/{total}")
        answer = query_model(prompt)
        answers.append(answer)

        if (i + 1) % 50 == 0:
            partial = df.iloc[: i + 1].copy()
            partial["model_answer"] = answers
            partial.to_csv(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved at {i + 1}/{total}")

    result_df = df.copy()
    result_df["model_answer"] = answers
    result_df.to_csv(os.path.join(output_dir, f"answers_{prompt_name}.csv"), index=False)

    scored = evaluate_answers(result_df)

    if judge:
        logger.info(f"Running truthfulness judge for {prompt_name}...")
        scored = evaluate_truthfulness(scored)

    scored.to_csv(os.path.join(output_dir, f"scored_{prompt_name}.csv"), index=False)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    stats = compute_summary_stats(scored)
    logger.info(f"[{prompt_name}] Stats: {stats}")
    return scored


def main():
    root = get_project_root()
    output_dir = os.path.join(root, "outputs")
    ensure_dir(output_dir)

    df = load_truthfulqa(os.path.join(root, "data", "TruthfulQA.csv"))
    logger.info(f"Loaded {len(df)} questions")

    all_results = []
    for name, rel_path in PROMPT_CONFIGS.items():
        prompt_path = os.path.join(root, rel_path)
        logger.info(f"Running experiment: {name}")
        scored = run_experiment(df, name, prompt_path, output_dir, judge=True)
        scored["prompt_type"] = name
        all_results.append(scored)

    comparison = pd.concat(all_results, ignore_index=True)
    comparison.to_csv(os.path.join(output_dir, "prompt_comparison.csv"), index=False)
    logger.info("All experiments complete. Results saved to outputs/")


if __name__ == "__main__":
    main()
