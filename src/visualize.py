import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import ensure_dir, get_project_root


def set_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150})


def plot_rouge_comparison(comparison_df, output_dir):
    set_style()
    metrics = ["rouge1", "rouge2", "rougeL"]
    summary = comparison_df.groupby("prompt_type")[metrics].mean()

    ax = summary.plot(kind="bar", rot=15)
    ax.set_title("ROUGE Scores by Prompt Type")
    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Prompt Type")
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rouge_comparison.png"))
    plt.close()


def plot_truthfulness_comparison(comparison_df, output_dir):
    if "is_truthful" not in comparison_df.columns:
        return
    set_style()
    rates = comparison_df.groupby("prompt_type")["is_truthful"].mean() * 100

    ax = rates.plot(kind="bar", color=sns.color_palette("muted"), rot=15)
    ax.set_title("Truthfulness Rate by Prompt Type")
    ax.set_ylabel("Truthful (%)")
    ax.set_xlabel("Prompt Type")
    for i, v in enumerate(rates):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "truthfulness_comparison.png"))
    plt.close()


def plot_category_breakdown(comparison_df, output_dir):
    if "Category" not in comparison_df.columns:
        return
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    cat_rouge = comparison_df.groupby(["Category", "prompt_type"])["rougeL"].mean().unstack()
    cat_rouge.plot(kind="barh", ax=axes[0])
    axes[0].set_title("ROUGE-L by Category and Prompt Type")
    axes[0].set_xlabel("Mean ROUGE-L")

    if "is_truthful" in comparison_df.columns:
        cat_truth = comparison_df.groupby(["Category", "prompt_type"])["is_truthful"].mean().unstack() * 100
        cat_truth.plot(kind="barh", ax=axes[1])
        axes[1].set_title("Truthfulness Rate by Category and Prompt Type")
        axes[1].set_xlabel("Truthful (%)")
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_breakdown.png"))
    plt.close()


def plot_score_distributions(comparison_df, output_dir):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["rouge1", "rouge2", "rougeL"]):
        sns.boxplot(data=comparison_df, x="prompt_type", y=metric, ax=ax)
        ax.set_title(f"{metric} Distribution")
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"))
    plt.close()


def generate_all_plots(comparison_csv=None):
    root = get_project_root()
    output_dir = os.path.join(root, "outputs", "openai")
    figures_dir = os.path.join(root, "outputs", "openai", "figures")
    ensure_dir(figures_dir)

    if comparison_csv is None:
        comparison_csv = os.path.join(output_dir, "prompt_comparison.csv")
    df = pd.read_csv(comparison_csv)

    plot_rouge_comparison(df, figures_dir)
    plot_truthfulness_comparison(df, figures_dir)
    plot_category_breakdown(df, figures_dir)
    plot_score_distributions(df, figures_dir)
    print(f"All plots saved to {figures_dir}")


if __name__ == "__main__":
    generate_all_plots()
