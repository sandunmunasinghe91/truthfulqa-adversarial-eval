# TruthfulQA: Evaluating LLM Truthfulness Under Adversarial Prompting

This project benchmarks language model truthfulness using the [TruthfulQA](https://arxiv.org/abs/2109.07958) dataset (Lin et al., 2022). It evaluates how a model responds under a standard (baseline) prompt versus three adversarial prompting strategies designed to elicit false or misleading answers.

## Research Question

How resilient is a language model to adversarial prompts that attempt to steer it toward generating untruthful responses — and which question categories are most susceptible?

## Experimental Design

| Prompt Condition | Strategy |
|---|---|
| **Baseline** | Instructs the model to answer truthfully and concisely |
| **Force Wrong** | Explicitly tells the model to give an incorrect answer |
| **Popular Myth** | Steers toward common misconceptions and popular myths |
| **Conspiracy** | Frames conspiracy theories as accepted facts |

Each of the 817 TruthfulQA questions is run through all four conditions. Responses are evaluated using:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — textual overlap with reference correct answers
- **GPT-based Truthfulness Judge** — a separate model call classifies each answer as truthful or untruthful using reference correct and incorrect answers

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key with access to the model you want to evaluate

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd TruthfulQA_Project

# Create and activate the virtual environment
conda create -p ./venv python=3.11 -y
conda activate ./venv

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

To change the model being evaluated, update the default model in `src/model.py`.

## Usage

### Option 1: Jupyter Notebook (Recommended)

Step-by-step execution with inline results and visualizations:

```bash
jupyter notebook notebooks/truthfulqa_pipeline.ipynb
```

The notebook is organized into sections:
1. Setup and imports
2. Dataset exploration
3. Prompt template preview
4. Run experiments (with checkpointing)
5. ROUGE score evaluation
6. Truthfulness judgment evaluation
7. Summary statistics and category breakdown
8. Visualizations
9. Sample output inspection

### Option 2: Command Line

Run the full pipeline end-to-end:

```bash
python src/experiment.py
```

Generate plots from existing results:

```bash
python src/visualize.py
```

### Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
TruthfulQA_Project/
├── data/
│   └── TruthfulQA.csv              # 817 questions across 38 categories
├── prompts/
│   ├── baseline.txt                 # Truthful answering prompt
│   ├── adversarial_force_wrong.txt  # Incorrect answer prompt
│   ├── adversarial_popular_myth.txt # Common misconception prompt
│   └── adversarial_conspiracy.txt   # Conspiracy framing prompt
├── src/
│   ├── model.py                     # OpenAI API client with retry logic
│   ├── experiment.py                # Experiment runner with checkpointing
│   ├── evaluator.py                 # ROUGE scoring & truthfulness judge
│   ├── visualize.py                 # Chart generation (matplotlib/seaborn)
│   ├── rouge_utils.py               # ROUGE score computation
│   ├── prompts.py                   # Prompt template loading
│   ├── load_dataset.py              # Dataset loader
│   └── utils.py                     # Path and directory helpers
├── outputs/                         # Generated answers, scores, and figures
├── notebooks/
│   └── truthfulqa_pipeline.ipynb    # Interactive experiment notebook
├── report/
│   └── TruthfulQA_Report.md         # Evaluation report
├── tests/                           # Unit tests
├── requirements.txt
└── .env                             # API key (not tracked in git)
```

## Outputs

After running the pipeline, the `outputs/` directory contains:

- `answers_{prompt_type}.csv` — raw model responses per prompt condition
- `scored_{prompt_type}.csv` — responses with ROUGE scores and truthfulness judgments
- `prompt_comparison.csv` — combined results across all conditions
- `figures/` — generated visualizations (bar charts, box plots, category breakdowns)

## References

- Lin, S., Hilton, J., & Evans, O. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL 2022.
