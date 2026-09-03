# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TruthfulQA adversarial evaluation pipeline. Benchmarks an LLM's truthfulness under four prompting conditions (baseline + 3 adversarial) using the TruthfulQA dataset (817 questions, 38 categories). Evaluation uses ROUGE scores and a GPT-based truthfulness judge.

## Commands

```bash
# Run full experiment pipeline (all 4 prompt conditions)
./venv/bin/python src/experiment.py

# Generate visualizations from existing results
./venv/bin/python src/visualize.py

# Run all tests
./venv/bin/python -m pytest tests/ -v

# Run a single test file
./venv/bin/python -m pytest tests/test_evaluator.py -v

# Run a single test
./venv/bin/python -m pytest tests/test_evaluator.py::test_evaluate_answers -v

# Install dependencies
./venv/bin/pip install -r requirements.txt
```

Always use `./venv/bin/python` — the system Python does not have the project dependencies.

## Architecture

The pipeline runs in this order: **load data → format prompts → query model → evaluate (ROUGE + judge) → visualize**.

- **`src/model.py`** — OpenAI API client. `query_model()` handles all API calls with exponential backoff retries. `judge_truthfulness()` uses a second GPT call to classify answers. The `REASONING_MODELS` set controls which models skip the `temperature` parameter (reasoning models like `gpt-5.6-sol` don't support it). The default model is configured here.

- **`src/experiment.py`** — Orchestrates the full pipeline. Uses `get_project_root()` for all paths (never relative). Checkpoints every 50 questions to `outputs/checkpoint_{name}.csv` for crash recovery.

- **`src/evaluator.py`** — Two evaluation layers: `evaluate_answers()` adds ROUGE scores, `evaluate_truthfulness()` adds GPT judge verdicts with its own checkpointing. `compute_summary_stats()` and `compute_category_stats()` aggregate results.

- **`src/rouge_utils.py`** — Named `rouge_utils` (not `rouge_score`) to avoid shadowing the `rouge-score` pip package. All imports must use `from rouge_utils import ...`.

- **`src/visualize.py`** — Generates four chart types to `outputs/figures/`. Reads from `prompt_comparison.csv`.

- **`notebooks/truthfulqa_pipeline.ipynb`** — Interactive version of the same pipeline with inline plots. Sections 1-3 are setup; Section 4 runs experiments (skip if `outputs/answers_*.csv` exist); Sections 5-6 evaluate; Sections 7-9 analyze and visualize.

## Key Constraints

- The `.env` file holds `OPENAI_API_KEY` and must exist at project root.
- Output CSVs and figures are gitignored — only code and prompts are tracked.
- The notebook uses `sys.path.append('../src')` to import source modules. Kernel must point to `./venv/bin/python`.
- Prompt templates in `prompts/` use `{question}` as the placeholder — `format_prompt()` does simple string replacement (not f-strings or `.format()`).
