# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TruthfulQA adversarial evaluation pipeline. Benchmarks LLM truthfulness under four prompting conditions (baseline + 3 adversarial) using the TruthfulQA dataset (790 questions, 37 categories). Two model tracks: OpenAI API models and open-weight models (Gemma 3 4B IT via Hugging Face). Evaluation uses ROUGE scores and a GPT-based truthfulness judge. Behavioral flip analysis identifies questions where models switch from truthful to untruthful under adversarial prompting. Logit-lens interpretability analysis probes transformer hidden states to locate where adversarial effects take hold.

## Commands

```bash
# Run full OpenAI experiment pipeline (all 4 prompt conditions)
./venv/bin/python src/experiment.py

# Generate OpenAI visualizations from existing results
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

## Project Structure

```
TruthfulQA_Project/
├── src/                                  # Shared source modules
├── notebooks/
│   ├── openai/                           # OpenAI model notebooks
│   │   ├── truthfulqa_pipeline.ipynb
│   │   └── behavioral_flips.ipynb
│   └── open_model/                       # Open-weight model notebooks
│       ├── open_model_pipeline.ipynb
│       ├── test_open_model.ipynb
│       ├── behavioral_flip_analysis.ipynb
│       └── logit_lens_analysis.ipynb
├── outputs/
│   ├── openai/                           # OpenAI results + figures/
│   └── open_model/                       # Gemma results + figures/
│       ├── behavioral_analysis/          # Flip pattern CSVs + figures/
│       └── logit_lens/                   # Logit-lens analysis CSVs + figures/
├── report/                               # Evaluation report
├── prompts/                              # Prompt templates
├── data/                                 # TruthfulQA.csv
└── tests/                                # Unit tests
```

## Architecture

The pipeline runs in this order: **load data -> format prompts -> query model -> evaluate (ROUGE + judge) -> visualize**.

- **`src/model.py`** — OpenAI API client. `query_model()` handles all API calls with exponential backoff retries. `judge_truthfulness()` uses a second GPT call to classify answers. The `REASONING_MODELS` set controls which models skip the `temperature` parameter. The default model is configured here.

- **`src/open_model.py`** — Hugging Face model client. `load_model()` loads a model/tokenizer onto the best available device (CUDA > MPS > CPU). `generate_answer()` runs inference. Default model is `google/gemma-3-4b-it`.

- **`src/experiment.py`** — Orchestrates the full OpenAI pipeline. Uses `get_project_root()` for all paths (never relative). Writes to `outputs/openai/`. Checkpoints every 50 questions for crash recovery.

- **`src/evaluator.py`** — Two evaluation layers: `evaluate_answers()` adds ROUGE scores, `evaluate_truthfulness()` adds GPT judge verdicts with its own checkpointing. `compute_summary_stats()` and `compute_category_stats()` aggregate results.

- **`src/rouge_utils.py`** — Named `rouge_utils` (not `rouge_score`) to avoid shadowing the `rouge-score` pip package. All imports must use `from rouge_utils import ...`.

- **`src/visualize.py`** — Generates four chart types to `outputs/openai/figures/`. Reads from `prompt_comparison.csv`.

## Notebooks

- **`notebooks/openai/truthfulqa_pipeline.ipynb`** — Full OpenAI pipeline with inline plots. Sections 1-3 setup; Section 4 runs experiments (skip if `outputs/openai/answers_*.csv` exist); Sections 5-6 evaluate; Sections 7-9 analyze and visualize.

- **`notebooks/openai/behavioral_flips.ipynb`** — Identifies OpenAI model behavioral flips (baseline truthful -> adversarial untruthful).

- **`notebooks/open_model/open_model_pipeline.ipynb`** — Full Gemma pipeline: load model, generate answers, ROUGE evaluation, truthfulness judging, behavioral flips, and visualization.

- **`notebooks/open_model/behavioral_flip_analysis.ipynb`** — Stratifies Gemma baseline-truthful questions by how many adversarial conditions flipped them (all-three, exactly-two, exactly-one, resistant). Outputs to `outputs/open_model/behavioral_analysis/`.

- **`notebooks/open_model/logit_lens_analysis.ipynb`** — Logit-lens interpretability analysis of 10 selected behavioral-flip cases. Extracts layer-wise hidden states, computes teacher-forced answer evidence (truthful vs incorrect targets), JSD, cosine similarity, and entropy across all 35 representation layers (embedding + 34 blocks). Compares flipping vs non-flipping conditions and the three adversarial strategies. Outputs to `outputs/open_model/logit_lens/`.

## Key Constraints

- The `.env` file holds `OPENAI_API_KEY` (required) and `HF_TOKEN` (required for open-weight models) at project root.
- Output CSVs and figures are gitignored — only code and prompts are tracked.
- Notebooks use `sys.path.append('../../src')` to import source modules. Kernel must point to `./venv/bin/python`.
- Prompt templates in `prompts/` use `{question}` as the placeholder — `format_prompt()` does simple string replacement (not f-strings or `.format()`).
