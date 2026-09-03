# TruthfulQA Evaluation Report

## Objective

Evaluate the truthfulness of a language model (GPT-3.5-Turbo) under baseline and adversarial prompting conditions using the TruthfulQA benchmark (Lin et al., 2022). The benchmark contains 817 questions across 38 categories designed to test whether models reproduce common human misconceptions.

## Methodology

### Dataset

- **TruthfulQA** (817 questions, 38 categories): Questions span topics including Health, Law, Finance, History, Science, and more. Each question includes reference correct and incorrect answers.

### Prompt Conditions

| Condition | Description |
|-----------|-------------|
| **Baseline** | Standard truthful answering prompt |
| **Adversarial (Force Wrong)** | Instructs the model to give incorrect answers |
| **Adversarial (Popular Myth)** | Steers toward common misconceptions |
| **Adversarial (Conspiracy)** | Frames conspiracy theories as fact |

### Evaluation Metrics

1. **ROUGE Scores** (ROUGE-1, ROUGE-2, ROUGE-L): Measures textual overlap between model answers and reference correct answers.
2. **GPT-based Truthfulness Judge**: A separate GPT call classifies each answer as "truthful" or "untruthful" using the question, model answer, and reference correct/incorrect answers.

### Experimental Setup

- Model: `gpt-3.5-turbo` (temperature=0.0 for deterministic outputs)
- Each question is formatted with the respective prompt template and sent to the API
- Answers are evaluated on both ROUGE overlap and truthfulness judgment
- Results are broken down by prompt condition and question category

## Results

> **Note:** Run the pipeline (`python src/experiment.py` or the Jupyter notebook) to populate results below.

### Overall Results

| Prompt Type | ROUGE-1 | ROUGE-2 | ROUGE-L | Truthful Rate |
|-------------|---------|---------|---------|---------------|
| baseline | — | — | — | — |
| adversarial_force_wrong | — | — | — | — |
| adversarial_popular_myth | — | — | — | — |
| adversarial_conspiracy | — | — | — | — |

### Figures

After running the pipeline, the following figures are generated in `outputs/figures/`:

- `rouge_comparison.png` — Bar chart of mean ROUGE scores by prompt type
- `truthfulness_comparison.png` — Bar chart of truthfulness rates by prompt type
- `category_breakdown.png` — ROUGE-L and truthfulness rates by category
- `score_distributions.png` — Box plots showing score distributions per prompt type

### Expected Findings

1. **Baseline** should achieve the highest truthfulness rate, as it directly instructs the model to answer truthfully.
2. **Adversarial (Force Wrong)** should produce the lowest ROUGE scores against correct reference answers and lowest truthfulness rate.
3. **Adversarial (Popular Myth)** should reveal cases where the model reproduces common misconceptions even without being explicitly told to be wrong.
4. **Adversarial (Conspiracy)** tests whether the model can be steered toward conspiracy-aligned responses.

## Discussion

Key questions to address after collecting results:

- How resilient is GPT-3.5-Turbo to adversarial prompts?
- Which question categories are most susceptible to adversarial manipulation?
- Does ROUGE score correlate with the truthfulness judgment?
- Are there questions where the model gives untruthful answers even under the baseline condition?

## Conclusion

_To be completed after running experiments and analyzing results._

## References

- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022.
