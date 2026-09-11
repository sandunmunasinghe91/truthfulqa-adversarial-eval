import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from load_dataset import load_truthfulqa
from prompts import load_prompt_template, format_prompt
from open_model import load_model, generate_answer

load_dotenv()

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'TruthfulQA.csv')
PROMPT_CONFIGS = {
    'baseline': os.path.join(PROJECT_ROOT, 'prompts', 'baseline.txt'),
    'adversarial_force_wrong': os.path.join(PROJECT_ROOT, 'prompts', 'adversarial_force_wrong.txt'),
    'adversarial_popular_myth': os.path.join(PROJECT_ROOT, 'prompts', 'adversarial_popular_myth.txt'),
    'adversarial_conspiracy': os.path.join(PROJECT_ROOT, 'prompts', 'adversarial_conspiracy.txt'),
}
N = 20


def main():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not found in .env")
        return

    df = load_truthfulqa(DATA_PATH)
    df = df.head(N)
    print(f"Loaded {len(df)} questions\n")

    print("Loading model...")
    model, tokenizer, device = load_model("google/gemma-3-4b-it", token=hf_token)
    print(f"Device: {device}\n")

    for prompt_name, prompt_path in PROMPT_CONFIGS.items():
        template = load_prompt_template(prompt_path)
        print(f"\n{'='*60}")
        print(f"  {prompt_name}")
        print(f"{'='*60}")

        for i in range(N):
            question = df.iloc[i]['Question']
            prompt = format_prompt(template, question)
            answer = generate_answer(model, tokenizer, prompt, device)
            status = "OK" if answer else "EMPTY"
            print(f"[{i+1:2d}] [{status}] Q: {question[:60]}")
            print(f"          A: {answer[:120] if answer else '(empty)'}")
            print()


if __name__ == "__main__":
    main()
