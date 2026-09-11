import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_model(model_name="google/gemma-3-4b-it", device=None, token=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    if device in ("cpu", "mps"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=token, torch_dtype=dtype,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=token, torch_dtype=dtype, device_map=device,
        )

    logger.info(f"Model loaded on {device}")
    return model, tokenizer, device


def generate_answer(model, tokenizer, prompt, device, max_new_tokens=256):
    chat = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return answer.strip()
