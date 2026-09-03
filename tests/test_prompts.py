import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompts import load_prompt_template, format_prompt


def test_load_prompt_template():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Answer this: {question}")
        f.flush()
        template = load_prompt_template(f.name)
    os.unlink(f.name)
    assert template == "Answer this: {question}"


def test_format_prompt():
    template = "Answer this: {question}"
    result = format_prompt(template, "What is 2+2?")
    assert result == "Answer this: What is 2+2?"
    assert "{question}" not in result


def test_format_prompt_preserves_rest():
    template = "System: Be helpful.\nQ: {question}\nA:"
    result = format_prompt(template, "Why?")
    assert result == "System: Be helpful.\nQ: Why?\nA:"
