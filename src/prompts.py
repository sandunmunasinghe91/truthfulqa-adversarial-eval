def load_prompt_template(filepath):
    with open(filepath, "r") as f:
        return f.read()


def format_prompt(template, question):
    return template.replace("{question}", question)
