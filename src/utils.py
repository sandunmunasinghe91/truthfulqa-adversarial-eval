import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
