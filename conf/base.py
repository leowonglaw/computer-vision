import os
from pathlib import Path

# absolute directory where main.py is located
BASE_PATH = Path(__file__).resolve().parents[1]


def register_directory(*args):
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def register_file(*args):
    path = os.path.join(*args)
    directory = os.path.dirname(path)
    register_directory(directory)
    if not os.path.exists(path):
        file = open(path, "x")
        file.close()
    return path
