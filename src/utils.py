from pathlib import Path

def root():
    return Path(__file__).resolve().parents[1]

def raw_dir():
    return root() / "data" / "raw"

def processed_dir():
    return root() / "data" / "processed"

def models_dir():
    return root() / "models"
