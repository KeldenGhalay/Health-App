import argparse
import os
import pandas as pd
from src.data_loader import load_csv, validate_dataframe, save_processed
from src.preprocessing import basic_clean
from src.train_model import train

def run(raw_csv: str | None = None, target: str = "stress_level", task: str = "classification"):
    if raw_csv is None:
        return
    df = load_csv(raw_csv)
    df = validate_dataframe(df)
    df = basic_clean(df)
    out = save_processed(df, "cleaned.csv")
    if target not in df.columns:
        return
    train(str(out), target, task, "models/trained_model.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=False)
    parser.add_argument("--target", default="stress_level")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    args = parser.parse_args()
    run(args.raw, args.target, args.task)

if __name__ == "__main__":
    main()
