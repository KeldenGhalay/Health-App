import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train(data_path: str = "data/processed/cleaned.csv", target: str = "RATE_PER_100_N", task: str = "regression", out_path: str = "models/trained_model.pkl") -> str:
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]
    if task == "classification":
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()
    model.fit(X, y)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/cleaned.csv")
    parser.add_argument("--target", default="RATE_PER_100_N")
    parser.add_argument("--task", choices=["classification", "regression"], default="regression")
    parser.add_argument("--out", default="models/trained_model.pkl")
    args = parser.parse_args()
    path = train(args.data, args.target, args.task, args.out)
    print(path)

if __name__ == "__main__":
    main()
