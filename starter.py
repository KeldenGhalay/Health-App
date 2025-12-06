import argparse
import glob
from src.data_loader import load_csv, validate_dataframe, save_processed
from src.preprocessing import basic_clean

def find_default_raw() -> str | None:
    candidates = glob.glob("data/raw/**/*1F96863*Dataset_*.csv", recursive=True)
    if candidates:
        return candidates[0]
    any_ds = glob.glob("data/raw/**/*Dataset_*.csv", recursive=True)
    return any_ds[0] if any_ds else None

def run(raw_csv: str | None = None):
    if raw_csv is None:
        raw_csv = find_default_raw()
        if raw_csv is None:
            print("No raw dataset found under data/raw")
            return
    df = load_csv(raw_csv)
    df = validate_dataframe(df)
    df = basic_clean(df)
    out = save_processed(df, "cleaned.csv")
    print(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=False)
    args = parser.parse_args()
    run(args.raw)

if __name__ == "__main__":
    main()
