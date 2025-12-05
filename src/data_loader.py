import pandas as pd
from pathlib import Path

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("empty dataframe")
    return df

def save_processed(df: pd.DataFrame, name: str = "cleaned.csv") -> Path:
    p = Path("data") / "processed"
    p.mkdir(parents=True, exist_ok=True)
    out = p / name
    df.to_csv(out, index=False)
    return out
