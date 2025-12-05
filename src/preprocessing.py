import pandas as pd

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_clean(df)
    df = pd.get_dummies(df, drop_first=True)
    return df
