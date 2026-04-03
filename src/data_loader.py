import pandas as pd
from config import DATA_PATH

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df["Streams"] = pd.to_numeric(df["Streams"], errors="coerce")
    df = df.dropna(subset=["Streams"])
    return df