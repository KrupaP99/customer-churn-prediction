"""
load_data.py
Loads the raw Telco Customer Churn dataset and performs basic type fixes.
"""

import pandas as pd
from pathlib import Path

# Path to the raw CSV, relative to project root
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Telco-Customer-Churn.csv"


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw churn dataset from CSV."""
    df = pd.read_csv(path)
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Print a quick shape/dtype summary — useful sanity check after loading."""
    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())


if __name__ == "__main__":
    df = load_raw_data()
    basic_info(df)