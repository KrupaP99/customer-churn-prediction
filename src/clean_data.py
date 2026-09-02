"""
clean_data.py
Cleans the raw Telco churn data: fixes types, handles the TotalCharges
blank-string issue, and encodes the target variable.
"""

import pandas as pd
from load_data import load_raw_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps and return a cleaned copy of the dataframe."""
    df = df.copy()

    # TotalCharges is read as text because ~11 rows contain a blank string
    # instead of a number (usually customers with tenure = 0, i.e. brand new).
    # errors="coerce" turns those blanks into real NaN so we can see them.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    n_missing = df["TotalCharges"].isnull().sum()
    print(f"TotalCharges: found {n_missing} rows that were blank strings, now proper NaN")

    # These are new customers (tenure = 0), so 0 total charges is the
    # logical, defensible fill — not a guess.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target as 0/1 for modeling later
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID — it's an identifier, not a predictive feature
    df = df.drop(columns=["customerID"])

    return df


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    print(f"\nCleaned shape: {clean_df.shape}")
    print(f"Churn distribution:\n{clean_df['Churn'].value_counts(normalize=True)}")