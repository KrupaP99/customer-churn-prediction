"""
features.py
Converts the cleaned churn dataframe into a fully numeric feature matrix
suitable for scikit-learn models.
"""

import pandas as pd
from clean_data import clean_data
from load_data import load_raw_data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns, leave numeric columns as-is."""
    df = df.copy()

    # Drop the tenure_group column if it exists (we added it just for EDA charts,
    # it's redundant with the raw 'tenure' column for modeling)
    if "tenure_group" in df.columns:
        df = df.drop(columns=["tenure_group"])

    # pandas automatically detects text/categorical columns and encodes them.
    # drop_first=True avoids redundant columns (e.g. gender_Male implies
    # gender_Female is the opposite), which helps avoid multicollinearity.
    df_encoded = pd.get_dummies(df, drop_first=True)

    return df_encoded


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    features_df = build_features(clean_df)
    print(f"Shape after encoding: {features_df.shape}")
    print(f"\nColumns:\n{features_df.columns.tolist()}")