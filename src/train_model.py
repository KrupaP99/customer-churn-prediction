"""
train_model.py
Trains Logistic Regression and Random Forest models to predict churn,
and reports ROC-AUC and recall for each.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler

from load_data import load_raw_data
from clean_data import clean_data
from features import build_features


def prepare_train_test(features_df: pd.DataFrame):
    """Split features/target and train/test sets."""
    X = features_df.drop(columns=["Churn"])
    y = features_df["Churn"]

    # stratify=y keeps the ~73/27 churn split consistent in both
    # train and test sets — important given the class imbalance we found earlier.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """Scale features and train a Logistic Regression model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_random_forest(X_train, y_train):
    """Train a Random Forest model (no scaling needed for tree-based models)."""
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model


def evaluate_model(name, model, X_test, y_test, scaler=None):
    """Print ROC-AUC, recall, and full classification report."""
    X_eval = scaler.transform(X_test) if scaler else X_test

    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Recall (catching actual churners): {recall:.3f}")
    print(f"\nFull classification report:\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    features_df = build_features(clean_df)

    X_train, X_test, y_train, y_test = prepare_train_test(features_df)

    log_model, scaler = train_logistic_regression(X_train, y_train)
    evaluate_model("Logistic Regression", log_model, X_test, y_test, scaler=scaler)

    rf_model = train_random_forest(X_train, y_train)
    evaluate_model("Random Forest", rf_model, X_test, y_test)