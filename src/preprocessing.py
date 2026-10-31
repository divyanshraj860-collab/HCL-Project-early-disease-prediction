
"""Data validation and preprocessing for Early Disease Prediction."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"

EXPECTED_COLUMNS = [
    "patient_id",
    "age",
    "bmi",
    "blood_pressure",
    "glucose",
    "cholesterol",
    "smoking",
    "label",
]

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "blood_pressure",
    "glucose",
    "cholesterol",
    "smoking",
]


def load_dataset(path=DATA_PATH):
    """Load and validate the dataset's schema and basic integrity."""
    df = pd.read_csv(path)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns. Expected: {EXPECTED_COLUMNS}; "
            f"received: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs found.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("The label column must contain only 0 and 1.")

    if not set(df["smoking"].unique()).issubset({0, 1}):
        raise ValueError("The smoking column must contain only 0 and 1.")

    return df


def prepare_features_and_target(df):
    """Separate model features and binary target."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    return X, y


def split_dataset(X, y, test_size=0.20, random_state=42):
    """Create a reproducible stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
