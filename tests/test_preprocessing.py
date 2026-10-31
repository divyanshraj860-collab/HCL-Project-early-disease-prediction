import pandas as pd

from src.preprocessing import (
    load_dataset,
    prepare_features_and_target,
    split_dataset,
)


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


def test_load_dataset():
    df = load_dataset()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5000, 8)
    assert df.columns.tolist() == EXPECTED_COLUMNS
    assert df["patient_id"].is_unique
    assert not df.isna().any().any()


def test_prepare_features_and_target():
    df = load_dataset()
    X, y = prepare_features_and_target(df)

    assert X.shape == (5000, 6)
    assert len(y) == 5000
    assert "patient_id" not in X.columns
    assert "label" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert not X.isna().any().any()


def test_split_dataset():
    df = load_dataset()
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert len(X_train) == 4000
    assert len(X_test) == 1000
    assert len(y_train) == 4000
    assert len(y_test) == 1000

    assert X_train.columns.tolist() == X_test.columns.tolist()
    assert X_train.index.intersection(X_test.index).empty
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})