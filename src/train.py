
"""Train and compare baseline classification models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.preprocessing import (
    load_dataset,
    prepare_features_and_target,
    split_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def build_models():
    """Return the required baseline models."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        ),
    }


def main():
    df = load_dataset()
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    results = []
    fitted_models = {}

    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        })

        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values(
        "roc_auc", ascending=False
    )

    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    # Save each fitted model for later evaluation and explainability.
    for name, model in fitted_models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)

    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))
    print(f"\nMetrics saved to: {results_path}")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
