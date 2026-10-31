
"""Explain Random Forest predictions using SHAP."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.preprocessing import (
    load_dataset,
    prepare_features_and_target,
    split_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def main():
    df = load_dataset()
    X, y = prepare_features_and_target(df)
    _, X_test, _, _ = split_dataset(X, y)

    model_path = MODEL_DIR / "random_forest.joblib"
    model = joblib.load(model_path)

    # A subset keeps explanation computation manageable.
    X_explain = X_test.sample(
        n=min(200, len(X_test)),
        random_state=42,
    )

    # Explain class-1 probabilities from the fitted classifier.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # SHAP output shape differs across SHAP versions.
    if isinstance(shap_values, list):
        values_class_1 = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        values_class_1 = shap_values[:, :, 1]
    else:
        values_class_1 = shap_values

    # Global feature importance.
    plt.figure()
    shap.summary_plot(
        values_class_1,
        X_explain,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "shap_feature_importance.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    # Detailed feature contributions.
    plt.figure()
    shap.summary_plot(
        values_class_1,
        X_explain,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "shap_summary.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    importance = pd.DataFrame({
        "feature": X_explain.columns,
        "mean_absolute_shap": abs(values_class_1).mean(axis=0),
    }).sort_values("mean_absolute_shap", ascending=False)

    importance.to_csv(
        REPORT_DIR / "shap_feature_importance.csv",
        index=False,
    )

    print("SHAP analysis completed.")
    print(importance.to_string(index=False))
    print(f"\nResults saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
