
"""Evaluate trained models and save diagnostic reports."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

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
    _, X_test, _, y_test = split_dataset(X, y)

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
    }

    reports = []
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(15, 4))
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for ax, (name, filename) in zip(axes_cm, model_files.items()):
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model = joblib.load(path)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)

        reports.append(
            f"\n{'=' * 50}\n{name}\n{'=' * 50}\n"
            + classification_report(
                y_test, predictions, zero_division=0
            )
            + f"\nROC-AUC: {auc:.4f}\n"
        )

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["Label 0", "Label 1"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax_roc.plot(
        [0, 1], [0, 1], linestyle="--", label="Random baseline"
    )
    ax_roc.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curves — Held-Out Test Split",
    )
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)

    fig_cm.tight_layout()
    fig_roc.tight_layout()

    fig_cm.savefig(REPORT_DIR / "confusion_matrices.png", dpi=160)
    fig_roc.savefig(REPORT_DIR / "roc_curves.png", dpi=160)
    plt.close(fig_cm)
    plt.close(fig_roc)

    report_path = REPORT_DIR / "evaluation_details.txt"
    report_path.write_text("\n".join(reports), encoding="utf-8")

    print("\n".join(reports))
    print(f"\nSaved report: {report_path}")
    print(f"Saved plots in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
