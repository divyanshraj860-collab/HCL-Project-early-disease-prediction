from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
DATA_PATH = DATA_DIR / "dataset.csv"
METRICS_PATH = REPORT_DIR / "model_comparison.csv"
SHAP_PATH = REPORT_DIR / "shap_feature_importance.csv"

FEATURES = [
    "age",
    "bmi",
    "blood_pressure",
    "glucose",
    "cholesterol",
    "smoking",
]

st.set_page_config(
    page_title="Early Disease Prediction | AI Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1020;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stMetric"] {
        background-color: #151e32;
        border: 1px solid #293650;
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stForm"] {
        background-color: #111a2c;
        padding: 20px;
        border: 1px solid #293650;
        border-radius: 16px;
    }

    .hero {
        background: linear-gradient(120deg, #172554, #172033);
        border: 1px solid #334155;
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .hero p {
        color: #cbd5e1;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-top: 16px;
        margin-bottom: 12px;
    }

    .disclaimer {
        background: #2b2115;
        border: 1px solid #8b642d;
        border-radius: 12px;
        padding: 16px;
        color: #fde7bd;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load project artifacts
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if not set(FEATURES + ["label"]).issubset(df.columns):
        raise ValueError("Dataset schema is incomplete.")

    return df


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_shap_importance():
    if not SHAP_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(SHAP_PATH)


try:
    df = load_dataset()
    model = load_model()
    metrics = load_metrics()
    shap_importance = load_shap_importance()
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.info(
        "Check that dataset.csv and the trained model files "
        "exist in the project folders."
    )
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("🔬 AI Health Lab")
    st.caption("Early Disease Prediction Project")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Experimental Prediction",
            "Model Analytics",
            "About the Project",
        ],
    )

    st.divider()

    st.caption("Project status")
    st.write("Dataset records:", f"{len(df):,}")
    st.write("Input features:", len(FEATURES))
    st.write("Prediction model:", "Logistic Regression")

    st.warning(
        "Experimental research only. Not for clinical diagnosis."
    )

# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <h1>Early Disease Prediction</h1>
            <p>
                An experimental machine-learning dashboard for
                exploring health measurements, binary predictions,
                model evaluation, and feature importance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project Overview")

    positive_count = int((df["label"] == 1).sum())
    negative_count = int((df["label"] == 0).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", f"{len(df):,}")
    col2.metric("Input Features", len(FEATURES))
    col3.metric("Target Classes", "2")

    st.markdown(
        '<div class="section-title">Dataset Label Distribution</div>',
        unsafe_allow_html=True,
    )

    distribution = pd.DataFrame(
        {
            "Label": ["Label 0", "Label 1"],
            "Records": [negative_count, positive_count],
        }
    )

    st.bar_chart(
        distribution.set_index("Label"),
        y="Records",
    )

    st.caption(
        "The meanings of Label 0 and Label 1 have not been "
        "clinically verified."
    )

    st.markdown(
        '<div class="section-title">Project Workflow</div>',
        unsafe_allow_html=True,
    )

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 01")
        st.write("Health data exploration")
        st.caption("Quality checks and visualizations")

    with step2:
        st.markdown("### 02")
        st.write("Machine learning")
        st.caption("Training and model comparison")

    with step3:
        st.markdown("### 03")
        st.write("Explainability")
        st.caption("SHAP feature contribution analysis")

# --------------------------------------------------
# Experimental prediction
# --------------------------------------------------

elif page == "Experimental Prediction":
    st.title("Experimental Prediction")
    st.write(
        "Enter measurements to see how the trained model "
        "classifies the supplied feature values."
    )

    st.warning(
        "This tool predicts a dataset label, not a verified "
        "medical condition. Do not use its output to make "
        "healthcare decisions."
    )

    with st.form("prediction_form"):
        st.subheader("Patient Measurements")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=70.0,
                value=25.0,
                step=0.1,
            )

            blood_pressure = st.number_input(
                "Blood pressure (dataset units)",
                min_value=30,
                max_value=250,
                value=120,
                step=1,
            )

        with col2:
            glucose = st.number_input(
                "Glucose (dataset units)",
                min_value=20.0,
                max_value=500.0,
                value=100.0,
                step=0.5,
            )

            cholesterol = st.number_input(
                "Cholesterol (dataset units)",
                min_value=50,
                max_value=600,
                value=200,
                step=1,
            )

            smoking = st.selectbox(
                "Smoking status",
                options=[0, 1],
                format_func=lambda value: (
                    "0 — encoded as non-smoking"
                    if value == 0
                    else "1 — encoded as smoking"
                ),
            )

        submitted = st.form_submit_button(
            "Run Experimental Prediction",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame(
            [[
                age,
                bmi,
                blood_pressure,
                glucose,
                cholesterol,
                smoking,
            ]],
            columns=FEATURES,
        )

        try:
            prediction = int(model.predict(input_df)[0])

            st.divider()
            st.subheader("Model Output")

            if prediction == 1:
                st.warning("Predicted dataset label: 1")
            else:
                st.info("Predicted dataset label: 0")

            st.caption(
                "This is a binary model output only. The clinical "
                "meaning of the label is unverified. It is not "
                "a diagnosis or an estimate of disease risk."
            )

            st.write("Submitted measurements")
            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# Model analytics
# --------------------------------------------------

elif page == "Model Analytics":
    st.title("Model Analytics")
    st.write(
        "Review the evaluation metrics generated during "
        "the model-training stage."
    )

    if metrics.empty:
        st.info("Run the training script to generate model metrics.")
    else:
        st.subheader("Model Comparison")

        # Format a copy for display; keep the original numeric metrics
        # unchanged so the accuracy chart can use numeric values.
        display_metrics = metrics.copy()

        for column in ["accuracy", "precision", "recall", "f1_score"]:
            if column in display_metrics.columns:
                display_metrics[column] = display_metrics[column].map(
                    lambda value: f"{value:.2%}"
                )

        if "roc_auc" in display_metrics.columns:
            display_metrics["roc_auc"] = display_metrics["roc_auc"].map(
                lambda value: f"{value:.4f}"
            )

        st.dataframe(
            display_metrics,
            use_container_width=True,
            hide_index=True,
        )

        if "accuracy" in metrics.columns and "model" in metrics.columns:
            st.subheader("Accuracy Comparison")

            accuracy_chart = (
                metrics.set_index("model")[["accuracy"]].astype(float) * 100
            )

            st.bar_chart(accuracy_chart)
            st.caption("Accuracy is displayed as a percentage (%).")

    st.divider()
    st.subheader("SHAP Feature Importance — Random Forest")

    st.caption(
        "These SHAP values explain the Random Forest model's "
        "behaviour, not the Logistic Regression model used for "
        "the experimental prediction page."
    )

    if shap_importance.empty:
        st.info(
            "Run the SHAP explanation script to generate "
            "feature importance results."
        )
    else:
        st.bar_chart(
            shap_importance.set_index("feature")[
                ["mean_absolute_shap"]
            ]
        )

        st.dataframe(
            shap_importance,
            use_container_width=True,
            hide_index=True,
        )

# --------------------------------------------------
# About the project
# --------------------------------------------------

elif page == "About the Project":
    st.title("About This Project")

    st.markdown(
        """
        ### Objective

        Develop an experimental early disease prediction
        workflow using structured health-related data and
        supervised machine learning.

        ### Technologies

        - Python
        - Pandas and NumPy
        - Scikit-learn
        - XGBoost
        - SHAP
        - Streamlit

        ### Model development

        Logistic Regression, Random Forest, and XGBoost
        were trained and compared using accuracy, precision,
        recall, F1-score, and ROC-AUC.

        ### Data limitations

        The dataset contains synthetic records, and the
        target labels have not been clinically verified.
        Performance on these labels does not establish
        real-world disease-prediction performance.

        ### Intended use

        Educational demonstration and machine-learning
        experimentation only.
        """
    )

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Medical Disclaimer</strong><br><br>
            This application is not a medical device and must
            not be used to diagnose, prevent, or treat disease.
            Do not use its outputs to make clinical decisions.
            Consult a qualified healthcare professional for
            medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Early Disease Prediction | Experimental AI Research "
    "| Not for clinical use"
)
