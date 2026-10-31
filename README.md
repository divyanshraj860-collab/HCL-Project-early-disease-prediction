# Early Disease Prediction Using Machine Learning on Health Data

**HCLTech Career Shaper Programme | Project No. 29**

Domain: Digital-Data-AI / AIML
Project Type: Individual Industry Project
Duration: 4 Weeks

## 1. Project Overview

This project develops an experimental machine-learning application using structured health-related data. It demonstrates data preprocessing, exploratory data analysis, supervised learning, model evaluation and explainability through SHAP.

The application is built with Python and Streamlit and compares three classification algorithms:

- Logistic Regression
- Random Forest
- XGBoost

The current application produces a binary dataset-label prediction. It does not provide separate, clinically validated predictions for diabetes, heart disease and hypertension.

## 2. Project Objectives

- Explore and validate structured health-related data.
- Perform data cleaning and preprocessing.
- Train and compare multiple machine-learning models.
- Evaluate models using classification metrics.
- Explain model behaviour through SHAP.
- Develop an interactive Streamlit dashboard.
- Document the limitations and ethical considerations of experimental health-data modelling.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Frontend / User interface | Streamlit |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Gradient boosting | XGBoost |
| Explainable AI | SHAP |
| Visualisation | Matplotlib, Seaborn, Streamlit |
| Model storage | Joblib |
| Testing | Pytest |
| Dataset storage | CSV files |

## 4. Dataset Description

The project uses `data/dataset.csv`.

| Property | Details |
|---|---|
| Total records | 5,000 |
| Total columns | 8 |
| Input features | 6 |
| Target variable | label |
| Missing cells | 0 in the validated dataset |
| Duplicate patient IDs | 0 in the validated dataset |

### Dataset columns

| Column | Description |
|---|---|
| patient_id | Record identifier |
| age | Age |
| bmi | Body mass index |
| blood_pressure | Blood pressure measurement |
| glucose | Glucose measurement |
| cholesterol | Cholesterol measurement |
| smoking | Binary smoking-status encoding |
| label | Binary target label |

The current dataset contains 500 records from the supplied base CSV and 4,500 additional synthetic records.

**Data provenance limitation:** The synthetic records are experimental data, not verified clinical observations. The source and clinical meaning of the original records and target labels must be confirmed. The prepared dataset has not been established as a validated merge of the PIMA Indians Diabetes and UCI Heart Disease datasets.

The project results must therefore be interpreted as experimental binary-label classification results, not clinically validated disease-prediction performance.

Refer to `data/data_dictionary.csv` for the column descriptions.

## 5. Exploratory Data Analysis

The EDA notebook is available at:

`notebooks/eda.ipynb`

The analysis includes:

- Dataset structure and data-quality checks.
- Target-label distribution.
- Feature distributions and histograms.
- Numerical feature correlation heatmap.

The generated evaluation and explainability visualisations are stored in `reports/`.

## 6. Machine-Learning Models

Three classification models are trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

The dataset is divided into training and testing sets using an 80:20 stratified split with a fixed random seed.

### Latest model comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 66.10% | 65.96% | 63.39% | 64.65% | 0.7160 |
| XGBoost | 64.40% | 63.77% | 62.99% | 63.37% | 0.6934 |
| Random Forest | 64.20% | 63.85% | 61.76% | 62.79% | 0.6901 |

Logistic Regression currently achieves the highest accuracy and ROC-AUC among the three tested models on this dataset.

These scores measure agreement with the dataset's binary labels. They do not establish clinical diagnostic accuracy.

## 7. Model Explainability Using SHAP

SHAP is used to examine the contribution of input features to the Random Forest model's predictions.

Latest mean absolute SHAP values:

| Feature | Mean absolute SHAP |
|---|---:|
| glucose | 0.104662 |
| age | 0.057580 |
| blood_pressure | 0.053636 |
| bmi | 0.047371 |
| cholesterol | 0.036622 |
| smoking | 0.024034 |

The SHAP results describe model behaviour. They must not be interpreted as proof of medical causation or clinical importance.

## 8. Streamlit Application

The application includes four sections:

### Overview
Displays dataset information, class distribution and project workflow.

### Experimental Prediction
Accepts six input features and returns a binary dataset-label prediction using the saved Logistic Regression model.

### Model Analytics
Displays model comparison metrics, accuracy comparison and Random Forest SHAP feature importance.

### About the Project
Describes the project objectives, technology stack, data limitations and medical disclaimer.

## 9. Project Structure

```text
early-disease-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── dataset.csv
│   └── data_dictionary.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
│
├── notebooks/
│   └── eda.ipynb
│
├── reports/
│   ├── confusion_matrices.png
│   ├── evaluation_details.txt
│   ├── model_comparison.csv
│   ├── roc_curves.png
│   ├── shap_feature_importance.csv
│   ├── shap_feature_importance.png
│   └── shap_summary.png
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
│
└── tests/
    └── test_preprocessing.py
```

## 10. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/divyanshraj860-collab/HCL-Project-early-disease-prediction.git
cd HCL-Project-early-disease-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 11. Running the Project

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local application:

```text
http://localhost:8501
```

To retrain the models:

```bash
python -m src.train
```

To regenerate evaluation reports:

```bash
python -m src.evaluate
```

To regenerate SHAP explanations:

```bash
python -m src.explain
```

To run automated tests:

```bash
python -m pytest tests/ -v
```

## 12. Generated Reports

The `reports/` folder contains:

- Model comparison metrics.
- Classification evaluation details.
- Confusion matrices.
- ROC curves.
- SHAP feature importance values.
- SHAP feature importance chart.
- SHAP summary plot.

## 13. Ethical Considerations and Limitations

- Synthetic records must not be represented as real patient observations.
- The provenance and clinical meaning of the target label have not been independently verified.
- Current results do not establish generalisation to real clinical populations.
- The application is an educational and experimental prototype.
- No output should be used for medical diagnosis, prevention or treatment.
- Medical decisions must be made with qualified healthcare professionals.

This application is not a medical device and is not intended for clinical use.

## 14. References

1. HCLTech Career Shaper Programme, Project No. 29: Early Disease Prediction Using Machine Learning on Health Data.
2. UCI Machine Learning Repository — Pima Indians Diabetes Dataset: https://archive.ics.uci.edu/dataset/34/diabetes
3. Scikit-learn Documentation: https://scikit-learn.org/stable/
4. XGBoost Documentation: https://xgboost.readthedocs.io/en/stable/
5. SHAP Documentation: https://shap.readthedocs.io/en/latest/
6. Streamlit Documentation: https://docs.streamlit.io/

---

**Project:** Early Disease Prediction Using Machine Learning on Health Data
**Programme:** HCLTech Career Shaper Programme
**Project Number:** 29
**Academic Year:** 2025-26
