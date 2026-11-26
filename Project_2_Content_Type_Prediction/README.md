# Project 2: Netflix Content Type Prediction Model

## Overview
This project builds a predictive binary classification system to accurately predict whether a given Netflix title is a **Movie** or a **TV Show** based on its contextual metadata, including director presence, country of origin, release year, maturity ratings, and genre tags.

---

## Technical Highlights & Target Leakage Prevention
In the raw Netflix dataset, the `duration` column directly contains strings like `90 min` for Movies and `1 Season` for TV Shows. Including duration unit text creates artificial 100% data leakage without learning real content patterns. In this project:
- We intentionally isolate content-level attributes: **Director presence**, **Country distribution**, **Release chronology**, **Genre composition**, and **Title semantics**.
- This enforces true machine learning classification based on creative format patterns.

---

## Task Workflow (As per Specification)
1. **Step 1: Select relevant dataset features**
   - Extracted features: Director attribution flag, Release year, Year/Month added to catalog, Title length, Top countries, and Audience rating tier.
2. **Step 2: Encode categorical variables**
   - Multi-label genre representation via tokenized boolean vectorization.
   - One-Hot Encoding for categorical features (`OneHotEncoder(handle_unknown='ignore')`).
   - Feature scaling via `StandardScaler`.
3. **Step 3: Train classification models**
   - **Logistic Regression**: Linear baseline with L2 penalty.
   - **Decision Tree**: Non-linear hierarchical decision rules.
   - **Random Forest**: Ensemble bagging with bootstrap aggregation.
   - **Gradient Boosting**: Sequential residual boosting.
4. **Step 4: Evaluate prediction performance**
   - Computed Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Evaluated Confusion Matrix and false positive/negative trade-offs.
5. **Step 5: Compare model accuracy**
   - Comparative benchmark table and visual diagnostic charts (ROC curves, metric comparisons, and feature importances).

---

## Project Structure
```
Project_2_Content_Type_Prediction/
├── README.md               # Technical specification & analysis
├── run_project.py          # Standalone training & benchmarking runner
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data ingestion & cleaning
│   ├── feature_engineering.py # Leakage-free preprocessing pipeline
│   ├── train.py            # Multi-algorithm model training
│   └── evaluate.py         # Metrics calculation & visualization
├── models/                 # Serialized champion model & preprocessor
└── reports/                # Saved charts (ROC, Confusion Matrix, Feature Importance)
```

---

## How to Run
```bash
python run_project.py
```
Outputs and visual reports are saved automatically in `reports/`.
