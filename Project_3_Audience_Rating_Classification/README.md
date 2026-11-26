# Project 3: Netflix Audience Rating Classification

## Overview
This project establishes a multi-class predictive classification framework to predict the audience rating category of Netflix content based on genres, runtime attributes, release era, origin country, and directional format.

---

## Audience Maturity Tier Mapping
Rather than treating sparse ratings independently (e.g. `NC-17` with 3 items), ratings are categorized into industry-standard maturity tiers:
- **Mature_Adults**: TV-MA, R, NC-17, UR
- **Teens_14_Plus**: TV-14, PG-13
- **Older_Kids_Family**: TV-PG, PG, TV-G, G
- **Young_Children**: TV-Y, TV-Y7, TV-Y7-FV

---

## Task Workflow (As per Specification)
1. **Step 1: Analyze rating categories**
   - Performed distribution analysis across raw and tiered content classes.
2. **Step 2: Prepare training datasets**
   - Engineered runtime metrics, director indicators, country flags, and multi-label genre tags.
   - Performed stratified split into training and test datasets.
3. **Step 3: Train classification models**
   - Evaluated **Decision Tree**, **Random Forest**, **Extra Trees**, and **Multinomial Logistic Regression**.
4. **Step 4: Optimize model performance**
   - Employed `GridSearchCV` 3-fold cross-validation on `RandomForestClassifier` targeting `weighted_f1`.
   - Tuned parameters: `n_estimators`, `max_depth`, `min_samples_split`, and `criterion`.
5. **Step 5: Evaluate prediction accuracy**
   - Evaluated Multi-Class Confusion Matrix, Class-by-Class Precision/Recall/F1-score, and Feature Importances.

---

## Project Structure
```
Project_3_Audience_Rating_Classification/
├── README.md               # Technical specification
├── run_project.py          # Standalone execution pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading
│   ├── feature_engineering.py # Categorical & numerical pipeline
│   ├── train.py            # Model training implementations
│   ├── tune.py             # Hyperparameter tuning (GridSearchCV)
│   └── evaluate.py         # Multi-class evaluation & confusion matrix
├── models/                 # Saved best rating model & label encoder
└── reports/                # Confusion matrix, tuning charts, report JSON
```

---

## How to Run
```bash
python run_project.py
```
