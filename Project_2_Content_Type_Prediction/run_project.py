import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.feature_engineering import build_features
from src.train import train_models
from src.evaluate import evaluate_and_compare_models

def main():
    print("=================================================================")
    print("      TASK 2: CONTENT TYPE PREDICTION BENCHMARK PIPELINE         ")
    print("=================================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Step 1: Select relevant dataset features
    print("\n[Step 1/5] Selecting relevant dataset features and eliminating leakage...")
    df = load_data()
    print(f" Dataset loaded: {len(df)} titles (Movies: {(df['type']=='Movie').sum()}, TV Shows: {(df['type']=='TV Show').sum()})")
    
    # Step 2: Encode categorical variables & extract feature matrix
    print("\n[Step 2/5] Encoding categorical variables and multi-label genres...")
    X, y, preprocessor, feature_names = build_features(df)
    print(f" Feature matrix constructed: {X.shape[0]} samples x {X.shape[1]} features.")
    
    # Step 3: Train classification models
    print("\n[Step 3/5] Training 4 distinct classification models...")
    results, X_train, X_test, y_train, y_test = train_models(X, y)
    print(" Training completed for: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.")
    
    # Step 4 & 5: Evaluate prediction performance & compare model accuracy
    print("\n[Step 4 & 5/5] Evaluating performance metrics and generating benchmark comparison...")
    df_metrics, best_model_name = evaluate_and_compare_models(results, y_test, feature_names, reports_dir=reports_dir)
    
    print("\n" + "="*50)
    print("           MODEL BENCHMARK COMPARISON TABLE       ")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50)
    print(f"\n Champion Model: {best_model_name}")
    
    # Persist Best Model and Preprocessor
    best_clf = results[best_model_name]['model']
    joblib.dump(best_clf, os.path.join(models_dir, 'best_model.joblib'))
    joblib.dump(preprocessor, os.path.join(models_dir, 'feature_pipeline.joblib'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.joblib'))
    
    print(f" Best model ({best_model_name}) saved to '{models_dir}'.")
    print(f" Performance plots, ROC curves, and reports saved to '{reports_dir}'.")
    print("\n>>> Task 2 Pipeline Completed Successfully! <<<")

if __name__ == '__main__':
    main()
