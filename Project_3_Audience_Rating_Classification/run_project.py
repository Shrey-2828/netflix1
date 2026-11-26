import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.feature_engineering import analyze_and_map_ratings, prepare_features
from src.train import train_baseline_models
from src.tune import tune_random_forest
from src.evaluate import evaluate_models_and_tuning

def main():
    print("=================================================================")
    print("  TASK 3: AUDIENCE RATING MULTI-CLASS CLASSIFICATION PIPELINE    ")
    print("=================================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Step 1: Analyze rating categories
    print("\n[Step 1/5] Analyzing rating categories and structuring audience maturity tiers...")
    raw_df = load_data()
    clean_df, tier_dist, raw_dist = analyze_and_map_ratings(raw_df)
    print(f" Catalog sample count after mapping: {len(clean_df)}")
    print(" Audience Maturity Tier Distribution:")
    for tier, count in tier_dist.items():
        print(f"   - {tier}: {count} ({count/len(clean_df)*100:.1f}%)")
        
    # Step 2: Prepare training datasets
    print("\n[Step 2/5] Preparing feature matrix and label encoding target classes...")
    X, y, preprocessor, label_encoder, feature_names = prepare_features(clean_df)
    print(f" Engineered matrix: {X.shape[0]} samples x {X.shape[1]} features.")
    
    # Step 3: Train classification models
    print("\n[Step 3/5] Training baseline Decision Tree, Random Forest, Extra Trees & Logistic...")
    baseline_models, X_train, X_test, y_train, y_test = train_baseline_models(X, y)
    print(" Baseline training completed.")
    
    # Step 4: Optimize model performance
    print("\n[Step 4/5] Executing hyperparameter tuning (GridSearchCV) on Random Forest...")
    tuned_rf, best_params, best_cv_score, cv_results = tune_random_forest(X_train, y_train)
    print(f" Optimal Parameters identified: {best_params}")
    print(f" Best CV Weighted F1: {best_cv_score:.4f}")
    
    # Step 5: Evaluate prediction accuracy
    print("\n[Step 5/5] Evaluating multi-class prediction accuracy & confusion matrix...")
    df_comp, summary = evaluate_models_and_tuning(
        baseline_models, tuned_rf, best_params, X_test, y_test, label_encoder, feature_names, reports_dir=reports_dir
    )
    
    print("\n" + "="*55)
    print("      AUDIENCE RATING CLASSIFICATION BENCHMARK     ")
    print("="*55)
    print(df_comp.to_string(index=False))
    print("="*55)
    
    # Save Artifacts
    print("\n Persisting trained models and encoders...")
    joblib.dump(tuned_rf, os.path.join(models_dir, 'best_rating_model.joblib'))
    joblib.dump(preprocessor, os.path.join(models_dir, 'rating_preprocessor.joblib'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'rating_encoder.joblib'))
    joblib.dump(feature_names, os.path.join(models_dir, 'rating_feature_names.joblib'))
    
    print(f" Serialized artifacts saved to '{models_dir}'.")
    print(f" Classification reports & confusion matrices saved to '{reports_dir}'.")
    print("\n>>> Task 3 Pipeline Completed Successfully! <<<")

if __name__ == '__main__':
    main()
