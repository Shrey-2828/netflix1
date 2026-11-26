import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def evaluate_models_and_tuning(
    baseline_models, tuned_rf, best_params, X_test, y_test, label_encoder, 
    feature_names, reports_dir: str = 'reports'
):
    """
    Step 5: Evaluate prediction accuracy and performance gains.
    Produces multi-class confusion matrix, precision/recall per tier, and diagnostic plots.
    """
    class_names = list(label_encoder.classes_)
    
    # 1. Compare all models on test set
    comparison = []
    for name, clf in baseline_models.items():
        y_pred = clf.predict(X_test)
        comparison.append({
            'Model': name,
            'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'Weighted_F1': round(float(f1_score(y_test, y_pred, average='weighted')), 4),
            'Tuned': False
        })
        
    # Tuned model
    y_pred_tuned = tuned_rf.predict(X_test)
    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted')
    
    comparison.append({
        'Model': 'Tuned Random Forest',
        'Accuracy': round(float(tuned_acc), 4),
        'Weighted_F1': round(float(tuned_f1), 4),
        'Tuned': True
    })
    
    df_comp = pd.DataFrame(comparison).sort_values(by='Weighted_F1', ascending=False)
    
    # Classification Report for Tuned Model
    report_dict = classification_report(y_test, y_pred_tuned, target_names=class_names, output_dict=True)
    
    # Save Report & Summary
    summary = {
        'model_comparison': df_comp.to_dict(orient='records'),
        'best_hyperparameters': best_params,
        'tuned_model_classification_report': report_dict
    }
    with open(f"{reports_dir}/classification_report.json", 'w') as f:
        json.dump(summary, f, indent=4)
        
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred_tuned)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=[c.replace('_', ' ') for c in class_names],
                yticklabels=[c.replace('_', ' ') for c in class_names])
    plt.title('Tuned Random Forest - Multi-Class Confusion Matrix', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Audience Tier')
    plt.ylabel('True Audience Tier')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/multiclass_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 3. Model Tuning Performance Gain Chart
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_comp, x='Model', y='Weighted_F1', palette='crest')
    plt.ylim(0.4, 0.9)
    plt.title('Audience Rating Classification: Weighted F1-Score Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('Weighted F1-Score')
    for idx, row in df_comp.reset_index(drop=True).iterrows():
        plt.text(idx, row['Weighted_F1'] + 0.01, f"{row['Weighted_F1']:.3f}", ha='center', fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/hyperparameter_tuning_results.png", dpi=300)
    plt.close()
    
    # 4. Feature Importance for Tuned Random Forest
    if hasattr(tuned_rf, 'feature_importances_') and len(tuned_rf.feature_importances_) == len(feature_names):
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': tuned_rf.feature_importances_})
        top_fi = fi_df.sort_values(by='Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_fi, x='Importance', y='Feature', palette='flare')
        plt.title('Top 15 Predictors of Netflix Content Audience Rating', fontsize=12, fontweight='bold')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{reports_dir}/audience_rating_feature_importance.png", dpi=300)
        plt.close()
        
    return df_comp, summary
