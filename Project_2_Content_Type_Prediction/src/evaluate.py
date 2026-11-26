import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

def evaluate_and_compare_models(results, y_test, feature_names, reports_dir: str = 'reports'):
    """
    Step 4 & Step 5: Evaluate prediction performance and compare model accuracy.
    Generates comparison tables, ROC curves, confusion matrices, and feature importance.
    """
    metrics_list = []
    
    for name, res in results.items():
        y_pred = res['y_pred']
        y_prob = res['y_prob']
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        
        metrics_list.append({
            'Model': name,
            'Accuracy': round(float(acc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1_Score': round(float(f1), 4),
            'ROC_AUC': round(float(roc_auc), 4)
        })
        
    df_metrics = pd.DataFrame(metrics_list).sort_values(by='F1_Score', ascending=False)
    
    # Save metrics JSON
    with open(f"{reports_dir}/model_comparison_metrics.json", 'w') as f:
        json.dump(df_metrics.to_dict(orient='records'), f, indent=4)
        
    best_model_name = df_metrics.iloc[0]['Model']
    best_model_dict = results[best_model_name]
    
    # 1. Plot Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    metrics_plot = df_metrics.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1_Score'], 
                                   var_name='Metric', value_name='Score')
    sns.barplot(data=metrics_plot, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.ylim(0.5, 1.05)
    plt.title('Task 2: Classification Model Performance Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/model_comparison_metrics.png", dpi=300)
    plt.close()
    
    # 2. Plot Confusion Matrix for Best Model
    cm = confusion_matrix(y_test, best_model_dict['y_pred'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['TV Show (0)', 'Movie (1)'],
                yticklabels=['TV Show (0)', 'Movie (1)'])
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    # 3. Plot ROC Curve for All Models
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        if res['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            roc_auc = roc_auc_score(y_test, res['y_prob'])
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
            
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/roc_curve.png", dpi=300)
    plt.close()
    
    # 4. Feature Importance for Best Model (if tree-based)
    model_obj = best_model_dict['model']
    if hasattr(model_obj, 'feature_importances_'):
        importances = model_obj.feature_importances_
        if len(importances) == len(feature_names):
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            top_fi = fi_df.sort_values(by='Importance', ascending=False).head(15)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=top_fi, x='Importance', y='Feature', palette='magma')
            plt.title(f'Top 15 Feature Importances ({best_model_name})', fontsize=12, fontweight='bold')
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(f"{reports_dir}/feature_importance.png", dpi=300)
            plt.close()

    return df_metrics, best_model_name
