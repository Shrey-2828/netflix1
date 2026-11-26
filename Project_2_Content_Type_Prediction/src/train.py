import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def train_models(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Step 3: Train classification models.
    Trains Logistic Regression, Random Forest, Gradient Boosting, and Decision Tree.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=random_state, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=5, random_state=random_state)
    }
    
    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        results[name] = {
            'model': clf,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
    return results, X_train, X_test, y_train, y_test
