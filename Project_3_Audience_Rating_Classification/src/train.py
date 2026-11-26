from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

def train_baseline_models(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Step 3: Train classification models.
    Trains Decision Tree, Random Forest, Extra Trees, and Logistic Regression on multi-class ratings.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1),
        'Multinomial Logistic': LogisticRegression(max_iter=1000, random_state=random_state)
    }
    
    fitted_models = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        fitted_models[name] = clf
        
    return fitted_models, X_train, X_test, y_train, y_test
