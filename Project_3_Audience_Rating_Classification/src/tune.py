from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_random_forest(X_train, y_train, random_state: int = 42):
    """
    Step 4: Optimize model performance through systematic hyperparameter tuning.
    Explores tree depth, split criteria, and ensemble estimators.
    """
    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    param_grid = {
        'n_estimators': [100, 180],
        'max_depth': [12, 18, 24],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    return best_model, best_params, best_cv_score, grid_search.cv_results_
