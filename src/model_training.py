import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X: pd.DataFrame, y: pd.Series, params: dict):
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, params, cv=5, scoring='accuracy')
    grid.fit(X, y)
    print("Best params:", grid.best_params_)
    return grid.best_estimator_