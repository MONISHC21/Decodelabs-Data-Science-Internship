"""Model training for Week 2.

Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers.
Saves models to `models/`.
"""
from typing import Tuple
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def train_models(X_train, y_train, models_dir: str = 'models') -> dict:
    os.makedirs(models_dir, exist_ok=True)
    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(models_dir, 'logistic_regression.pkl'))
    models['logistic_regression'] = lr

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(models_dir, 'random_forest.pkl'))
    models['random_forest'] = rf

    # Gradient Boosting (sklearn)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
    gb.fit(X_train, y_train)
    joblib.dump(gb, os.path.join(models_dir, 'gradient_boosting.pkl'))
    models['gradient_boosting'] = gb

    return models


def predict_models(models: dict, X_test) -> dict:
    preds = {}
    for name, m in models.items():
        preds[name] = m.predict(X_test)
    return preds

