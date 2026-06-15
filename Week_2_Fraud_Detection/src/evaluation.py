"""Evaluation utilities for Week 2.

Produces metrics and plots for trained models.
"""
import os
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report


def save_class_distribution(y, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title('Class distribution')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix(y_true, y_pred, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_roc_curve(model, X_test, y_test, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:,1]
    else:
        y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_precision_recall(model, X_test, y_test, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:,1]
    else:
        y_score = model.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_feature_importance(model, feature_names, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        df = pd.DataFrame({'feature': feature_names, 'importance': imp})
        df = df.sort_values('importance', ascending=False).head(20)
        plt.figure(figsize=(8,6))
        sns.barplot(x='importance', y='feature', data=df)
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def evaluate_and_report(models: Dict[str, object], X_test, y_test, out_dir: str = 'charts') -> Dict[str, Dict]:
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        results[name] = report
        # Save plots per model
        save_confusion_matrix(y_test, preds, os.path.join(out_dir, f'confusion_matrix_{name}.png'))
        try:
            save_roc_curve(model, X_test, y_test, os.path.join(out_dir, f'roc_curve_{name}.png'))
        except Exception:
            pass
        try:
            save_precision_recall(model, X_test, y_test, os.path.join(out_dir, f'precision_recall_{name}.png'))
        except Exception:
            pass
        try:
            save_feature_importance(model, X_test.columns.tolist(), os.path.join(out_dir, f'feature_importance_{name}.png'))
        except Exception:
            pass
    return results

