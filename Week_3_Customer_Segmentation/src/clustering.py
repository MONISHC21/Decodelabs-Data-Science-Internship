"""Clustering utilities for Week 3.

This module provides functions to compute the elbow method, silhouette
analysis and to run a final KMeans clustering and export results.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

from .visualization import plot_elbow, plot_silhouette, plot_cluster_distribution


def elbow_and_silhouette(X: pd.DataFrame, ks: List[int], out_dir: str) -> Tuple[List[float], List[float]]:
    """Compute inertia and silhouette scores for a list of k values.

    Args:
        X: Feature matrix (rows x features).
        ks: List of candidate k values.
        out_dir: Directory where plots will be saved.

    Returns:
        inertia_list, silhouette_list
    """
    inertia_list = []
    silhouette_list = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia_list.append(km.inertia_)
        if k > 1:
            try:
                score = silhouette_score(X, km.labels_)
            except Exception:
                score = float('nan')
        else:
            score = float('nan')
        silhouette_list.append(score)

    # Save plots
    os.makedirs(out_dir, exist_ok=True)
    plot_elbow(inertia_list, ks, os.path.join(out_dir, 'elbow_method.png'))
    plot_silhouette(silhouette_list, ks, os.path.join(out_dir, 'silhouette_score.png'))
    return inertia_list, silhouette_list


def run_kmeans(X: pd.DataFrame, n_clusters: int, out_dir: str, models_dir: str, original_df: pd.DataFrame = None) -> Tuple[KMeans, pd.Series]:
    """Run final KMeans clustering, save model and cluster assignments.

    Args:
        X: Feature matrix used to fit KMeans.
        n_clusters: Number of clusters to create.
        out_dir: Directory to save outputs (CSVs, charts).
        models_dir: Directory to save the trained KMeans model.
        original_df: Optional original DataFrame to attach cluster labels to.

    Returns:
        model: Trained KMeans model.
        labels: Pandas Series of cluster labels indexed like X.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(X)

    # Save model
    model_path = os.path.join(models_dir, 'kmeans_model.pkl')
    joblib.dump(model, model_path)

    # Attach to original dataframe if provided and save segmented csv
    labels_ser = pd.Series(labels, index=X.index, name='cluster')
    if original_df is not None:
        out_df = original_df.copy()
        out_df['cluster'] = labels_ser.values
        out_df.to_csv(os.path.join(out_dir, 'segmented_customers.csv'), index=False)
    else:
        pd.DataFrame({'cluster': labels_ser}).to_csv(os.path.join(out_dir, 'segmented_customers.csv'))

    # Save cluster distribution
    plot_cluster_distribution(labels, os.path.join(out_dir, 'cluster_distribution.png'))

    return model, labels_ser

