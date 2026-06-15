"""Visualization helpers for Week 3.

This module provides functions to generate and save plots used in the
customer segmentation pipeline. All plotting functions accept a target
output directory and write PNG files so the notebook and scripts can run
non-interactively.
"""
from typing import Sequence
import matplotlib.pyplot as plt
import pandas as pd


def plot_elbow(inertia: Sequence[float], ks: Sequence[int], out_path: str) -> None:
    """Save an elbow-plot (inertia vs K).

    Args:
        inertia: Sequence of inertia values corresponding to `ks`.
        ks: Sequence of K values.
        out_path: File path to save the PNG.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertia, '-o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_silhouette(scores: Sequence[float], ks: Sequence[int], out_path: str) -> None:
    """Save silhouette score vs K plot.

    Args:
        scores: Sequence of silhouette scores (float) for each k.
        ks: Corresponding K values.
        out_path: File path to save the PNG.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, '-o', color='tab:orange')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score by k')
    plt.grid(True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_pca_scatter(pca_df: pd.DataFrame, labels: Sequence[int], out_path: str) -> None:
    """Save a 2D PCA scatter plot colored by cluster labels.

    Args:
        pca_df: DataFrame with two PCA columns ['PC1','PC2'].
        labels: Cluster labels for each row.
        out_path: File path to save the PNG.
    """
    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        mask = [l == lab for l in labels]
        plt.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], label=f'Cluster {lab}', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA projection with cluster labels')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_cluster_distribution(labels: Sequence[int], out_path: str) -> None:
    """Save a bar chart of cluster counts.

    Args:
        labels: Iterable of cluster labels.
        out_path: File path to save the PNG.
    """
    ser = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    ser.plot(kind='bar', color='tab:green')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Cluster distribution')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def save_personas_table(personas: pd.DataFrame, out_path: str) -> None:
    """Save the customer personas table as an image (simple matplotlib table).

    Args:
        personas: DataFrame summarizing personas.
        out_path: File path to save the PNG.
    """
    plt.figure(figsize=(10, max(2, 0.5 * len(personas))))
    plt.axis('off')
    table = plt.table(cellText=personas.values, colLabels=personas.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Customer Personas')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

