"""Preprocessing utilities for Week 2.

Provides functions to clean, encode, scale and split the fraud dataset.
"""
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Drop obvious identifiers
    for c in ['nameOrig', 'nameDest']:
        if c in df.columns:
            df = df.drop(columns=[c])
    # Fill NA numeric
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(0)
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Create balance error features
    if {'oldbalanceOrig', 'newbalanceOrig', 'amount'}.issubset(df.columns):
        df['orig_balance_delta'] = df['oldbalanceOrig'] - df['newbalanceOrig']
    if {'oldbalanceDest', 'newbalanceDest', 'amount'}.issubset(df.columns):
        df['dest_balance_delta'] = df['oldbalanceDest'] - df['newbalanceDest']
    # One-hot encode `type`
    if 'type' in df.columns:
        dummies = pd.get_dummies(df['type'], prefix='type')
        df = pd.concat([df.drop(columns=['type']), dummies], axis=1)
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train_scaled, X_test_scaled, scaler


def prepare_train_test(df: pd.DataFrame, target: str = 'isFraud', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    if target not in df.columns:
        raise ValueError(f'Target column {target} not found')
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_clean(df)
    df = feature_engineer(df)
    return df
