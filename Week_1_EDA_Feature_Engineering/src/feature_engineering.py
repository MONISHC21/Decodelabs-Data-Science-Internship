"""Feature engineering helpers for Week 1.

Adds features required by the Phase 1 spec:
- HasCoupon
- Month, Year, DayOfWeek, Quarter
- AverageOrderValue
- CustomerOrderFrequency
"""
from typing import Optional
import pandas as pd


def create_time_features(df: pd.DataFrame, date_col: str = 'OrderDate') -> pd.DataFrame:
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['DayOfWeek'] = df[date_col].dt.day_name()
        df['Quarter'] = df[date_col].dt.quarter
    else:
        df['Year'] = None
        df['Month'] = None
        df['DayOfWeek'] = None
        df['Quarter'] = None
    return df


def create_coupon_feature(df: pd.DataFrame, coupon_col: str = 'CouponCode') -> pd.DataFrame:
    df = df.copy()
    def _has_coupon_val(x):
        if pd.isna(x):
            return False
        s = str(x).strip()
        if s == '':
            return False
        if s.lower() in {'unknown', 'none', 'nan'}:
            return False
        return True

    if coupon_col in df.columns:
        df['HasCoupon'] = df[coupon_col].apply(_has_coupon_val)
    else:
        df['HasCoupon'] = False
    return df


def create_order_value_features(df: pd.DataFrame, total_col: str = 'TotalPrice') -> pd.DataFrame:
    df = df.copy()
    if total_col not in df.columns and {'Quantity', 'UnitPrice'}.issubset(df.columns):
        df[total_col] = df['Quantity'] * df['UnitPrice']

    # Average order value per OrderID
    if 'OrderID' in df.columns and total_col in df.columns:
        aov = df.groupby('OrderID')[total_col].sum().rename('OrderValue')
        df = df.merge(aov, on='OrderID', how='left')
        df['AverageOrderValue'] = df['OrderValue']
    elif total_col in df.columns:
        df['AverageOrderValue'] = df[total_col]
    else:
        df['AverageOrderValue'] = None
    return df


def create_customer_frequency(df: pd.DataFrame, customer_col: str = 'CustomerID') -> pd.DataFrame:
    df = df.copy()
    if customer_col in df.columns:
        freq = df.groupby(customer_col)['OrderID'].nunique().rename('CustomerOrderFrequency')
        df = df.merge(freq, on=customer_col, how='left')
    else:
        df['CustomerOrderFrequency'] = 0
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper to create all required features and return new DataFrame."""
    df = df.copy()
    df = create_time_features(df)
    df = create_coupon_feature(df)
    df = create_order_value_features(df)
    df = create_customer_frequency(df)
    return df
