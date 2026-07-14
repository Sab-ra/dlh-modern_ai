#!/usr/bin/env python3
"""
Generate new features for DF
"""
import pandas as pd


def create_features(df):
    """Two new features, refer README"""

    df = df.copy()

    service_yes_cols = [
        'MultipleLines',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    existing_service_cols = [c for c in service_yes_cols if c in df.columns]

    num_services = (df[existing_service_cols] == 'Yes').sum(axis=1)
    if 'InternetService' in df.columns:
        num_services += df['InternetService'
                           ].isin(['DSL', 'Fiber optic']
                                  ).astype(int)

    df['NumServices'] = num_services

    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, float('inf')],
        labels=['0-12', '13-24', '25-48', '49-60', '60+'],
        right=True,
        include_lowest=False
    )

    cols_to_drop = existing_service_cols + ['InternetService', 'tenure']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df
