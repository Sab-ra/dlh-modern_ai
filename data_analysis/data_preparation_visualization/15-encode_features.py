#!/usr/bin/env python3
"""
Encode features for modelling
"""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    DF-specific encoding features function
    """
    df = df.copy()

    churn_le = preprocessing.LabelEncoder()
    df['Churn'] = churn_le.fit_transform(df['Churn']).astype(int)

    binary_cols = [
        'Partner',
        'Dependents',
        'PaperlessBilling',
        'SeniorCitizen'
    ]
    binary_oe = preprocessing.OrdinalEncoder(
        categories=[['No', 'Yes']] * len(binary_cols)
    )
    df[binary_cols] = binary_oe.fit_transform(df[binary_cols]).astype(int)

    tenure_oe = preprocessing.OrdinalEncoder(
        categories=[['0-12', '13-24', '25-48', '49-60', '60+']]
    )
    df[['TenureGroup']] = tenure_oe.fit_transform(df[['TenureGroup']])
    df['TenureGroup'] = df['TenureGroup'].astype(int)

    df = pd.get_dummies(
        df,
        columns=['Contract', 'PaymentMethod'],
        drop_first=True,
        dtype=int
    )

    return df, churn_le, binary_oe, tenure_oe
