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

    # LabelEncoder for Churn
    le_churn = preprocessing.LabelEncoder()
    df['Churn'] = le_churn.fit_transform(df['Churn'])

    # OrdinalEncoder for binary Yes/No columns
    binary_cols = ['Partner', 'Dependents', 'PaperlessBilling', 'SeniorCitizen']
    binary_cols = [c for c in binary_cols if c in df.columns]
    oe_binary = preprocessing.OrdinalEncoder(categories=[['No', 'Yes']] * len(binary_cols))
    df[binary_cols] = oe_binary.fit_transform(df[binary_cols])

    # OrdinalEncoder for TenureGroup
    tenure_cats = [['0-12', '13-24', '25-48', '49-60', '60+']]
    oe_tenure = preprocessing.OrdinalEncoder(categories=tenure_cats)
    df['TenureGroup'] = oe_tenure.fit_transform(df[['TenureGroup']])

    # One-hot encoding for Contract and PaymentMethod
    ohe_cols = [c for c in ['Contract', 'PaymentMethod'] if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    return df, le_churn, oe_binary, oe_tenure
