#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """Encode dataset features using LabelEncoder for 'Churn',
    OrdinalEncoder for binary columns and 'TenureGroup',
    and One-Hot encoding for nominal features."""
    # Turn Churn (No/Yes) into 0/1, this is the target column
    le = preprocessing.LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])

    # Turn these Yes/No feature columns into 0/1 too
    ordinal_cols = ["Partner", "Dependents",
                    "PaperlessBilling", "SeniorCitizen"]
    oe = preprocessing.OrdinalEncoder(
        categories=[['No', 'Yes']])
    for col in ordinal_cols:
        df[[col]] = oe.fit_transform(df[[col]]).astype(int)

    # TenureGroup gets its own encoder, alphabetical order by default
    tenure_oe = preprocessing.OrdinalEncoder()
    df[['TenureGroup']] = tenure_oe.fit_transform(
        df[['TenureGroup']]).astype(int)

    # Contract and PaymentMethod have no real order, so one-hot
    # instead, drop first to avoid redundant columns
    ohe = preprocessing.OneHotEncoder(drop='first', sparse_output=False)
    ohe_cols = ["Contract", "PaymentMethod"]
    encoded = ohe.fit_transform(df[ohe_cols])

    # New columns come with auto-generated names, add them in
    df[ohe.get_feature_names_out(ohe_cols)] = encoded
    # Then remove the original text columns, already encoded now
    df = df.drop(columns=ohe_cols)

    return df, le, oe, tenure_oe
