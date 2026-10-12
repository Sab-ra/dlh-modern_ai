#!/usr/bin/env python3
"""
Prescale numerics for future ML
"""
from sklearn import preprocessing


def scale_numeric(df):
    """
    Acheave: mean - 0 and stdiv - 1
    """

    df = df.copy()

    # Initialise scaler tool
    standard_scaler = preprocessing.StandardScaler()

    cols_to_scale = ['MonthlyCharges', 'TotalCharges']

    df[cols_to_scale] = standard_scaler.fit_transform(df[cols_to_scale])

    return df
