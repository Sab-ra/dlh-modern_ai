#!/usr/bin/env python3
"""
Handle missing values:
- drop rows
- fill with columnd median
- impute with calculated values
"""


def clean_total_charges(df, method='drop'):
    """Handles missings in total_charges field"""

    # Error messages
    mver = 'Method shall be: "drop", "median", or "impute"'

    # Calculations
    median_value = df['TotalCharges'].median()
    impute_charge = df['MonthlyCharges'] * df['tenure']

    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    elif method == 'median':
        df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
    elif method == 'impute':
        df['TotalCharges'] = df['TotalCharges'].fillna(impute_charge)
    else:
        raise ValueError(mver)

    return df
