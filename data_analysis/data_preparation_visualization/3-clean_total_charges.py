#!/usr/bin/env python3
"""
Handle missing values:
- drop rows
- fill with columnd median
- impute with calculated values
"""


def clean_total_charges(df, method='drop'):
    """Handles missings in total_charges field"""

    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    if method == 'median':
        df['TotalCharges'] = df.to_numeric(df['TotalCharges'], errors='coerce')
        median_value = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
    if method == 'impute':
        impute_charge = df['MonthlyCharges'] * df['tenure']
        df['TotalCharges'] = df['TotalCharges'].fillna(impute_charge)

    return df
