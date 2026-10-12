#!/usr/bin/env python3
"""
Chi-square test
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """
    Compute chi-square p-values between each
    categorical feature and Churn.
    Returns a dict: {feature_name: p_value}
    """
    p_values = {}
    cat_cols = [c for c in df.select_dtypes(include='object'
                                            ).columns if c != 'Churn']

    for col in cat_cols:
        contingency = pd.crosstab(df[col], df['Churn'])
        _, p_value, _, _ = stats.chi2_contingency(contingency)
        p_values[col] = p_value

    return p_values
