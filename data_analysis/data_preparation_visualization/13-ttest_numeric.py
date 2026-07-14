#!/usr/bin/env python3
"""
T-test
"""
from scipy import stats


def ttest_numeric(df):
    """
    Compute Welch's t-test p-values for numeric features by Churn group.
    Returns a dict: {feature_name: p_value}
    """

    p_values = {}
    numeric_cols = df.select_dtypes(include='number')
    numeric_cols = numeric_cols.columns.tolist()

    for col in numeric_cols:
        sit_puts = df.loc[df['Churn'] == 'No', col].dropna()
        fuc_offs = df.loc[df['Churn'] == 'Yes', col].dropna()

        _, p_value = stats.ttest_ind(fuc_offs, sit_puts, equal_var=False)
        p_values[col] = p_value

    return p_values
