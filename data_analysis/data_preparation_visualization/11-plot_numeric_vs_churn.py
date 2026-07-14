#!/usr/bin/env python3
"""
Continuos numeric vars VS churn
"""
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """
    Plots side-by-side histograms
    """

    plt.figure(figsize=(12, 8))

    sit_puts = df[df['Churn'] == 'No'][col].dropna()
    fuc_offs = df[df['Churn'] == 'Yes'][col].dropna()

    plt.hist([sit_puts, fuc_offs],
              bins=30,
              label=['No', 'Yes'])
    plt.hist([sit_puts, fuc_offs],
              bins=30,
              label=['No', 'Yes'])
    plt.title(f'{col} Distribution by Churn')
    plt.xlabel(col)
    plt.legend(title='Churn')

    plt.tight_layout()
    plt.show()
