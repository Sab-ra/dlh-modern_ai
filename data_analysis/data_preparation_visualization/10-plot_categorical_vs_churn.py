#!/usr/bin/env python3
"""
Plot Categorical bars VS Chorn
"""
import pandas as pt
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """
    Compare given column with Chorn
    (client morality rate)
    """

    # Churn rate (Yes proportion) per category
    churn_rate = (df['Churn'] == 'Yes'
                  ).astype(int
                           ).groupby(df[col]
                                     ).mean()

    plt.figure(figsize=(12, 8))

    plt.bar(churn_rate.index, churn_rate.values)

    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)

    plt.show()
