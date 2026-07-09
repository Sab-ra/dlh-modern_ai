#!/usr/bin/env python3
"""
Evaluate missing values
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Visible only what is missing
    """
    plt.figure(figsize=(12, 8))

    row_indices, col_indices = np.where(df.isnull())
    plt.scatter(row_indices, col_indices, marker='|')
    plt.yticks(range(len(df.columns)), df.columns)
    plt.xlabel('Row Index')
    plt.ylabel('Columns')
    plt.title('Missing Values Map')

    plt.tight_layout()
    plt.show()
