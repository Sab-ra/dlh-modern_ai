#!/usr/bin/env python3
"""
Module bar+KDE, and boxplot for numbers
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Visualise distribution of numerical fields
    """

    if columns_to_plot is None:
        num_df = df.select_dtypes(include='number')
        columns_to_plot = num_df.columns.tolist()
    elif isinstance(columns_to_plot, str):
        columns_to_plot = [columns_to_plot]

    grid_width = 2
    grid_height = len(columns_to_plot)

    fig, axes = plt.subplots(grid_height,
                             grid_width,
                             figsize=(15, 3 * grid_height)
                             )

    if grid_height == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(columns_to_plot):
        data = df[col].dropna()

        # Left Subplot
        left_sub = axes[i, 0]
        left_sub.hist(data,
                      bins=30,
                      density=True,
                      alpha=0.7,
                      edgecolor='black')
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        left_sub.plot(x_range,
                      kde(x_range), color='red', ls='--')
        left_sub.set_title(f'{col} Histogram + KDE')

        # Right Subplot
        right_sub = axes[i, 1]
        right_sub.boxplot(data, vert=False)
        right_sub.set_title(f'{col} Boxplot')

    plt.tight_layout()
    plt.show()
