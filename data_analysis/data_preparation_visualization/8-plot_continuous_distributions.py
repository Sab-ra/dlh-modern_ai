#!/usr/bin/env python3
"""
Module bar+KDE, and boxplot for numbers
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Visualise distribution of numerical fields
    """

    if columns_to_plot is None:
        num_df = df.select_dtypes(include='number')
        columns_to_plot = num_df.columns.tolist()

    grid_width = 2
    grid_height = len(columns_to_plot)

    fig, axes = plt.subplots(grid_height,
                             grid_width,
                             figsize=(10, 3 * grid_height))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        data = df[col].dropna()

        ax_hist = axes[i * 2]
        ax_hist.hist(data,
                     bins=30,
                     density=True,
                     alpha=0.7,
                     edgecolor='black',
                     )
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(),
                             data.max(),
                             200
                             )
        ax_hist.plot(x_vals, kde(x_vals), color='red')
        ax_hist.set_title(f'{col} Histogram + KDE')

        ax_box = axes[2 * i + 1]
        ax_box.boxplot(data,)
        ax_box.set_title(f'{col} Boxplot')

    plt.tight_layout()
    plt.show()
