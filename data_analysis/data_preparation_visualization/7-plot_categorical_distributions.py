#!/usr/bin/env python3
"""
Multiple graphs
"""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    Plot bar charts for OBJECT data columns
    """

    if columns_to_plot is None:
        obj_df = df.select_dtypes(include='object')
        columns_to_plot = obj_df.drop(['Churn'], axis=1)
        columns_to_plot = columns_to_plot.columns.tolist()

    grid_width = 3
    grid_height = (len(columns_to_plot) - 1 + grid_width) // grid_width

    fig, axes = plt.subplots(grid_height,
                             grid_width,
                             figsize=(15, 5 * grid_height))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        counts = df[col].value_counts()

        axes[i].bar(counts.index, counts.values)
        axes[i].set_title(col)

        axes[i].tick_params(axis='x', rotation=45)

    for j in range(len(columns_to_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
