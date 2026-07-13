#!/usr/bin/env python3
"""
Multiple graphs
"""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    Generates bar plots for each categorical feature in a grid layout.
    """

    # 1. Determine columns to plot
    if columns_to_plot is None:
        columns_to_plot = [col for col in df.select_dtypes(
            include=['object']).columns if col != 'Churn']

    num_cols = len(columns_to_plot)
    if num_cols == 0:
        return

    # 2. Setup dynamic grid size
    ncols = 3
    nrows = (num_cols + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))

    # Flatten axes array for straightforward iteration
    if num_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 3. Plot each categorical column in its respective subplot
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        counts = df[col].value_counts()

        # Use pandas `.plot()` on the axis or Matplotlib's `ax.bar` directly

        ax.bar(counts.index, counts.values)

        ax.set_title(f'{col}')
        # Rotate the bottom tick labels by 45 degrees
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel(None)

    # 4. Remove any unused subplot spots in the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 5. Prevent label/title overlap and display
    plt.tight_layout()
    plt.show()
