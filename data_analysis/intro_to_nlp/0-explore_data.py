#!/usr/bin/env python3
"""
Initial text-message dataset exploration.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """
    Creates a figure with two subplots
    side by side: bar- and hist-plot.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4)
    )

    counts = df['label'].value_counts()
    sns.barplot(
        x=counts.index,
        y=counts.values,
        ax=ax1
    )
    ax1.set_title('Ham vs Spam Counts')
    ax1.set_xlabel('label')
    ax1.set_ylabel('count')

    lengths = df['message'].str.len()
    sns.histplot(
        lengths,
        bins=50,
        ax=ax2
    )
    ax2.set_title('Histogram of Raw Message Lengths')
    ax2.set_xlabel('lenght')
    ax2.set_ylabel('count')

    plt.tight_layout()
