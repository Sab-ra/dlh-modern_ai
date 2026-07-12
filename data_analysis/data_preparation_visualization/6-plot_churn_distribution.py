#!/usr/bin/env python3
"""Univaried Analysis Bar Chart"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Univaried Analysis"""

    plt.figure(figsize=(12, 8))

    colour_map = {'yes': 'salmon', 'no': 'skyblue'}

    counts = df['Churn'].value_counts()
    colours = counts.index.str.lower().map(colour_map)

    counts.plot(kind='bar', color=colours)
    plt.show()
