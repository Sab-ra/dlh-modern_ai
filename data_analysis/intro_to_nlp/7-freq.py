#!/usr/bin/env python3
"""
Word frequency distribution for
a preprocessed corpus.
"""
import nltk
import matplotlib.pyplot as plt


def plot_top_n_frequencies(corpus_tokens, n=20):
    """
    Bar-plot the n most frequent tokens across
    a token corpus.
    """
    flat = [token for doc in corpus_tokens for token in doc]
    freq_dist = nltk.FreqDist(flat)

    top = freq_dist.most_common(n)
    words = [w for w, _ in top]
    counts = [c for _, c in top]

    plt.figure(figsize=(12, 5))
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {n} Most Frequent Words')
    plt.xlabel('Word')
    plt.ylabel('Frequency')

    plt.tight_layout()

    return freq_dist
