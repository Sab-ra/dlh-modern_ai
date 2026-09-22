#!/usr/bin/env python3
"""
Generate a word cloud from
a preprocessed corpus.
"""
import wordcloud
import matplotlib.pyplot as plt


def generate_wordcloud(
    corpus_tokens,
    max_words=200,
    label=None
):
    """
    Build and display a WordCloud object.
    """
    text = ''.join(tok for doc in corpus_tokens for tok in doc)
    wc = wordcloud.WordCloud(
        max_words=max_words,
        background_color='white',
        width=800,
        height=400,
        random_state=42
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if label:
        plt.title(f'WordCloud — {label}')
    else:
        plt.title('WordCloud')

    plt.tight_layout()

    return wc
