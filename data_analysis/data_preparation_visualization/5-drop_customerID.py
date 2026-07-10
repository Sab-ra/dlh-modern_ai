#!/usr/bin/env python3
"""
Drop useless columns
"""


def drop_cusomerID(df):
    """drop()"""

    df = df.copy()

    df = df.drop(['customerID'], axis=1)
    return df
