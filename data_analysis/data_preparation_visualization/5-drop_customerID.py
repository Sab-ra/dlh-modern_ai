#!/usr/bin/env python3
"""
Drop useless columns
"""


def drop_customerID(df):
    """drop()"""

    df = df.copy()

    df = df.drop(['customerID'], axis=1)
    return df
