#!/usr/bin/env python3
"""
Drop useless columns
"""


def drop_cusomerID(df):
    """drop()"""

    return df.drop(column=['customerID'])
