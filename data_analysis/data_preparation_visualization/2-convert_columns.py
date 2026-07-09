#!/usr/bin/env python3
"""
Learn how to change data type in cols
"""
import pandas as pd


def convert_columns(df):
    """
    TotalCharges: object -> numeric
    SeniorCitizen: 1/0 -> yes/no
    """

    ye_no_dic = {1: 'Yes', 0: 'No'}
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],
                                       errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map(ye_no_dic)

    return df
