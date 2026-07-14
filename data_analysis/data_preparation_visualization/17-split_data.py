#!/usr/bin/env python3
"""
Split data into training and testing sets
"""
from sklearn import model_selection


def split_data(df,
               target='Churn',
               test_size=0.2,
               random_state=42):
    """
    Split dataframe into two parts:
    - training set
    - testing set
    Well, assumption that some combination of input_df
    (the entire initial df, but the tartet set)
    results in target_set (single column)
    """

    input_df = df.drop([target], axis=1)
    target_set = df[target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        input_df,
        target_set,
        test_size=test_size,
        stratify=target_set,
        random_state=random_state
    )

    return (X_train, X_test, y_train, y_test)
