#!/usr/bin/env python3
"""
Find the most optimal parameters per
given decision tree
"""
from sklearn import model_selection


def prepruning(X, y, clf):
    """
    Perform grid search,
    return the best hype(r)parameters
    """

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(2, 5),
        'min_samples_leaf': range(2, 5),
        'min_samples_split': range(2, 5)
    }

    grid_search = model_selection.GridSearchCV(
        clf,
        param_grid
    )
    grid_search.fit(X, y)

    return grid_search.best_params_
