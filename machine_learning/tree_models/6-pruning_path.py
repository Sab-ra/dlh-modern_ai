#!/usr/bin/env python3
"""
Retrieve the cost-complexity prunning path
of a decision tree
"""


def get_pruning_path(clf, X, y):
    """
    Return ccp_alphas and impurities
    from the prunning path
    """

    path = clf.cost_complexity_pruning_path(X, y)

    return path.ccp_alphas, path.impurities
