#!/usr/bin/env python3
"""
Reveal Executive Summary of the Descision Tree :)
"""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """
    Print a readable text representation
    of the decision tree structure
    """

    printable_executive = tree.export_text(
        decision_tree=clf,
        feature_names=feature_names,
        class_names=class_names
    )
    print(printable_executive)

    return None
