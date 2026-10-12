#!/usr/bin/env python3
"""
What is it in Blackbox?
Lloyd Shaply Game Theory approach
"""
import shap


def get_shap_explainer_and_values(
        model, X_train, X_test
):
    """
    Takes a blackbox model with its
    train and test datasets
    """

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return explainer, shap_values
