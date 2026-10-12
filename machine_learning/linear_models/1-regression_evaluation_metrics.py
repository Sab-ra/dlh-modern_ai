#!/usr/bin/env python3
"""
Reading Vitalities of Model
"""
import numpy as np
from sklearn import metrics


def evaluation_metrics_for_regression(
        y_true,
        y_pred
):
    """
    Compares: MSE, RMSE, MAE, R2
    """

    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return (mse, rmse, mae, r2)
