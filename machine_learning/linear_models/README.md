# Linear Models

## 0-linear_regression.py

### Linear Regression

Write a function `Linear_Regression()` to create a linear regression model using Scikit-learn, which uses ordinary least squares to fit a linear model to the data.

Arguments:
- None

Returns:
- model: An untrained LinearRegression instance.

_Required import: `from sklearn import linear_model`._

## 1-regression_evaluation_metrics.py

### Regression Evaluation Metrics

Write a function `evaluation_metrics_for_regression(y_true, y_pred)` to compute common evaluation metrics for regression tasks using Scikit-learn.

Arguments:
- y_true: A 1D NumPy array containing the true target values.
- y_pred: A 1D NumPy array containing the predicted target values.

Returns:
- A tuple (mse, rmse, mae, r2) where:
- mse: Mean Squared Error - average of the squared differences between actual and predicted values.
- rmse: Root Mean Squared Error - square root of the MSE, representing error in the original units.
- mae: Mean Absolute Error - average absolute difference between actual and predicted values.
- r2: R2 Score - indicates how well the model explains the variance in the target variable (1 means perfect prediction).

_Required imports: `from sklearn import metrics`, `import numpy as np`._

## 2-ridge_regression.py

### Ridge Regression

Write a function `ridge_regression(random_state)` that creates and returns a Ridge Regression model using Scikit-learn.

Ridge Regression extends ordinary linear regression by adding L2 regularization, which helps stabilize the model by shrinking large coefficients.

Arguments:
- random_state: An integer used to set the random seed for reproducibility.

Returns:
- model: An untrained Ridge regression model instance.

_Required import: `from sklearn import linear_model`._

## 3-Lasso_regression.py

### Lasso Regression

Write a function `lasso_regression(random_state)` that creates and returns a Lasso Regression model using Scikit-learn.

Lasso Regression extends ordinary linear regression by adding L1 regularization, which helps simplify the model by forcing some coefficients to zero, enabling automatic feature selection.

Arguments:
- random_state: An integer used to set the random seed for reproducibility.

Returns:
- model: An untrained Lasso regression model instance.

_Required import: `from sklearn import linear_model`._

## 4-shap.py

### SHAP: In-Depth Model Explainability and Feature Insights

Write a function `get_shap_explainer_and_values(model, X_train, X_test)` that helps generate model explanations using the SHAP library.

The function should:
- Create a SHAP explainer using X_train as the background dataset
- Compute SHAP values for X_test

Arguments:
- model: A trained regression model
- X_train: Input data used to initialize the explainer
- X_test: Input data to explain

Returns:
- explainer: SHAP explainer object
- shap_values: SHAP values for the predictions on X_test

_Required import: `import shap`._

## 5-logisitc_regression.py

### Build a Logistic Classifier

Write a function `Logistic_Regression_Model(random_state)` to create a logistic regression model using Scikit-learn, which performs binary classification by fitting a logistic function.

Arguments:
- random_state: An integer used to set the random seed for reproducibility.

Returns:
- model: An untrained LogisticRegression instance.

_Required import: `from sklearn import linear_model`._

## 6-svm.py

### Build SVM Classifier with Different Kernels

Write a function `get_SVM_model(name, random_state)` to create a Support Vector Machine (SVM) classifier using Scikit-learn with the specified kernel.

Arguments:
- name: A string indicating the type of model to return. Accepted values are:
	- 'linear': returns a SVM model with a linear kernel
	- 'poly': returns a SVM model with a polynomial kernel
	- 'rbf': returns a SVM model with a radial basis function (RBF) kernel
- random_state: The seed used by the random number generator for reproducibility.

Returns: An untrained instance of SVC

_Required import: `from sklearn import svm`._