# Linear Models

## 0-linear_regression.py

### Linear Regression

Write a function `Linear_Regression()` to create a linear regression model using Scikit-learn, which uses ordinary least squares to fit a linear model to the data.

Arguments:
- None

Returns:
- model: An untrained LinearRegression instance.

_Required import: `from sklearn import linear_model`._

### Test Example

```python
$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
Linear_Regression = __import__('0-linear_regression').Linear_Regression

np.random.seed(42)

X1 = np.random.rand(200) * 10
X2 = X1 + np.random.normal(0, 0.05, 200)
X3 = np.random.rand(200) * 5
X = np.column_stack([X1, X2, X3])
y = 4*X1 + 3*X3 + np.random.normal(0, 5, 200)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

lr = Linear_Regression()
lr.fit(X_train, y_train)

print(lr.get_params())

print("\nLinear Regression Coefficients (for [X1, X2, X3]):", lr.coef_)
print("Linear Regression Intercept (bias term):", lr.intercept_)

$ ./0-main.py
{'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}

Linear Regression Coefficients (for [X1, X2, X3]): [-5.07832453  9.10426101  3.0841171 ]
Linear Regression Intercept (bias term): -0.45574437100450993
```

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

## Custom Dataset for tasks 0-2, 4

### To Visualise the Data

```python
$ cat visualize_data.py
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X1 = np.random.rand(200) * 10
X2 = X1 + np.random.normal(0, 0.05, 200)
X3 = np.random.rand(200) * 5
X = np.column_stack([X1, X2, X3])
y = 4*X1 + 3*X3 + np.random.normal(0, 5, 200)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(X1, X2, X3, c='blue', alpha=0.7, edgecolor='k')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('3D Scatter Plot of Features X1, X2, and X3')

plt.show()

$ ./visualize_data.py
```

### Relationships between features and Y

```python
$ cat explore_features_target_correlation.py
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X1 = np.random.rand(200) * 10
X2 = X1 + np.random.normal(0, 0.05, 200)
X3 = np.random.rand(200) * 5
X = np.column_stack([X1, X2, X3])
y = 4*X1 + 3*X3 + np.random.normal(0, 5, 200)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X[:, 0], y, color='blue', alpha=0.6, edgecolors='k')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('y')
axes[0].set_title('X1 vs y')

axes[1].scatter(X[:, 1], y, color='green', alpha=0.6, edgecolors='k')
axes[1].set_xlabel('X2')
axes[1].set_ylabel('y')
axes[1].set_title('X2 vs y')

axes[2].scatter(X[:, 2], y, color='orange', alpha=0.6, edgecolors='k')
axes[2].set_xlabel('X3')
axes[2].set_ylabel('y')
axes[2].set_title('X3 vs y')

fig.suptitle('Exploring the Relationship Between Input Features (X1, X2, X3) and the Target Variable y', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

$ ./explore_features_target_correlation.py
```