# Data Preparation & Visualisation

## 0-describe_data.py

Complete the following source code (found below):

- [x] shape Shape of the DataFrame
- [x] data_types Data types of each column
- [x] head First 5 rows of the DataFrame
- [x] missing_count Count of missing values per column
- [x] duplicates Number of duplicate rows
- [x] Your program should be exactly 13 lines

## 1-plot_missingness.py

Write a function `def plot_missingness(df):` that visualizes missing values in a DataFrame:

df: pandas DataFrame to analyze

Generates a scatter plot where:

- [x] The x-axis represents row indices (DataFrame records)
- [x] The y-axis represents column names.
- [x] Y-tick labels are explicitly mapped to the DataFrame column names.
- [x] Each missing value is displayed as a vertical bar (|), using the default plotting color.
- [x] Displays the plot using Matplotlib
- [x] Returns: None

NOTE: Your plot must be identical to the reference plot provided in the task.

## 2-convert_columns.py

Write a function `def convert_columns(df):` that performs type conversion for specific columns:

- [x] df: pandas DataFrame containing the columns TotalCharges and SeniorCitizen
- [x] Converts the TotalCharges column to numeric.
- [x] Non-numeric entries should be converted to NaN
- [x] Maps the numeric values in the SeniorCitizen column (0 and 1) to categorical strings "No" and "Yes" respectively
- [x] Returns: The modified DataFrame
- [x] You are only allowed to use the following import: `import pandas as pd`.

## 3-clean_total_charges.py

Write a function `def clean_total_charges(df, method='drop'):` that handles missing values in TotalCharges:

- [x] df: pandas DataFrame with missing values in TotalCharges
- [x] method: Strategy to handle missing values:
- [x] 'drop': Remove rows with missing TotalCharges
- [x] 'median': Fill with column median
- [x] 'impute': Replace with MonthlyCharges * tenure
- [x] Returns the modified DataFrame

## 4-remove_duplicates.py

Write a function `def remove_duplicates(df):` that removes duplicate rows:

- [x] df: pandas DataFrame to process
- [x] Drops all duplicate rows
- [x] Returns the deduplicated DataFrame

## 5-drop_customerID.py

Write a function `def drop_customerID(df):` that removes the customerID column since unique identifiers lack predictive value:

- [x] df: pandas DataFrame containing a customerID column
- [x] Drops the customerID column
- [x] Returns the modified DataFrame

## 6-plot_churn_distribution.py

Write a function `def plot_churn_distribution(df):` that visualizes churn class distribution:

- [x] df: pandas DataFrame with a Churn column
- [x] Generates a bar plot of Churn value counts
- [x] Uses colors: skyblue for ('No'), salmon for ('Yes')
- [x] Displays the plot
- [x] Returns: None
- [x] NOTE: Your plot must be identical to the reference plot provided in the task.

## 7-plot_categorical_distributions.py

Write a function `def plot_categorical_distributions(df, columns_to_plot=None):` that visualizes categorical feature distributions:

- [x] df: pandas DataFrame
- [x] columns_to_plot: Optional list of categorical columns (Default: all columns with dtype object, excluding the target variable Churn.)
- [x] Generates bar plots for each categorical feature in a grid layout
- [x] Rotates x-axis labels by 45°
- [x] Displays the plot
- [x] Returns: None

_NOTE: Your plots must be identical to the reference plots provided in the task._

## 8-plot_continuous_distributions.py

Write a function `def plot_continuous_distributions(df, columns_to_plot=None):` that visualizes the distributions of continuous numerical features.

- [x] df: pandas DataFrame
- [x] columns_to_plot: Optional list of continuous numeric columns to plot. If None, it selects all numeric columns
- [x] For each selected column, generate:
    - [x] Left subplot: Histogram with KDE using the following settings:
        - [x] bins = 30
        - [x] density = True
        - [x] alpha = 0.7
        - [x] edgecolor = 'black'
        - [x] KDE line color should be red
        - [x] Title format: "<column_name> Histogram + KDE"
    - [x] Right subplot: Box Plot
        - [x] Title format: "<column_name> Boxplot"
- [x] Displays the plot
- [x] Returns: None

_NOTE: Your plots must be identical to the reference plots provided in the task._

## 9-plot_correlation_heatmap.py


Write a function `def plot_correlation_heatmap(df):` that visualizes correlations between continuous numeric features using seaborn:

- [x] df: pandas DataFrame
- [x] Computes pairwise correlations
- [x] Generates an annotated heatmap with coolwarm colormap
- [x] Set vmin = -1 and vmax = 1 so that the heatmap color mapping reflects the full correlation range
- [x] Displays the plot
- [x] Returns: None

_NOTE: Your plot must be identical to the reference plot provided in the task._

## 10-plot_categorical_vs_churn.py

Write a function `def plot_categorical_vs_churn(df, col):` that visualizes churn rates per category:

- [x] df: pandas DataFrame with Churn column
- [x] col: Categorical column name
- [x] Uses a figure size of (12, 8)
- [x] Adds a title to the plot in the format: "Churn Rate by <col>"
- [x] Plots churn rate (Yes proportion) per category as a bar plot
- [x] Sets y-axis label to "Churn Rate"
- [x] Rotates the x-axis labels by 45°
- [x] Displays the plot
- [x] Returns: None

_NOTE: Your plots must be identical to the reference plots provided in the task._

## 11-plot_numeric_vs_churn.py

Write a function `def plot_numeric_vs_churn(df, col):` that compares continuous numeric feature distributions by churn:

- [x] df: pandas DataFrame with Churn column
- [x] col: Numeric column name
- [x] Uses a figure size of (12, 8)
- [x] Adds a title to the plot in the format: "<col> Distribution by Churn"
- [x] Plots overlapping histograms for Churn=Yes and Churn=No
- [x] Sets the x-axis label to "<col>"
- [x] Uses 30 bins to group the data along the x-axis
- [x] Adds a legend with a title
- [x] Displays the plot
- [x] Returns: None

_NOTE: Your plot must be identical to the reference plot provided in the task._

## 12-chi_square_tests.py

Write a function `def chi_square_tests(df):` that performs chi-square tests for categorical features, using SciPy:

- [x] df: pandas DataFrame with Churn and categorical columns
- [x] Computes the Chi-square p-value to test the independence between each categorical feature and the target variable Churn, excluding Churn itself from the features tested.
- [x] Returns a dictionary: {feature_name: p_value}
- [x] You are only allowed to use the following imports: `import pandas as pd`, `from scipy import stats`

## 13-ttest_numeric.py

Write a function `def ttest_numeric(df):` that performs Welch's t-tests for continuous numeric features using scipy:

- [x] df: pandas DataFrame with Churn column
- [x] Computes t-test p-value comparing Churn=Yes vs Churn=No for each numeric feature

**The Hypothesis being tested is:**

- H₀ (null): The means of the variable are equal in Churn=Yes and Churn=No groups
- H₁ (alternative): The means differ significantly
Returns a dictionary: {feature_name: p_value}

_You are only allowed to use the following import: `from scipy import stats`_

## 14-create_features.py

Write a function `def create_features(df):` that engineers new features from the dataset:

**df: pandas DataFrame**

- [ ] Creates:
    - [ ] NumServices: Number of services the customer is subscribed to (counting only those with 'Yes' in selected service-related columns)
        - [ ] Do not include the PhoneService column, as it was dropped based on the decision made in Task 12 (nph_df)
        - [ ] For InternetService, count 'DSL' and 'Fiber optic' as 'Yes' (i.e., subscribed to the service), and 'No' as not subscribed
    - [ ] TenureGroup: A categorical column that bins the tenure into intervals: 0-12, 13-24, 25-48, 49-60, 60+ , where 0 is excluded and upper bounds are inclusive.

    - [ ] Drops the original columns that were used to create the new ones

- [ ] Returns the modified DataFrame

_You are only allowed to use the following import: `import pandas as pd`_

## 15-encode_features.py

Write a function `def encode_features(df):` that encodes features for modeling using Scikit-learn:

- [x] df: pandas DataFrame
- [x] lines of code in the function shall not be more than 79 characters
- [x] The function should encode:
    - Churn: LabelEncoder (No→0, Yes→1)
    - Partner, Dependents, PaperlessBilling, SeniorCitizen: OrdinalEncoder (No→0, Yes→1)
    - Contract, PaymentMethod: One-hot encoding with drop first set to True
    - TenureGroup: Alphabetical order OrdinalEncoder
- [x] Returns:
    - The encoded DataFrame
    - The Fitted LabelEncoder for Churn
    - The Fitted OrdinalEncoder for binary columns
    - The Fitted OrdinalEncoder for TenureGroup

_You are only allowed to use the following imports: `import pandas as pd`, `from sklearn import preprocessing`_

### How checker will test:

#### Main 1

```python
$ cat 15-main_0.py
#!/usr/bin/env python3

import pandas as pd
create_features = __import__('14-create_features').create_features
encode_features = __import__('15-encode_features').encode_features


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
df.drop(columns=['gender', 'PhoneService'], inplace=True)
df = create_features(df)
df_enc, churn_le, binary_oe, tenure_oe = encode_features(df)

pd.set_option('display.max_columns', None)
print(df_enc.head())

$ ./15-main_0.py
   SeniorCitizen  Partner  Dependents  PaperlessBilling  MonthlyCharges  \
0              0        1           0                 1           29.85
1              0        0           0                 0           56.95
2              0        0           0                 1           53.85
3              0        0           0                 0           42.30
4              0        0           0                 1           70.70

   TotalCharges  Churn  NumServices  TenureGroup  Contract_One year  \
0         29.85      0            2            0                  0
1       1889.50      0            3            2                  1
2        108.15      1            3            0                  0
3       1840.75      0            4            2                  1
4        151.65      1            1            0                  0

   Contract_Two year  PaymentMethod_Credit card (automatic)  \
0                  0                                      0
1                  0                                      0
2                  0                                      0
3                  0                                      0
4                  0                                      0

   PaymentMethod_Electronic check  PaymentMethod_Mailed check
0                               1                           0
1                               0                           1
2                               0                           1
3                               0                           0
4                               1                           0
4                               1                           0
```

#### Main 2

```python
$ cat 15-main_1.py
#!/usr/bin/env python3

import pandas as pd
create_features = __import__('14-create_features').create_features
encode_features = __import__('15-encode_features').encode_features


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
df.drop(columns=['gender', 'PhoneService'], inplace=True)
df = create_features(df)
df_enc, churn_le, binary_oe, tenure_oe = encode_features(df)

print(df_enc.dtypes)

$ ./15-main_1.py
SeniorCitizen                              int64
Partner                                    int64
Dependents                                 int64
PaperlessBilling                           int64
MonthlyCharges                           float64
TotalCharges                             float64
Churn                                      int64
NumServices                                int64
TenureGroup                                int64
Contract_One year                          int64
Contract_Two year                          int64
PaymentMethod_Credit card (automatic)      int64
PaymentMethod_Electronic check             int64
PaymentMethod_Mailed check                 int64
dtype: object
```

## 16-scale_numeric.py

Write a function `def scale_numeric(df):` that standardizes numeric columns:

- [x] df: pandas DataFrame
- [x] Scales MonthlyCharges and TotalCharges using StandardScaler (mean=0, std=1)
- [x] Returns the modified DataFrame

_You are only allowed to use the following import: `from sklearn import preprocessing`_

## 17-split_data.py

