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

- [ ] df: pandas DataFrame with a Churn column
- [ ] Generates a bar plot of Churn value counts
- [ ] Uses colors: skyblue for ('No'), salmon for ('Yes')
- [ ] Displays the plot
- [ ] Returns: None
- [ ] NOTE: Your plot must be identical to the reference plot provided in the task.
