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

- [ ] The x-axis represents row indices (DataFrame records)
- [ ] The y-axis represents column names.
- [ ] Y-tick labels are explicitly mapped to the DataFrame column names.
- [ ] Each missing value is displayed as a vertical bar (|), using the default plotting color.
- [ ] Displays the plot using Matplotlib
- [ ] Returns: None

NOTE: Your plot must be identical to the reference plot provided in the task.
