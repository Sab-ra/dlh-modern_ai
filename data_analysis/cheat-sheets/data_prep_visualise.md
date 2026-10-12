# ML Data Preparation Cheat Sheet (Pandas & Scikit-Learn)

## 1. Initial Inspection & Discovery
Basic commands to understand the dataset's structure and variable types.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, model_selection

# Load and Inspect
df = pd.read_csv('data.csv')
print(df.shape)              # (Rows, Columns)
print(df.dtypes)             # Check if numeric columns are 'object'
print(df.isnull().sum())     # Count missing values per column
print(df.duplicated().sum()) # Count total duplicate rows
print(df.describe())         # Summary stats for numeric columns
```

## 2. Data Cleaning & Missingness
Handling "hidden" nulls (like spaces `" "`) and imputing values.

```python
# Force numeric conversion & coerce errors (like " ") to NaN
df['col'] = pd.to_numeric(df['col'], errors='coerce') # [Conversation history]

# Handle Missing Values (Strategies)
df = df.dropna(subset=['Target'])              # Drop rows missing target
df['col'] = df['col'].fillna(df['col'].median()) # Impute with robust median
df['col'] = df['col'].fillna(0)                  # Replace with constant

# Data Deduplication
df = df.drop_duplicates() # Remove redundant records
```

## 3. Exploratory Data Analysis (EDA)
Analyzing distributions and relationships before modeling.

### Univariate Analysis (Single Variable)
- **Categorical:** `df['col'].value_counts().plot(kind='bar')`.
- **Numerical:** Histograms + KDE for shape, Boxplots for outliers.

### Bivariate/Multivariate Analysis (Relationships)
- **Correlation:** `df.corr(numeric_only=True)` + `sns.heatmap(annot=True, cmap='coolwarm')`.
- **Chi-Square Test:** Test independence between categorical features and target.
- **Welch's t-test:** Compare means of numeric features across target groups.

## 4. Feature Engineering
Creating new signals from existing data.

```python
# Binning Continuous Data
df['Group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf], 
                     labels=['0-12', '13-24', '25-48', '49-60', '60+']) #

# Aggregate features
df['NumServices'] = df[['Service1', 'Service2']].apply(lambda x: (x == 'Yes').sum(), axis=1) #
```

## 5. Preprocessing for Machine Learning
Transforming data into formats compatible with algorithms.

```python
# 1. Encoding Categorical Variables
le = preprocessing.LabelEncoder()
df['Target'] = le.fit_transform(df['Target']) # Binary (No/Yes -> 0/1)

oe = preprocessing.OrdinalEncoder()
df[binary_cols] = oe.fit_transform(df[binary_cols]) # Ordinal relationships

df = pd.get_dummies(df, columns=['Category'], drop_first=True) # One-hot encoding

# 2. Outlier Removal (IQR Method)
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[~((df['col'] < (Q1 - 1.5 * IQR)) | (df['col'] > (Q3 + 1.5 * IQR)))] #

# 3. Feature Scaling (Essential for distance-based models)
scaler = preprocessing.StandardScaler()
df[['col1', 'col2']] = scaler.fit_transform(df[['col1', 'col2']]) # Mean=0, Std=1
```

## 6. Data Splitting
Final step before training to ensure valid evaluation.

```python
X = df.drop('Target', axis=1)
y = df['Target']

# Stratified split to preserve class balance in imbalanced datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) #
```
