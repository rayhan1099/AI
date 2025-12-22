# NumPy and Pandas - Essential Data Manipulation

## 📖 Table of Contents
1. [NumPy Fundamentals](#numpy-fundamentals)
2. [NumPy Array Operations](#numpy-array-operations)
3. [Pandas DataFrames](#pandas-dataframes)
4. [Data Cleaning with Pandas](#data-cleaning-with-pandas)
5. [Data Analysis with Pandas](#data-analysis-with-pandas)
6. [Real-World ML Examples](#real-world-ml-examples)

---

## NumPy Fundamentals

### Why NumPy?
- **Fast**: Written in C, optimized for numerical operations
- **Memory efficient**: Better than Python lists for large datasets
- **Foundation**: All ML libraries (TensorFlow, PyTorch) use NumPy arrays

### Installation
```bash
pip install numpy
```

### Creating Arrays

```python
import numpy as np

# From Python list
arr = np.array([1, 2, 3, 4, 5])
print(arr)  # [1 2 3 4 5]
print(type(arr))  # <class 'numpy.ndarray'>

# Multi-dimensional arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix)
# [[1 2 3]
#  [4 5 6]]

# Special arrays
zeros = np.zeros((3, 4))        # 3x4 array of zeros
ones = np.ones((2, 3))          # 2x3 array of ones
identity = np.eye(3)            # 3x3 identity matrix
range_arr = np.arange(0, 10, 2)  # [0 2 4 6 8]
linspace = np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ]
random_arr = np.random.rand(3, 3)  # Random values 0-1

# Array properties
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)      # (2, 3) - dimensions
print(arr.size)       # 6 - total elements
print(arr.ndim)       # 2 - number of dimensions
print(arr.dtype)      # int64 - data type
```

### Array Indexing and Slicing

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Basic indexing
print(arr[0])        # 0
print(arr[-1])       # 9

# Slicing
print(arr[2:5])      # [2 3 4]
print(arr[:5])       # [0 1 2 3 4]
print(arr[5:])       # [5 6 7 8 9]
print(arr[::2])      # [0 2 4 6 8] - every 2nd element

# Multi-dimensional indexing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0, 1])      # 2 - row 0, column 1
print(matrix[1, :])      # [4 5 6] - entire row 1
print(matrix[:, 2])      # [3 6 9] - entire column 2
print(matrix[0:2, 1:3])  # [[2 3] [5 6]] - submatrix

# Boolean indexing (very useful in ML)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = arr > 5
print(mask)              # [False False False False False  True  True  True  True  True]
print(arr[mask])         # [ 6  7  8  9 10]
print(arr[arr > 5])      # Same as above
print(arr[(arr > 3) & (arr < 8)])  # [4 5 6 7]
```

---

## NumPy Array Operations

### Mathematical Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Element-wise operations
print(a + b)      # [ 6  8 10 12]
print(a - b)      # [-4 -4 -4 -4]
print(a * b)      # [ 5 12 21 32] - element-wise multiplication
print(a / b)      # [0.2 0.333... 0.428... 0.5]
print(a ** 2)     # [ 1  4  9 16] - square each element

# Scalar operations
print(a + 10)     # [11 12 13 14]
print(a * 2)      # [2 4 6 8]

# Matrix multiplication
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.dot(a, b))      # Matrix multiplication
print(a @ b)             # Same as above (Python 3.5+)
```

### Statistical Operations

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Basic statistics
print(np.mean(arr))      # 5.5 - mean
print(np.median(arr))    # 5.5 - median
print(np.std(arr))       # 2.872... - standard deviation
print(np.var(arr))       # 8.25 - variance
print(np.min(arr))       # 1
print(np.max(arr))       # 10
print(np.sum(arr))       # 55
print(np.prod(arr))      # 3628800 - product

# Percentiles
print(np.percentile(arr, 25))   # 3.25 - 25th percentile
print(np.percentile(arr, 75))   # 7.75 - 75th percentile

# For 2D arrays, specify axis
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.mean(matrix, axis=0))  # [4. 5. 6.] - mean of each column
print(np.mean(matrix, axis=1))  # [2. 5. 8.] - mean of each row
```

### Array Manipulation

```python
arr = np.array([1, 2, 3, 4, 5])

# Reshaping
matrix = arr.reshape(1, 5)      # [[1 2 3 4 5]]
matrix = arr.reshape(5, 1)      # Column vector
flat = matrix.flatten()          # Back to 1D

# Concatenation
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.concatenate([a, b]))   # [1 2 3 4 5 6]

# Stacking
print(np.vstack([a, b]))        # Vertical stack
print(np.hstack([a, b]))        # Horizontal stack

# Splitting
arr = np.array([1, 2, 3, 4, 5, 6])
print(np.split(arr, 3))         # [array([1, 2]), array([3, 4]), array([5, 6])]

# Transpose
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.T)                 # [[1 4] [2 5] [3 6]]
```

### Broadcasting - NumPy's Superpower

```python
# Broadcasting allows operations on arrays of different shapes
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([1, 2, 3])

# Add b to each row of a
print(a + b)
# [[ 2  4  6]
#  [ 5  7  9]
#  [ 8 10 12]]

# Multiply each column by a scalar
print(a * 2)
# [[ 2  4  6]
#  [ 8 10 12]
#  [14 16 18]]
```

---

## Pandas DataFrames

### Why Pandas?
- **Data manipulation**: Excel-like operations in Python
- **Data cleaning**: Handle missing values, duplicates
- **Data analysis**: Groupby, pivot tables, aggregations
- **ML preparation**: Feature engineering, data preprocessing

### Installation
```bash
pip install pandas
```

### Creating DataFrames

```python
import pandas as pd
import numpy as np

# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'city': ['NYC', 'LA', 'Chicago', 'NYC'],
    'salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)
print(df)

# From CSV file
df = pd.read_csv('data.csv')

# From Excel
df = pd.read_excel('data.xlsx')

# From NumPy array
arr = np.random.rand(5, 3)
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# Basic info
print(df.shape)        # (rows, columns)
print(df.info())       # Data types and memory usage
print(df.describe())    # Statistical summary
print(df.head())       # First 5 rows
print(df.tail())       # Last 5 rows
```

### Accessing Data

```python
# Select columns
print(df['name'])              # Single column (Series)
print(df[['name', 'age']])     # Multiple columns (DataFrame)

# Select rows
print(df.iloc[0])              # First row by index
print(df.iloc[0:3])            # First 3 rows
print(df.loc[0])               # Row by label
print(df.loc[0:2, 'name'])     # Specific rows and columns

# Boolean indexing
print(df[df['age'] > 30])      # Rows where age > 30
print(df[df['city'] == 'NYC']) # Rows where city is NYC
print(df[(df['age'] > 30) & (df['salary'] > 60000)])

# Query method (more readable)
print(df.query('age > 30 and salary > 60000'))
```

### Adding/Modifying Data

```python
# Add new column
df['bonus'] = df['salary'] * 0.1

# Modify existing column
df['age'] = df['age'] + 1

# Add new row
new_row = {'name': 'Eve', 'age': 28, 'city': 'Boston', 'salary': 55000}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Drop columns
df = df.drop('bonus', axis=1)  # axis=1 for columns
df = df.drop([0, 1], axis=0)   # axis=0 for rows

# Rename columns
df = df.rename(columns={'name': 'full_name', 'age': 'years'})
```

---

## Data Cleaning with Pandas

### Handling Missing Values

```python
# Check for missing values
print(df.isnull().sum())       # Count missing per column
print(df.isnull().any())       # True if any missing in column

# Drop missing values
df_clean = df.dropna()                    # Drop rows with any NaN
df_clean = df.dropna(subset=['age'])      # Drop rows where age is NaN
df_clean = df.dropna(axis=1)              # Drop columns with NaN

# Fill missing values
df['age'].fillna(df['age'].mean(), inplace=True)  # Fill with mean
df['city'].fillna('Unknown', inplace=True)        # Fill with value
df.fillna(method='ffill', inplace=True)           # Forward fill
df.fillna(method='bfill', inplace=True)           # Backward fill

# Interpolation
df['age'].interpolate(method='linear', inplace=True)
```

### Handling Duplicates

```python
# Find duplicates
print(df.duplicated())         # Boolean series
print(df.duplicated().sum())  # Count of duplicates

# Drop duplicates
df = df.drop_duplicates()                    # Drop all duplicates
df = df.drop_duplicates(subset=['name'])     # Drop based on column
df = df.drop_duplicates(keep='first')        # Keep first occurrence
```

### Data Type Conversion

```python
# Convert data types
df['age'] = df['age'].astype(int)
df['salary'] = df['salary'].astype(float)

# Convert to datetime
df['date'] = pd.to_datetime(df['date_column'])

# Convert to category (saves memory)
df['city'] = df['city'].astype('category')
```

### String Operations

```python
# String methods (on string columns)
df['name'] = df['name'].str.upper()         # Uppercase
df['name'] = df['name'].str.lower()         # Lowercase
df['name'] = df['name'].str.strip()         # Remove whitespace
df['email'] = df['email'].str.replace('@', '_at_')  # Replace
df['name'] = df['name'].str.split(' ').str[0]  # Split and get first

# Check if contains
df[df['name'].str.contains('John')]
```

---

## Data Analysis with Pandas

### GroupBy Operations

```python
# Group by column
grouped = df.groupby('city')
print(grouped.mean())          # Mean of numeric columns per group
print(grouped['salary'].mean()) # Mean salary per city
print(grouped.size())          # Count per group

# Multiple aggregations
agg_df = df.groupby('city').agg({
    'salary': ['mean', 'min', 'max', 'count'],
    'age': 'mean'
})

# Custom aggregation
def salary_range(x):
    return x.max() - x.min()

df.groupby('city')['salary'].apply(salary_range)
```

### Pivot Tables

```python
# Create pivot table
pivot = df.pivot_table(
    values='salary',
    index='city',
    columns='age_group',
    aggfunc='mean'
)

# Cross-tabulation
pd.crosstab(df['city'], df['age_group'])
```

### Merging and Joining

```python
# Merge DataFrames (like SQL JOIN)
df1 = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'id': [2, 3, 4], 'age': [25, 30, 35]})

# Inner join
merged = pd.merge(df1, df2, on='id', how='inner')

# Left join
merged = pd.merge(df1, df2, on='id', how='left')

# Outer join
merged = pd.merge(df1, df2, on='id', how='outer')

# Concatenate
df_concat = pd.concat([df1, df2], axis=0)  # Vertical
df_concat = pd.concat([df1, df2], axis=1)  # Horizontal
```

### Sorting and Ranking

```python
# Sort by column
df_sorted = df.sort_values('salary', ascending=False)
df_sorted = df.sort_values(['city', 'salary'], ascending=[True, False])

# Rank
df['salary_rank'] = df['salary'].rank(ascending=False)
```

---

## Real-World ML Examples

### Example 1: Feature Engineering

```python
# Create features from existing data
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 100], labels=['Young', 'Middle', 'Old'])
df['salary_per_age'] = df['salary'] / df['age']
df['is_high_earner'] = (df['salary'] > df['salary'].median()).astype(int)

# One-hot encoding (for ML models)
df_encoded = pd.get_dummies(df, columns=['city'], prefix='city')
```

### Example 2: Data Preprocessing Pipeline

```python
def preprocess_data(df):
    # Copy to avoid modifying original
    df = df.copy()
    
    # Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    df['salary'].fillna(df['salary'].mean(), inplace=True)
    
    # Remove outliers (using IQR method)
    Q1 = df['salary'].quantile(0.25)
    Q3 = df['salary'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['salary'] >= Q1 - 1.5*IQR) & (df['salary'] <= Q3 + 1.5*IQR)]
    
    # Feature engineering
    df['salary_log'] = np.log1p(df['salary'])
    
    # Encode categorical
    df = pd.get_dummies(df, columns=['city'], drop_first=True)
    
    return df

processed_df = preprocess_data(df)
```

### Example 3: Train-Test Split Preparation

```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('target_column', axis=1)
y = df['target_column']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### Example 4: Time Series Data

```python
# Create time series DataFrame
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts_df = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(100).cumsum()
})

# Set date as index
ts_df.set_index('date', inplace=True)

# Resample (daily to weekly)
weekly = ts_df.resample('W').mean()

# Rolling window
ts_df['rolling_mean'] = ts_df['value'].rolling(window=7).mean()
ts_df['rolling_std'] = ts_df['value'].rolling(window=7).std()
```

---

## Performance Tips

### 1. Use Vectorized Operations
```python
# Slow (don't do this)
result = []
for value in df['column']:
    result.append(value * 2)

# Fast (do this)
result = df['column'] * 2
```

### 2. Use .loc and .iloc Properly
```python
# Avoid chained indexing
df['column'][0] = 5  # May not work

# Use .loc instead
df.loc[0, 'column'] = 5  # Correct
```

### 3. Use Categorical for Repeated Strings
```python
# Saves memory
df['city'] = df['city'].astype('category')
```

---

## Practice Exercises

### Exercise 1: Data Cleaning
```python
# Create a DataFrame with missing values, duplicates, and outliers
# Clean it completely
```

### Exercise 2: Feature Engineering
```python
# Load a dataset and create 5 new features
```

### Exercise 3: GroupBy Analysis
```python
# Group data by category and calculate multiple statistics
```

---

## Key Takeaways

1. **NumPy**: Fast numerical operations, foundation for all ML
2. **Pandas**: Essential for data manipulation and cleaning
3. **Vectorization**: Always prefer vectorized operations over loops
4. **Data Cleaning**: 80% of ML work is data preparation
5. **Practice**: Work with real datasets to master these tools

---

## Next Steps

Master these before moving to:
- **[03_Scikit_Learn_Complete_Guide.md](03_Scikit_Learn_Complete_Guide.md)** - Machine Learning algorithms

---

**Practice with real datasets from Kaggle or UCI ML Repository!**

