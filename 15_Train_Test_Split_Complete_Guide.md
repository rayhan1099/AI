# Train/Test Split - Complete Guide

## 📖 Table of Contents
1. [Why Split Data?](#why-split-data)
2. [Basic Splitting Methods](#basic-splitting-methods)
3. [Advanced Splitting Strategies](#advanced-splitting-strategies)
4. [Time Series Splitting](#time-series-splitting)
5. [Cross-Validation](#cross-validation)
6. [Stratified Splitting](#stratified-splitting)
7. [Best Practices](#best-practices)
8. [Complete Examples](#complete-examples)

---

## Why Split Data?

### Purpose
- **Training Set**: Learn patterns and relationships
- **Validation Set**: Tune hyperparameters, select models
- **Test Set**: Final evaluation (unseen data)

### Common Split Ratios
- **70/30**: Simple, common for large datasets
- **80/20**: More training data, good default
- **60/20/20**: Includes validation set
- **90/10**: When data is limited

---

## Basic Splitting Methods

### 1. Simple Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # For reproducibility
    shuffle=True          # Shuffle before splitting
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

### 2. Train/Validation/Test Split

```python
# First split: Train + Validation vs Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Second split: Train vs Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,      # 25% of remaining = 20% of total
    random_state=42
)

print(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
```

### 3. Function for Three-Way Split

```python
def train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Labels
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1"
    
    # First split: Train vs (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_size + test_size),
        random_state=random_state
    )
    
    # Second split: Val vs Test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Usage
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, train_size=0.7, val_size=0.15, test_size=0.15
)
```

---

## Advanced Splitting Strategies

### 1. Stratified Split (For Imbalanced Data)

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Stratified split maintains class distribution
splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Or use train_test_split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,          # Maintain class distribution
    random_state=42
)

# Check distribution
print("Train distribution:", np.bincount(y_train))
print("Test distribution:", np.bincount(y_test))
```

### 2. Group Split (For Grouped Data)

```python
from sklearn.model_selection import GroupShuffleSplit

# Split by groups (e.g., same patient, same subject)
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Ensures no data leakage between groups
```

### 3. Shuffle Split (Multiple Random Splits)

```python
from sklearn.model_selection import ShuffleSplit

# Multiple random splits
splitter = ShuffleSplit(
    n_splits=5,           # 5 different splits
    test_size=0.2,
    random_state=42
)

scores = []
for train_idx, test_idx in splitter.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"Mean score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
```

---

## Time Series Splitting

### 1. Simple Time Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series split (no shuffling, maintains order)
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Score: {score:.4f}")
```

### 2. Custom Time Series Split

```python
def time_series_split(data, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Split time series data maintaining temporal order.
    """
    n = len(data)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

# Usage
train, val, test = time_series_split(ts_data, train_size=0.7, val_size=0.15, test_size=0.15)
```

### 3. Walk-Forward Validation

```python
def walk_forward_validation(data, n_train, n_test):
    """
    Walk-forward validation for time series.
    """
    for i in range(len(data) - n_train - n_test + 1):
        train = data[i:i+n_train]
        test = data[i+n_train:i+n_train+n_test]
        yield train, test

# Usage
for train, test in walk_forward_validation(ts_data, n_train=100, n_test=20):
    model.fit(train)
    predictions = model.predict(test)
    # Evaluate
```

---

## Cross-Validation

### 1. K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, cross_val_score

# K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model,
    X, y,
    cv=kfold,
    scoring='accuracy'
)

print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### 2. Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-fold (maintains class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model,
    X, y,
    cv=skf,
    scoring='accuracy'
)
```

### 3. Leave-One-Out Cross-Validation

```python
from sklearn.model_selection import LeaveOneOut

# Leave-one-out (n folds for n samples)
loo = LeaveOneOut()

scores = cross_val_score(
    model,
    X, y,
    cv=loo,
    scoring='accuracy'
)
```

### 4. Nested Cross-Validation

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

# Outer CV for model evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_scores = []
for train_idx, test_idx in outer_cv.split(X, y):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Inner CV for hyperparameter tuning
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=inner_cv,
        scoring='accuracy'
    )
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Evaluate best model on outer test set
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test_outer, y_test_outer)
    outer_scores.append(score)

print(f"Outer CV score: {np.mean(outer_scores):.4f}")
```

---

## Stratified Splitting

### Why Stratified?
- **Maintains class distribution** in train and test sets
- **Important for imbalanced datasets**
- **Prevents bias** in evaluation

### Implementation

```python
def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Stratified split maintaining class proportions.
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,          # Key parameter
        random_state=random_state
    )
    
    # Verify distribution
    train_dist = np.bincount(y_train) / len(y_train)
    test_dist = np.bincount(y_test) / len(y_test)
    
    print(f"Train distribution: {train_dist}")
    print(f"Test distribution: {test_dist}")
    
    return X_train, X_test, y_train, y_test

# For multi-class
X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
```

---

## Best Practices

### 1. Always Use Random State

```python
# For reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42  # Always set this!
)
```

### 2. Check Data Distribution

```python
def check_split_distribution(y_train, y_test):
    """Check if split maintains distribution."""
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    
    print("Train distribution:")
    print(train_dist)
    print("\nTest distribution:")
    print(test_dist)
    
    # Check if similar
    diff = abs(train_dist - test_dist)
    print(f"\nMax difference: {diff.max():.4f}")

check_split_distribution(y_train, y_test)
```

### 3. Handle Data Leakage

```python
# WRONG: Preprocessing before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses all data!
X_train, X_test = train_test_split(X_scaled, y)

# CORRECT: Preprocessing after split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on train
X_test_scaled = scaler.transform(X_test)       # Transform test
```

### 4. Save Split Indices

```python
# Save indices for reproducibility
train_idx, test_idx = train_test_split(
    np.arange(len(X)),
    test_size=0.2,
    random_state=42
)

# Save to file
np.save('train_indices.npy', train_idx)
np.save('test_indices.npy', test_idx)

# Load later
train_idx = np.load('train_indices.npy')
test_idx = np.load('test_indices.npy')
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

---

## Complete Examples

### Example 1: Complete ML Pipeline with Proper Splitting

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,          # For imbalanced data
    random_state=42
)

# 2. Preprocessing (ONLY on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Don't fit on test!

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

### Example 2: Time Series with Validation

```python
# Time series data
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(1000).cumsum()
})

# Split maintaining temporal order
train_size = int(len(ts_data) * 0.7)
val_size = int(len(ts_data) * 0.15)

train_data = ts_data[:train_size]
val_data = ts_data[train_size:train_size+val_size]
test_data = ts_data[train_size+val_size:]

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
```

### Example 3: Cross-Validation with Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search with CV
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

# Fit on training data
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Evaluate on test set
test_score = best_model.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

---

## Key Takeaways

1. **Always split before preprocessing** - Prevent data leakage
2. **Use stratified split** - For imbalanced data
3. **Maintain temporal order** - For time series
4. **Use cross-validation** - For better evaluation
5. **Save random state** - For reproducibility
6. **Check distributions** - Verify split quality

---

## Common Mistakes to Avoid

1. ❌ **Preprocessing before split** - Data leakage!
2. ❌ **Using test set for validation** - Overfitting!
3. ❌ **Not stratifying** - Biased evaluation!
4. ❌ **Shuffling time series** - Breaks temporal order!
5. ❌ **No random state** - Non-reproducible results!

---

## Next Steps

- **[14_Large_Dataset_Training_Guide.md](14_Large_Dataset_Training_Guide.md)** - Handle large datasets



---

**Master train/test splitting for reliable model evaluation!**

