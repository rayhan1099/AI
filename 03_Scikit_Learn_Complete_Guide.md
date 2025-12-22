# Scikit-learn Complete Guide - Machine Learning Mastery

## 📖 Table of Contents
1. [Introduction to Scikit-learn](#introduction-to-scikit-learn)
2. [Supervised Learning - Classification](#supervised-learning---classification)
3. [Supervised Learning - Regression](#supervised-learning---regression)
4. [Unsupervised Learning](#unsupervised-learning)
5. [Model Evaluation](#model-evaluation)
6. [Feature Engineering](#feature-engineering)
7. [Pipelines](#pipelines)
8. [Model Selection](#model-selection)

---

## Introduction to Scikit-learn

### What is Scikit-learn?
- **Most popular ML library** for Python
- **Simple and consistent API** - easy to learn
- **Production-ready** - used in industry
- **Comprehensive** - covers all ML algorithms

### Installation
```bash
pip install scikit-learn
```

### Basic Workflow
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load and prepare data
# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
```

---

## Supervised Learning - Classification

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create model
model = LogisticRegression(
    C=1.0,              # Regularization strength
    penalty='l2',       # L2 regularization
    solver='lbfgs',     # Algorithm
    max_iter=1000
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Probability estimates

# Evaluate
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

# Linear SVM
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train, y_train)

# RBF (Radial Basis Function) SVM
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train, y_train)

# Polynomial SVM
poly_svm = SVC(kernel='poly', degree=3, C=1.0)
poly_svm.fit(X_train, y_train)
```

### Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Create model
tree = DecisionTreeClassifier(
    max_depth=5,           # Limit tree depth
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=5,    # Minimum samples in leaf
    criterion='gini'      # or 'entropy'
)

# Train
tree.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=feature_names)
plt.show()

# Feature importance
importance = tree.feature_importances_
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Create model
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)

# Train
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

# Create model
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Train
gb.fit(X_train, y_train)

# Predict
y_pred = gb.predict(X_test)
```

### K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

# Create model
knn = KNeighborsClassifier(
    n_neighbors=5,        # Number of neighbors
    weights='uniform',    # or 'distance'
    algorithm='auto'
)

# Train
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
```

### Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Gaussian Naive Bayes (for continuous features)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Multinomial Naive Bayes (for count data)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Bernoulli Naive Bayes (for binary features)
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
```

---

## Supervised Learning - Regression

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create model
lr = LinearRegression()

# Train
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Coefficients
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_}")
```

### Ridge Regression (L2 Regularization)

```python
from sklearn.linear_model import Ridge

# Create model
ridge = Ridge(alpha=1.0)  # alpha = regularization strength

# Train
ridge.fit(X_train, y_train)

# Predict
y_pred = ridge.predict(X_test)
```

### Lasso Regression (L1 Regularization)

```python
from sklearn.linear_model import Lasso

# Create model
lasso = Lasso(alpha=1.0)

# Train
lasso.fit(X_train, y_train)

# Predict
y_pred = lasso.predict(X_test)

# Feature selection (Lasso sets some coefficients to 0)
selected_features = np.where(lasso.coef_ != 0)[0]
```

### Elastic Net (L1 + L2 Regularization)

```python
from sklearn.linear_model import ElasticNet

# Create model
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Train
elastic.fit(X_train, y_train)
```

### Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create pipeline
poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Train
poly_reg.fit(X_train, y_train)

# Predict
y_pred = poly_reg.predict(X_test)
```

### Support Vector Regression (SVR)

```python
from sklearn.svm import SVR

# Create model
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# Train
svr.fit(X_train, y_train)

# Predict
y_pred = svr.predict(X_test)
```

### Random Forest Regression

```python
from sklearn.ensemble import RandomForestRegressor

# Create model
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train
rf_reg.fit(X_train, y_train)

# Predict
y_pred = rf_reg.predict(X_test)
```

---

## Unsupervised Learning

### K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create model
kmeans = KMeans(
    n_clusters=3,        # Number of clusters
    init='k-means++',    # Smart initialization
    n_init=10,          # Number of initializations
    random_state=42
)

# Train
kmeans.fit(X)

# Predict clusters
labels = kmeans.predict(X)

# Evaluate
silhouette = silhouette_score(X, labels)

# Find optimal number of clusters (Elbow method)
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
```

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create model
hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'  # or 'complete', 'average'
)

# Train
labels = hierarchical.fit_predict(X)

# Visualize dendrogram
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.show()
```

### DBSCAN (Density-Based Clustering)

```python
from sklearn.cluster import DBSCAN

# Create model
dbscan = DBSCAN(
    eps=0.5,        # Maximum distance between samples
    min_samples=5   # Minimum samples in neighborhood
)

# Train
labels = dbscan.fit_predict(X)

# Number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
```

### Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA

# Create PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions

# Fit and transform
X_pca = pca.fit_transform(X)

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum()}")

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

# Find optimal number of components
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1  # 95% variance
```

### t-SNE (Dimensionality Reduction for Visualization)

```python
from sklearn.manifold import TSNE

# Create t-SNE
tsne = TSNE(n_components=2, random_state=42)

# Transform
X_tsne = tsne.fit_transform(X)

# Visualize
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()
```

---

## Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# ROC AUC (for binary classification)
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Precision-Recall Curve
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score
)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"Mean accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

# Stratified K-Fold (for imbalanced data)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

# Leave-One-Out Cross-Validation
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

---

## Feature Engineering

### Scaling and Normalization

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard Scaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Min-Max Scaler (0-1 range)
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X_train)

# Robust Scaler (uses median and IQR, robust to outliers)
robust = RobustScaler()
X_robust = robust.fit_transform(X_train)
```

### Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Label Encoding (for target variable)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse=False, drop='first')
X_encoded = onehot.fit_transform(X_categorical)

# Ordinal Encoding (for ordinal categories)
ordinal = OrdinalEncoder()
X_ordinal = ordinal.fit_transform(X_ordinal_features)
```

### Handling Missing Values

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple Imputer
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)

# KNN Imputer (more sophisticated)
knn_imputer = KNNImputer(n_neighbors=5)
X_knn_imputed = knn_imputer.fit_transform(X)
```

### Feature Selection

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)

# Select K Best features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
from sklearn.ensemble import RandomForestClassifier
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# Select from model (using feature importance)
sfm = SelectFromModel(RandomForestClassifier(), threshold='median')
X_sfm = sfm.fit_transform(X, y)
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

---

## Pipelines

### Creating Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Simple Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', RandomForestClassifier())
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Column Transformer (different preprocessing for different columns)
numeric_features = ['age', 'salary']
categorical_features = ['city', 'gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

---

## Model Selection

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Randomized Search (faster for large parameter spaces)
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, 'o-', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', label='Validation score')
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.legend()
plt.show()
```

---

## Complete Example: End-to-End ML Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define preprocessing
numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# 5. Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 6. Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)

# 7. Train
grid_search.fit(X_train, y_train)

# 8. Predict
y_pred = grid_search.predict(X_test)

# 9. Evaluate
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## Key Takeaways

1. **Consistent API**: All models use `.fit()`, `.predict()`, `.score()`
2. **Preprocessing is crucial**: Always scale/normalize features
3. **Use pipelines**: Makes code cleaner and prevents data leakage
4. **Cross-validation**: Always validate your models
5. **Hyperparameter tuning**: Can significantly improve performance

---

## Next Steps

Master these before moving to:
- **[04_Advanced_ML_Techniques.md](04_Advanced_ML_Techniques.md)** - Advanced ML methods
- **[05_Deep_Learning_Fundamentals.md](05_Deep_Learning_Fundamentals.md)** - Neural networks

---

**Practice with real datasets and build multiple projects!**

