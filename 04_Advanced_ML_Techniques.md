# Advanced Machine Learning Techniques

## 📖 Table of Contents
1. [Ensemble Methods](#ensemble-methods)
2. [XGBoost, LightGBM, CatBoost](#xgboost-lightgbm-catboost)
3. [Handling Imbalanced Data](#handling-imbalanced-data)
4. [Advanced Feature Engineering](#advanced-feature-engineering)
5. [Time Series Forecasting](#time-series-forecasting)
6. [AutoML](#automl)
7. [Model Interpretation](#model-interpretation)

---

## Ensemble Methods

### Why Ensemble Methods?
- **Better performance**: Combine multiple models for improved accuracy
- **Reduced overfitting**: More robust predictions
- **Industry standard**: Used in most competitions and production

### Voting Classifier

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Create individual models
lr = LogisticRegression()
svm = SVC(probability=True)
dt = DecisionTreeClassifier()

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('dt', dt)],
    voting='soft'  # or 'hard'
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
```

### Bagging (Bootstrap Aggregating)

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

# Bagging Classifier
bag_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,      # 80% of samples per tree
    max_features=0.8,     # 80% of features per tree
    bootstrap=True,       # Sampling with replacement
    n_jobs=-1,
    random_state=42
)

bag_clf.fit(X_train, y_train)
```

### Boosting

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# AdaBoost
ada_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.5
)

ada_clf.fit(X_train, y_train)

# Gradient Boosting
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_clf.fit(X_train, y_train)
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Base models
base_models = [
    ('rf', RandomForestClassifier()),
    ('svm', SVC(probability=True)),
    ('gb', GradientBoostingClassifier())
]

# Meta-learner
meta_learner = LogisticRegression()

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5  # Cross-validation for meta-features
)

stacking_clf.fit(X_train, y_train)
```

---

## XGBoost, LightGBM, CatBoost

### XGBoost (Extreme Gradient Boosting)

```python
import xgboost as xgb

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,        # Row sampling
    colsample_bytree=0.8, # Column sampling
    gamma=0,              # Minimum loss reduction
    reg_alpha=0,          # L1 regularization
    reg_lambda=1,         # L2 regularization
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(X_train, y_train)

# Feature importance
importance = xgb_clf.feature_importances_

# Early stopping
xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)
```

### LightGBM (Light Gradient Boosting Machine)

```python
import lightgbm as lgb

# LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,        # Maximum tree leaves
    feature_fraction=0.8,  # Feature sampling
    bagging_fraction=0.8,  # Data sampling
    bagging_freq=5,       # Bagging frequency
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_clf.fit(X_train, y_train)

# Categorical features (automatic handling)
lgb_clf.fit(
    X_train, y_train,
    categorical_feature=['city', 'gender']  # Specify categorical columns
)
```

### CatBoost (Categorical Boosting)

```python
import catboost as cb

# CatBoost Classifier
cat_clf = cb.CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=False
)

# Automatically handles categorical features
cat_clf.fit(
    X_train, y_train,
    cat_features=['city', 'gender'],  # Specify categorical columns
    eval_set=(X_test, y_test),
    early_stopping_rounds=10
)
```

### Comparison and When to Use

| Library | Speed | Accuracy | Categorical Handling | Best For |
|---------|-------|----------|---------------------|----------|
| XGBoost | Medium | High | Manual encoding | General purpose |
| LightGBM | Fast | High | Good | Large datasets |
| CatBoost | Medium | High | Excellent | Categorical features |

---

## Handling Imbalanced Data

### Problem with Imbalanced Data
- Models tend to predict majority class
- Accuracy can be misleading
- Need different evaluation metrics

### Techniques

#### 1. Resampling

```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# SMOTE (Synthetic Minority Oversampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ADASYN (Adaptive Synthetic)
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Random Over-sampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Random Under-sampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Combined (SMOTE + Tomek)
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
```

#### 2. Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Use in model
model = RandomForestClassifier(
    class_weight='balanced'  # or dict: {0: 1, 1: 5}
)

model.fit(X_train, y_train)
```

#### 3. Threshold Tuning

```python
from sklearn.metrics import precision_recall_curve

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# F1 score for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Use optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
```

#### 4. Evaluation Metrics for Imbalanced Data

```python
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    average_precision_score, f1_score
)

# Use these instead of accuracy
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Specificity and Sensitivity
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)  # Same as recall
```

---

## Advanced Feature Engineering

### Target Encoding

```python
def target_encode(train_df, test_df, col, target, alpha=100):
    """Target encoding with smoothing"""
    global_mean = train_df[target].mean()
    
    # Calculate mean target per category
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    
    # Smoothing
    smooth = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
    
    # Apply to train and test
    train_df[col + '_encoded'] = train_df[col].map(smooth).fillna(global_mean)
    test_df[col + '_encoded'] = test_df[col].map(smooth).fillna(global_mean)
    
    return train_df, test_df
```

### Feature Interactions

```python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Manual interaction features
df['feature1_x_feature2'] = df['feature1'] * df['feature2']
df['feature1_div_feature2'] = df['feature1'] / (df['feature2'] + 1e-6)
df['feature1_plus_feature2'] = df['feature1'] + df['feature2']
```

### Time-Based Features

```python
# Extract time features
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['hour'] = pd.to_datetime(df['datetime']).dt.hour

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

### Aggregation Features

```python
# Group aggregations
df['mean_by_category'] = df.groupby('category')['value'].transform('mean')
df['std_by_category'] = df.groupby('category')['value'].transform('std')
df['count_by_category'] = df.groupby('category')['value'].transform('count')

# Rolling features
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['rolling_std_7'] = df['value'].rolling(window=7).std()
df['rolling_max_7'] = df['value'].rolling(window=7).max()
```

### Feature Selection Techniques

```python
from sklearn.feature_selection import (
    mutual_info_classif, chi2, f_classif,
    SelectKBest, SelectPercentile
)

# Mutual Information
mi_scores = mutual_info_classif(X, y)
mi_features = SelectKBest(mutual_info_classif, k=10).fit(X, y)

# Chi-squared (for categorical features)
chi2_features = SelectKBest(chi2, k=10).fit(X_categorical, y)

# Recursive Feature Elimination with Cross-Validation
from sklearn.feature_selection import RFECV

rfecv = RFECV(
    estimator=RandomForestClassifier(),
    step=1,
    cv=5,
    scoring='accuracy'
)
rfecv.fit(X, y)
```

---

## Time Series Forecasting

### ARIMA Model

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(ts_data, order=(1, 1, 1))  # (p, d, q)
fitted_model = model.fit()

# Forecast
forecast = fitted_model.forecast(steps=10)

# Auto ARIMA (finds best parameters)
from pmdarima import auto_arima

model = auto_arima(
    ts_data,
    seasonal=True,
    m=12,  # Seasonal period
    stepwise=True
)
```

### Prophet (Facebook)

```python
from prophet import Prophet

# Prepare data (must have 'ds' and 'y' columns)
df_prophet = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=100),
    'y': ts_data
})

# Create and fit model
model = Prophet()
model.fit(df_prophet)

# Make future dataframe
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
model.plot(forecast)
```

### LSTM for Time Series

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(ts_data, seq_length)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=32)
```

---

## AutoML

### Auto-Sklearn

```python
import autosklearn.classification

# AutoML Classifier
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=3600,  # 1 hour
    per_run_time_limit=300,        # 5 minutes per model
    memory_limit=3072              # 3GB
)

automl.fit(X_train, y_train)
y_pred = automl.predict(X_test)

# Get best model
print(automl.show_models())
```

### TPOT (Tree-based Pipeline Optimization Tool)

```python
from tpot import TPOTClassifier

# TPOT Classifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    random_state=42,
    n_jobs=-1
)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

# Export best pipeline
tpot.export('tpot_pipeline.py')
```

---

## Model Interpretation

### SHAP (SHapley Additive exPlanations)

```python
import shap

# For tree-based models
explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Waterfall plot
shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# For any model
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
```

### LIME (Local Interpretable Model-agnostic Explanations)

```python
from lime.lime_tabular import LimeTabularExplainer

# Create explainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# Explain single prediction
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

explanation.show_in_notebook(show_table=True)
```

### Feature Importance

```python
# Tree-based models
importance = xgb_clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Permutation importance
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42
)

perm_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)
```

---

## Key Takeaways

1. **Ensemble methods** often outperform single models
2. **XGBoost/LightGBM/CatBoost** are industry standards
3. **Handle imbalanced data** properly - don't just use accuracy
4. **Feature engineering** is crucial - often more important than algorithm
5. **Model interpretation** is essential for production systems

---

## Next Steps

- **[05_Deep_Learning_Fundamentals.md](05_Deep_Learning_Fundamentals.md)** - Neural networks
- **[09_MLOps_and_Deployment.md](09_MLOps_and_Deployment.md)** - Production deployment

---

**Master these techniques to become an advanced ML engineer!**

