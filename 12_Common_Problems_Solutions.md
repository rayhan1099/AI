# Common Problems and Solutions in ML/DL

## 📖 Table of Contents
1. [Data Problems](#data-problems)
2. [Model Training Issues](#model-training-issues)
3. [Performance Problems](#performance-problems)
4. [Deployment Issues](#deployment-issues)
5. [Debugging Tips](#debugging-tips)

---

## Data Problems

### Problem: Missing Values

**Solution:**
```python
# Check missing values
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Fill with mean/median
df['column'].fillna(df['column'].mean(), inplace=True)
df['column'].fillna(df['column'].median(), inplace=True)

# Fill with mode (for categorical)
df['column'].fillna(df['column'].mode()[0], inplace=True)

# Forward fill
df['column'].fillna(method='ffill', inplace=True)

# Use KNN Imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

### Problem: Outliers

**Solution:**
```python
# Detect outliers using IQR
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]

# Or cap outliers
df['column'] = df['column'].clip(lower=lower_bound, upper=upper_bound)

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['column']))
df = df[z_scores < 3]
```

### Problem: Imbalanced Classes

**Solution:**
```python
# Use class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
model = RandomForestClassifier(class_weight='balanced')

# Use SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Use stratified sampling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

### Problem: Categorical Variables

**Solution:**
```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Target encoding
mean_target = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(mean_target)
```

---

## Model Training Issues

### Problem: Overfitting

**Solution:**
```python
# Add regularization
model = LogisticRegression(C=0.1, penalty='l2')  # Lower C = more regularization

# Use dropout (for neural networks)
layers.Dropout(0.5)

# Early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Reduce model complexity
model = RandomForestClassifier(max_depth=5, min_samples_split=10)

# Increase training data
# Use data augmentation
```

### Problem: Underfitting

**Solution:**
```python
# Increase model complexity
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2
)

# Reduce regularization
model = LogisticRegression(C=10.0)

# Add more features
# Feature engineering

# Train longer
model.fit(X_train, y_train, epochs=100)  # For neural networks
```

### Problem: Slow Training

**Solution:**
```python
# Use GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Reduce batch size
model.fit(X_train, y_train, batch_size=32)

# Use fewer features
# Feature selection

# Use simpler model
# Or use pre-trained models

# Parallel processing
model = RandomForestClassifier(n_jobs=-1)  # Use all cores
```

### Problem: Gradient Explosion/Vanishing

**Solution:**
```python
# Gradient clipping
optimizer = keras.optimizers.Adam(clipnorm=1.0)

# Batch normalization
layers.BatchNormalization()

# Better initialization
layers.Dense(128, kernel_initializer='he_normal')

# Use different activation
layers.LeakyReLU(alpha=0.01)  # Instead of ReLU

# Residual connections
# Use ResNet architecture
```

---

## Performance Problems

### Problem: Low Accuracy

**Solution:**
```python
# Check data quality
print(df.isnull().sum())
print(df.describe())

# Feature engineering
df['feature1_x_feature2'] = df['feature1'] * df['feature2']

# Try different algorithms
models = [
    RandomForestClassifier(),
    XGBClassifier(),
    LogisticRegression(),
    SVC()
]

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Ensemble methods
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=[...])
```

### Problem: High Variance in Cross-Validation

**Solution:**
```python
# Increase number of folds
scores = cross_val_score(model, X, y, cv=10)

# Use stratified K-fold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
scores = cross_val_score(model, X, y, cv=skf)

# Increase training data
# Reduce model complexity
# Regularization
```

### Problem: Poor Generalization

**Solution:**
```python
# Check for data leakage
# Ensure train/test split is correct

# Use proper validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2
)

# Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)

# Regularization
# Early stopping
# Dropout
```

---

## Deployment Issues

### Problem: Model File Too Large

**Solution:**
```python
# Quantization (TensorFlow)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Pruning
# Remove less important features/neurons

# Use smaller model
# Transfer learning with smaller base model
```

### Problem: Slow Inference

**Solution:**
```python
# Batch predictions
predictions = model.predict(X_batch)

# Use ONNX Runtime (faster)
import onnxruntime
session = onnxruntime.InferenceSession('model.onnx')

# Optimize model
# Use TensorRT (NVIDIA GPUs)
# Use quantization

# Cache predictions
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(features):
    return model.predict(features)
```

### Problem: Memory Issues

**Solution:**
```python
# Process in batches
def predict_in_batches(model, X, batch_size=1000):
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        pred = model.predict(batch)
        predictions.extend(pred)
    return predictions

# Use generators
def data_generator(X, y, batch_size=32):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

# Clear memory
import gc
gc.collect()
```

---

## Debugging Tips

### Check Data

```python
# Print data info
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())

# Check distributions
import matplotlib.pyplot as plt
df['column'].hist()
plt.show()

# Check correlations
print(df.corr())
```

### Check Model

```python
# Model summary
print(model.summary())  # For Keras

# Feature importance
print(model.feature_importances_)  # For tree-based

# Check predictions
print(y_pred[:10])
print(y_test[:10])

# Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
```

### Check Training

```python
# Plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# Check gradients (PyTorch)
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Learning rate scheduling
# Reduce learning rate if loss plateaus
```

### Common Errors

```python
# ValueError: Input contains NaN
# Solution: Handle missing values

# ValueError: Found array with 0 sample(s)
# Solution: Check data shape

# MemoryError
# Solution: Reduce batch size, use generators

# CUDA out of memory
# Solution: Reduce batch size, use CPU
```

---

## Key Takeaways

1. **Always check data first** - Most problems are data-related
2. **Start simple** - Use simple models first
3. **Validate properly** - Use cross-validation
4. **Monitor training** - Plot loss curves
5. **Test incrementally** - Test each component separately

---

**Debug systematically and check each component!**

