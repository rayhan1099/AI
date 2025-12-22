# Python ML/DL Libraries Cheat Sheet

## Quick Reference for All Essential Libraries

### NumPy
```python
import numpy as np

# Arrays
arr = np.array([1, 2, 3])
arr = np.zeros((3, 4))
arr = np.ones((2, 3))
arr = np.arange(0, 10, 2)
arr = np.linspace(0, 1, 5)
arr = np.random.rand(3, 3)

# Operations
arr.shape, arr.size, arr.dtype
arr.reshape(2, 3)
arr.flatten()
np.concatenate([a, b])
np.vstack([a, b])
np.hstack([a, b])

# Math
np.mean(arr), np.std(arr), np.sum(arr)
np.min(arr), np.max(arr)
np.dot(a, b)  # Matrix multiplication
```

### Pandas
```python
import pandas as pd

# Create
df = pd.DataFrame(data)
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')

# Info
df.info(), df.describe(), df.head()
df.shape, df.columns, df.dtypes

# Select
df['col'], df[['col1', 'col2']]
df.iloc[0], df.loc[0]
df[df['col'] > 5]

# Modify
df['new_col'] = values
df.drop('col', axis=1)
df.rename(columns={'old': 'new'})

# Groupby
df.groupby('col').mean()
df.groupby('col').agg({'col1': 'mean', 'col2': 'sum'})

# Missing
df.isnull().sum()
df.dropna()
df.fillna(value)
```

### Scikit-learn
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
```

### TensorFlow/Keras
```python
from tensorflow import keras
from tensorflow.keras import layers

# Sequential
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)

# Predict
predictions = model.predict(X_test)

# Save/Load
model.save('model.h5')
model = keras.models.load_model('model.h5')
```

### PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Training
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Save/Load
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

### XGBoost
```python
import xgboost as xgb

# Classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Feature importance
importance = model.feature_importances_
```

### LightGBM
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train)
```

### Transformers (Hugging Face)
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello world", return_tensors='pt')
outputs = model(**inputs)
```

### OpenCV
```python
import cv2

# Read/Write
img = cv2.imread('image.jpg')
cv2.imwrite('output.jpg', img)

# Operations
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(img, (224, 224))
blurred = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(gray, 100, 200)

# Threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

### Matplotlib
```python
import matplotlib.pyplot as plt

# Plot
plt.plot(x, y)
plt.scatter(x, y)
plt.hist(data)
plt.bar(categories, values)

# Labels
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Title')
plt.legend()

# Show
plt.show()
plt.savefig('plot.png')
```

### Seaborn
```python
import seaborn as sns

# Plots
sns.scatterplot(x='col1', y='col2', data=df)
sns.heatmap(df.corr(), annot=True)
sns.boxplot(x='category', y='value', data=df)
sns.distplot(df['col'])
```

### FastAPI
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    data: list[float]

@app.get("/")
def read_root():
    return {"message": "Hello"}

@app.post("/predict")
def predict(request: Request):
    return {"prediction": model.predict(request.data)}
```

### Joblib
```python
import joblib

# Save
joblib.dump(model, 'model.pkl')

# Load
model = joblib.load('model.pkl')
```

### MLflow
```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

---

## Installation Commands

```bash
# Core
pip install numpy pandas matplotlib seaborn

# ML
pip install scikit-learn xgboost lightgbm catboost

# Deep Learning
pip install tensorflow torch

# NLP
pip install nltk spacy transformers

# Computer Vision
pip install opencv-python pillow

# Backend
pip install fastapi uvicorn flask

# Utilities
pip install jupyter notebook joblib mlflow
```

---

**Keep this cheat sheet handy for quick reference!**

