# Practical Projects Guide - Build Real AI Applications

## 📖 Table of Contents
1. [Project Ideas by Level](#project-ideas-by-level)
2. [Project Structure](#project-structure)
3. [End-to-End Project Examples](#end-to-end-project-examples)
4. [Best Practices](#best-practices)
5. [Portfolio Building](#portfolio-building)

---

## Project Ideas by Level

### Beginner Projects

1. **House Price Prediction**
   - Dataset: Boston Housing, California Housing
   - Techniques: Linear Regression, Random Forest
   - Skills: Data preprocessing, model evaluation

2. **Email Spam Classifier**
   - Dataset: Spam email dataset
   - Techniques: Naive Bayes, Logistic Regression
   - Skills: Text preprocessing, classification

3. **Iris Flower Classification**
   - Dataset: Iris dataset (built-in)
   - Techniques: KNN, SVM, Decision Trees
   - Skills: Multi-class classification

4. **Customer Churn Prediction**
   - Dataset: Telco customer churn
   - Techniques: Logistic Regression, Random Forest
   - Skills: Imbalanced data handling

### Intermediate Projects

1. **Sentiment Analysis**
   - Dataset: IMDB reviews, Twitter data
   - Techniques: LSTM, BERT
   - Skills: NLP, deep learning

2. **Image Classification**
   - Dataset: CIFAR-10, custom dataset
   - Techniques: CNN, Transfer Learning
   - Skills: Computer vision, CNNs

3. **Stock Price Prediction**
   - Dataset: Yahoo Finance
   - Techniques: LSTM, ARIMA
   - Skills: Time series, RNNs

4. **Recommendation System**
   - Dataset: MovieLens, Amazon products
   - Techniques: Collaborative Filtering, Matrix Factorization
   - Skills: Recommendation algorithms

### Advanced Projects

1. **Object Detection System**
   - Dataset: COCO, custom images
   - Techniques: YOLO, Faster R-CNN
   - Skills: Object detection, computer vision

2. **Chatbot**
   - Dataset: Conversational data
   - Techniques: Transformers, GPT
   - Skills: NLP, sequence-to-sequence

3. **Image Generation (GAN)**
   - Dataset: CelebA, custom images
   - Techniques: GANs, DCGAN
   - Skills: Generative models

4. **Medical Image Analysis**
   - Dataset: Medical imaging data
   - Techniques: U-Net, Transfer Learning
   - Skills: Medical AI, segmentation

---

## Project Structure

### Recommended Structure

```
project_name/
├── data/
│   ├── raw/              # Original data
│   ├── processed/        # Cleaned data
│   └── external/         # External data sources
├── models/
│   ├── trained/          # Saved models
│   └── checkpoints/      # Training checkpoints
├── src/
│   ├── data/            # Data processing scripts
│   ├── features/        # Feature engineering
│   ├── models/          # Model definitions
│   ├── training/        # Training scripts
│   └── evaluation/      # Evaluation scripts
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── api/
│   ├── main.py          # FastAPI/Flask app
│   └── endpoints.py
├── requirements.txt
├── README.md
├── .gitignore
└── setup.py
```

---

## End-to-End Project Examples

### Project 1: House Price Prediction

```python
# 1. Data Loading and Exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('data/housing.csv')

# Exploratory Data Analysis
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Visualizations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# 2. Data Preprocessing
# Handle missing values
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Feature engineering
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Encode categorical variables
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# 3. Prepare features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 6. Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# 7. Save model
import joblib
joblib.dump(model, 'models/house_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
```

### Project 2: Image Classification with CNN

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 1. Load and preprocess data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 2. Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# 3. Build CNN model
model = keras.Sequential([
    # Convolutional blocks
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4. Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    ),
    keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# 6. Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 7. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 8. Save model
model.save('models/cifar10_cnn.h5')
```

### Project 3: Sentiment Analysis API

```python
# train_model.py
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

dataset = Dataset.from_dict({
    'text': texts,
    'label': labels
})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()
model.save_pretrained('models/sentiment_model')

# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()
tokenizer = BertTokenizer.from_pretrained('models/sentiment_model')
model = BertForSequenceClassification.from_pretrained('models/sentiment_model')
model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    inputs = tokenizer(
        request.text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    sentiment = "positive" if predicted_class == 1 else "negative"
    
    return {
        "sentiment": sentiment,
        "confidence": confidence
    }
```

---

## Best Practices

### 1. Version Control
```python
# Use Git for code
git init
git add .
git commit -m "Initial commit"

# Use DVC for data and models
dvc init
dvc add data/raw/dataset.csv
dvc add models/trained_model.pkl
```

### 2. Configuration Management
```python
# config.yaml
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"

model:
  name: "RandomForest"
  n_estimators: 100
  max_depth: 20

training:
  test_size: 0.2
  random_state: 42

# Load config
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

### 3. Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Training started")
```

### 4. Testing
```python
# tests/test_model.py
import pytest
from src.models.train import train_model

def test_model_training():
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'predict')

def test_model_prediction():
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
```

### 5. Documentation
```python
def preprocess_data(df):
    """
    Preprocess the input dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
        
    Example:
        >>> df_processed = preprocess_data(df)
    """
    # Implementation
    pass
```

---

## Portfolio Building

### GitHub Portfolio Structure

1. **README.md** - Project description, setup instructions
2. **Requirements.txt** - Dependencies
3. **Notebooks/** - Jupyter notebooks with analysis
4. **Scripts/** - Production-ready code
5. **Results/** - Model performance, visualizations
6. **Documentation/** - Detailed documentation

### What to Include

1. **Problem Statement**: What problem are you solving?
2. **Data**: Description of dataset
3. **Approach**: Methods and techniques used
4. **Results**: Metrics, visualizations
5. **Deployment**: Link to deployed model/API
6. **Future Work**: Improvements and extensions

### Example README

```markdown
# House Price Prediction

## Problem
Predict house prices based on various features.

## Dataset
California Housing Dataset with 20,640 samples.

## Approach
- Exploratory Data Analysis
- Feature Engineering
- Random Forest Regressor
- Model Evaluation

## Results
- RMSE: $68,000
- R² Score: 0.82

## Usage
```python
python train.py
python predict.py --input data/new_house.csv
```

## Deployment
API available at: https://house-price-api.herokuapp.com
```

---

## Key Takeaways

1. **Start Simple**: Begin with basic projects
2. **Iterate**: Improve projects over time
3. **Document**: Write clear READMEs
4. **Deploy**: Show working applications
5. **Share**: Put projects on GitHub

---

## Next Steps

- Build 3-5 projects covering different domains
- Deploy at least one project
- Write blog posts about your projects
- Contribute to open-source ML projects

---

**Build projects to showcase your skills and learn by doing!**

