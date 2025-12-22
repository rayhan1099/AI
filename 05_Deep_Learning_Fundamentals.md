# Deep Learning Fundamentals - Neural Networks

## 📖 Table of Contents
1. [Introduction to Neural Networks](#introduction-to-neural-networks)
2. [TensorFlow/Keras Basics](#tensorflowkeras-basics)
3. [PyTorch Basics](#pytorch-basics)
4. [Neural Network Architecture](#neural-network-architecture)
5. [Training Neural Networks](#training-neural-networks)
6. [Regularization Techniques](#regularization-techniques)
7. [Optimization Algorithms](#optimization-algorithms)
8. [Complete Examples](#complete-examples)

---

## Introduction to Neural Networks

### What is Deep Learning?
- **Subset of ML** using neural networks with multiple layers
- **Automatic feature learning** - learns representations from data
- **State-of-the-art** performance in many domains

### Why Neural Networks?
- **Non-linear relationships**: Can model complex patterns
- **Scalability**: Works with large datasets
- **Flexibility**: Adaptable to different problem types

### Key Concepts
- **Neurons**: Basic computation units
- **Layers**: Groups of neurons
- **Weights**: Learnable parameters
- **Activation Functions**: Introduce non-linearity
- **Loss Function**: Measures prediction error
- **Optimizer**: Updates weights to minimize loss

---

## TensorFlow/Keras Basics

### Installation
```bash
pip install tensorflow
# For GPU support: pip install tensorflow-gpu
```

### Sequential API (Simple Models)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Predict
predictions = model.predict(X_test)
```

### Functional API (Complex Models)

```python
# Input layer
inputs = keras.Input(shape=(784,))

# Hidden layers
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Output layer
outputs = layers.Dense(10, activation='softmax')(x)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile and train (same as Sequential)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Model Subclassing (Maximum Flexibility)

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)

model = MyModel()
```

---

## PyTorch Basics

### Installation
```bash
pip install torch torchvision torchaudio
```

### Basic PyTorch Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = NeuralNet(input_size=784, hidden_size=128, num_classes=10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Neural Network Architecture

### Layer Types

#### Dense (Fully Connected) Layer
```python
# Keras
layers.Dense(units=128, activation='relu')

# PyTorch
nn.Linear(in_features=784, out_features=128)
```

#### Convolutional Layer (for images)
```python
# Keras
layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')

# PyTorch
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
```

#### Recurrent Layer (for sequences)
```python
# Keras
layers.LSTM(units=128, return_sequences=True)

# PyTorch
nn.LSTM(input_size=10, hidden_size=128)
```

### Activation Functions

```python
# ReLU (Rectified Linear Unit) - Most common
layers.Dense(128, activation='relu')
# or
nn.ReLU()

# Sigmoid - For binary classification output
layers.Dense(1, activation='sigmoid')

# Softmax - For multi-class classification
layers.Dense(10, activation='softmax')

# Tanh - Alternative to sigmoid
layers.Dense(128, activation='tanh')

# Leaky ReLU - Prevents dying ReLU problem
layers.LeakyReLU(alpha=0.01)

# ELU - Exponential Linear Unit
layers.ELU(alpha=1.0)
```

### Building a Complete Model

```python
# Keras Example
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(784,)),
    
    # Hidden layers
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(10, activation='softmax')
])
```

---

## Training Neural Networks

### Data Preparation

```python
# Normalize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# For images: normalize to 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape if needed (for images)
X_train = X_train.reshape(-1, 28, 28, 1)  # (samples, height, width, channels)
```

### Compiling the Model

```python
# Optimizers
optimizer_adam = keras.optimizers.Adam(learning_rate=0.001)
optimizer_sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
optimizer_rmsprop = keras.optimizers.RMSprop(learning_rate=0.001)

# Loss functions
loss_binary = 'binary_crossentropy'  # Binary classification
loss_multiclass = 'sparse_categorical_crossentropy'  # Multi-class
loss_regression = 'mse'  # Regression

# Metrics
metrics = ['accuracy']  # Classification
metrics = ['mae', 'mse']  # Regression

# Compile
model.compile(
    optimizer=optimizer_adam,
    loss=loss_multiclass,
    metrics=['accuracy']
)
```

### Training with Callbacks

```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)

# Callbacks
callbacks = [
    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    
    # Reduce learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    ),
    
    # TensorBoard logging
    TensorBoard(log_dir='./logs')
]

# Train with callbacks
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)
```

### Visualizing Training

```python
import matplotlib.pyplot as plt

# Plot training history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)
```

---

## Regularization Techniques

### Dropout

```python
# Randomly set some neurons to zero during training
layers.Dropout(0.3)  # 30% dropout rate
```

### Batch Normalization

```python
# Normalize activations
layers.BatchNormalization()
```

### L1/L2 Regularization

```python
# L2 regularization (weight decay)
layers.Dense(
    128,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(0.01)
)

# L1 regularization
layers.Dense(
    128,
    activation='relu',
    kernel_regularizer=keras.regularizers.l1(0.01)
)

# L1 + L2
layers.Dense(
    128,
    activation='relu',
    kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
)
```

### Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Use in training
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10
)
```

---

## Optimization Algorithms

### Adam (Adaptive Moment Estimation) - Most Popular

```python
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)
```

### SGD (Stochastic Gradient Descent)

```python
optimizer = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)
```

### RMSprop

```python
optimizer = keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9
)
```

### Learning Rate Scheduling

```python
# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

# Cosine decay
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=1000
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

---

## Complete Examples

### Example 1: Image Classification (MNIST)

```python
# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

# Build model
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=20,
    validation_data=(X_test, y_test)
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
```

### Example 2: Binary Classification

```python
# Build model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test)
)
```

### Example 3: Regression

```python
# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)  # No activation for regression
])

# Compile
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train
model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test)
)
```

---

## Key Takeaways

1. **Start with Keras** - Easier to learn, great for most tasks
2. **Use callbacks** - Early stopping, model checkpointing essential
3. **Regularize properly** - Dropout, batch norm prevent overfitting
4. **Monitor training** - Plot loss/accuracy curves
5. **Experiment** - Try different architectures, hyperparameters

---

## Next Steps

- **[06_Advanced_Deep_Learning.md](06_Advanced_Deep_Learning.md)** - CNNs, RNNs, Transfer Learning
- **[07_Transformers_and_NLP.md](07_Transformers_and_NLP.md)** - NLP and Transformers

---

**Practice building neural networks with different architectures!**

