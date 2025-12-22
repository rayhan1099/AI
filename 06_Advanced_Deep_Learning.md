# Advanced Deep Learning - CNNs, RNNs, Transfer Learning

## 📖 Table of Contents
1. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
2. [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks-rnns)
3. [Transfer Learning](#transfer-learning)
4. [Advanced Architectures](#advanced-architectures)
5. [Generative Models](#generative-models)
6. [Complete Examples](#complete-examples)

---

## Convolutional Neural Networks (CNNs)

### Why CNNs for Images?
- **Spatial hierarchy**: Learns features at different scales
- **Parameter sharing**: Fewer parameters than fully connected
- **Translation invariance**: Recognizes patterns regardless of position

### CNN Architecture Components

#### Convolutional Layer
```python
from tensorflow.keras import layers

# 2D Convolution
conv_layer = layers.Conv2D(
    filters=32,              # Number of filters
    kernel_size=(3, 3),       # Filter size
    strides=(1, 1),           # Step size
    padding='same',           # 'same' or 'valid'
    activation='relu',
    input_shape=(28, 28, 1)   # (height, width, channels)
)
```

#### Pooling Layer
```python
# Max Pooling
max_pool = layers.MaxPooling2D(
    pool_size=(2, 2),         # Pool size
    strides=2                  # Step size
)

# Average Pooling
avg_pool = layers.AveragePooling2D(pool_size=(2, 2))
```

#### Complete CNN Model
```python
model = keras.Sequential([
    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Convolutional Block 3
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten
    layers.Flatten(),
    
    # Dense layers
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### Advanced CNN Techniques

#### Batch Normalization
```python
layers.Conv2D(32, (3, 3)),
layers.BatchNormalization(),
layers.Activation('relu')
```

#### Depthwise Separable Convolution
```python
layers.SeparableConv2D(32, (3, 3), activation='relu')
```

#### Global Average Pooling
```python
# Instead of Flatten + Dense
layers.GlobalAveragePooling2D()
layers.Dense(10, activation='softmax')
```

---

## Recurrent Neural Networks (RNNs)

### Why RNNs for Sequences?
- **Memory**: Remembers previous inputs
- **Variable length**: Handles sequences of different lengths
- **Temporal patterns**: Captures time-dependent relationships

### LSTM (Long Short-Term Memory)

```python
# LSTM Layer
lstm_layer = layers.LSTM(
    units=128,                # Number of units
    return_sequences=True,    # Return full sequence
    dropout=0.2,             # Dropout on inputs
    recurrent_dropout=0.2     # Dropout on recurrent connections
)

# LSTM Model for Sequence Classification
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(10, activation='softmax')
])
```

### GRU (Gated Recurrent Unit)

```python
# GRU (simpler than LSTM, often similar performance)
gru_layer = layers.GRU(
    units=128,
    return_sequences=True,
    dropout=0.2
)

model = keras.Sequential([
    layers.GRU(64, return_sequences=True, input_shape=(timesteps, features)),
    layers.GRU(64),
    layers.Dense(10, activation='softmax')
])
```

### Bidirectional RNNs

```python
# Bidirectional LSTM
bidirectional_lstm = layers.Bidirectional(
    layers.LSTM(64, return_sequences=True)
)

model = keras.Sequential([
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(10, activation='softmax')
])
```

### Time Series Forecasting Example

```python
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Prepare data
seq_length = 10
X, y = create_sequences(ts_data, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build model
model = keras.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=32)
```

---

## Transfer Learning

### Why Transfer Learning?
- **Pre-trained models**: Use models trained on large datasets
- **Faster training**: Less data and time needed
- **Better performance**: Leverage learned features

### Image Classification with Pre-trained Models

```python
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0

# Load pre-trained model (without top layers)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom classifier
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

### Fine-tuning

```python
# Unfreeze some layers for fine-tuning
base_model.trainable = True

# Freeze early layers, unfreeze later layers
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Use lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with lower learning rate
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

### Available Pre-trained Models

```python
# ImageNet pre-trained models
from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50, ResNet101, ResNet152,
    InceptionV3,
    Xception,
    MobileNet, MobileNetV2,
    EfficientNetB0, EfficientNetB1, EfficientNetB2,
    DenseNet121, DenseNet169
)

# Example: EfficientNet (state-of-the-art)
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

---

## Advanced Architectures

### Residual Networks (ResNet)

```python
# Residual Block
def residual_block(x, filters):
    shortcut = x
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# Use pre-trained ResNet
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False)
```

### Inception Module

```python
# Multi-scale feature extraction
def inception_module(x, filters):
    # 1x1 convolution
    branch1 = layers.Conv2D(filters[0], (1, 1), activation='relu')(x)
    
    # 1x1 -> 3x3
    branch2 = layers.Conv2D(filters[1], (1, 1), activation='relu')(x)
    branch2 = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(branch2)
    
    # 1x1 -> 5x5
    branch3 = layers.Conv2D(filters[3], (1, 1), activation='relu')(x)
    branch3 = layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')(branch3)
    
    # 3x3 max pooling -> 1x1
    branch4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch4 = layers.Conv2D(filters[5], (1, 1), activation='relu')(branch4)
    
    # Concatenate
    return layers.Concatenate()([branch1, branch2, branch3, branch4])
```

### Attention Mechanism

```python
# Self-attention layer
class SelfAttention(layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
    
    def build(self, input_shape):
        self.query = layers.Dense(self.embed_dim)
        self.key = layers.Dense(self.embed_dim)
        self.value = layers.Dense(self.embed_dim)
    
    def call(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(float(self.embed_dim))
        attention_weights = tf.nn.softmax(attention_scores)
        
        output = tf.matmul(attention_weights, v)
        return output
```

---

## Generative Models

### Variational Autoencoder (VAE)

```python
# Encoder
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)

z_mean = layers.Dense(2, name='z_mean')(x)
z_log_var = layers.Dense(2, name='z_log_var')(x)

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_inputs = layers.Input(shape=(2,))
x = layers.Dense(7 * 7 * 64, activation='relu')(decoder_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

# VAE Model
vae = keras.Model(encoder_inputs, decoder_outputs)
```

### Generative Adversarial Network (GAN)

```python
# Generator
def build_generator(latent_dim):
    model = keras.Sequential([
        layers.Dense(128 * 7 * 7, input_dim=latent_dim),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), activation='tanh', padding='same')
    ])
    return model

# Discriminator
def build_discriminator():
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), strides=2, padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Conv2D(64, (3, 3), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# GAN
generator = build_generator(100)
discriminator = build_discriminator()

discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## Complete Examples

### Example 1: Image Classification with CNN

```python
# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocess
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Build CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Train
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test)
)
```

### Example 2: Text Classification with LSTM

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train_text)
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# Pad sequences
max_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length)

# Build LSTM model
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=max_length),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train_padded, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test_padded, y_test)
)
```

### Example 3: Transfer Learning for Custom Dataset

```python
# Load pre-trained model
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base
base_model.trainable = False

# Add custom head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Fine-tune
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)
```

---

## Key Takeaways

1. **CNNs** for images - Use pre-trained models when possible
2. **RNNs/LSTMs** for sequences - Bidirectional often helps
3. **Transfer Learning** - Start with pre-trained models
4. **Fine-tuning** - Unfreeze layers gradually with lower LR
5. **Data Augmentation** - Essential for small datasets

---

## Next Steps

- **[07_Transformers_and_NLP.md](07_Transformers_and_NLP.md)** - Modern NLP with Transformers
- **[08_Computer_Vision.md](08_Computer_Vision.md)** - Advanced Computer Vision

---

**Master these architectures to build state-of-the-art models!**

