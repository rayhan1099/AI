# Large Dataset Training and Testing - Complete Guide

## 📖 Table of Contents
1. [Handling Large Datasets](#handling-large-datasets)
2. [Memory-Efficient Training](#memory-efficient-training)
3. [Distributed Training](#distributed-training)
4. [Data Streaming](#data-streaming)
5. [Testing Large Datasets](#testing-large-datasets)
6. [Performance Optimization](#performance-optimization)
7. [Complete Examples](#complete-examples)

---

## Handling Large Datasets

### Problem with Large Datasets
- **Memory limitations**: Can't load entire dataset into RAM
- **Slow I/O**: Reading from disk is slow
- **Training time**: Takes too long to train

### Solutions

#### 1. Data Sampling

```python
import pandas as pd
import numpy as np

# Load large CSV in chunks
chunk_size = 10000
chunks = []

for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
    # Process each chunk
    processed_chunk = process_data(chunk)
    chunks.append(processed_chunk)

# Combine if needed
df = pd.concat(chunks, ignore_index=True)

# Or sample for initial exploration
df_sample = pd.read_csv('large_dataset.csv', nrows=100000)  # First 100k rows
df_sample = pd.read_csv('large_dataset.csv').sample(n=100000, random_state=42)  # Random sample
```

#### 2. Data Types Optimization

```python
# Reduce memory usage by optimizing data types
def optimize_dtypes(df):
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # Use float32 instead of float64
        else:
            # Convert object to category if low cardinality
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
    
    return df

# Usage
df = optimize_dtypes(df)
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

#### 3. Using Dask for Large DataFrames

```python
import dask.dataframe as dd

# Read large CSV with Dask
df = dd.read_csv('large_dataset.csv')

# Operations are lazy (computed on demand)
df_filtered = df[df['column'] > 100]
df_grouped = df_filtered.groupby('category').mean()

# Compute when needed
result = df_grouped.compute()

# Convert to pandas for ML
df_pandas = df.head(1000000)  # Get first million rows as pandas
```

#### 4. Using Apache Spark (PySpark)

```python
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("LargeDatasetTraining") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Read large dataset
df = spark.read.csv('large_dataset.csv', header=True, inferSchema=True)

# Process with Spark
df_filtered = df.filter(df['column'] > 100)
df_grouped = df_filtered.groupBy('category').agg({'value': 'mean'})

# Convert to Pandas for ML (if dataset fits in memory)
df_pandas = df_grouped.toPandas()

# Or use Spark MLlib for distributed ML
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

# Feature assembly
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
df_features = assembler.transform(df)

# Train model
rf = RandomForestClassifier(featuresCol='features', labelCol='label')
model = rf.fit(df_features)
```

---

## Memory-Efficient Training

### 1. Batch Processing

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Process data in batches
def train_in_batches(model, X, y, batch_size=10000):
    n_samples = len(X)
    n_batches = n_samples // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        # Partial fit (for incremental learning)
        model.partial_fit(X_batch, y_batch)
    
    return model

# Use SGDClassifier for incremental learning
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log_loss', learning_rate='adaptive')
model = train_in_batches(model, X_train, y_train, batch_size=10000)
```

### 2. Generators for Neural Networks

```python
from tensorflow.keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):
    def __init__(self, file_path, batch_size=32, shuffle=True):
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = self._get_indices()
    
    def _get_indices(self):
        # Get total number of samples (e.g., from file)
        # This is a placeholder - implement based on your data format
        return np.arange(1000000)  # Example: 1M samples
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Load batch from disk
        X_batch, y_batch = self._load_batch(batch_indices)
        
        return X_batch, y_batch
    
    def _load_batch(self, indices):
        # Implement your data loading logic
        # Read from HDF5, parquet, or other format
        X = []
        y = []
        for idx in indices:
            # Load single sample
            sample_x, sample_y = load_sample(idx)
            X.append(sample_x)
            y.append(sample_y)
        
        return np.array(X), np.array(y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Use generator
train_generator = DataGenerator('train_data.h5', batch_size=32)
val_generator = DataGenerator('val_data.h5', batch_size=32)

model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    workers=4,
    use_multiprocessing=True
)
```

### 3. HDF5 for Large Arrays

```python
import h5py
import numpy as np

# Create HDF5 file
with h5py.File('large_dataset.h5', 'w') as f:
    # Create datasets
    f.create_dataset('X', shape=(1000000, 784), dtype='float32')
    f.create_dataset('y', shape=(1000000,), dtype='int32')
    
    # Write data in chunks
    chunk_size = 10000
    for i in range(0, 1000000, chunk_size):
        f['X'][i:i+chunk_size] = X_data[i:i+chunk_size]
        f['y'][i:i+chunk_size] = y_data[i:i+chunk_size]

# Read from HDF5
def load_hdf5_batch(file_path, start_idx, end_idx):
    with h5py.File(file_path, 'r') as f:
        X_batch = f['X'][start_idx:end_idx]
        y_batch = f['y'][start_idx:end_idx]
    return X_batch, y_batch
```

### 4. Parquet Format (Efficient for Large DataFrames)

```python
import pandas as pd
import pyarrow.parquet as pq

# Save to Parquet (compressed, columnar format)
df.to_parquet('large_dataset.parquet', compression='snappy')

# Read in chunks
parquet_file = pq.ParquetFile('large_dataset.parquet')
for batch in parquet_file.iter_batches(batch_size=100000):
    df_batch = batch.to_pandas()
    # Process batch
    process_batch(df_batch)

# Or read specific columns (columnar format advantage)
df = pd.read_parquet('large_dataset.parquet', columns=['col1', 'col2', 'col3'])
```

---

## Distributed Training

### 1. TensorFlow Distributed Training

```python
import tensorflow as tf

# Multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training automatically uses all GPUs
model.fit(train_dataset, epochs=10)

# Multi-worker strategy (for multiple machines)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
```

### 2. PyTorch Distributed Training

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def train_distributed(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # or 'gloo' for CPU
        init_method='tcp://localhost:23456',
        rank=rank,
        world_size=world_size
    )
    
    # Create model and move to device
    model = create_model().to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Training loop
    for epoch in range(10):
        train_epoch(model, train_loader, rank)

# Launch distributed training
world_size = 4  # Number of GPUs
mp.spawn(train_distributed, args=(world_size,), nprocs=world_size)
```

### 3. Horovod (Multi-framework Distributed Training)

```python
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to local rank
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Create model
model = create_model()

# Wrap optimizer
optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.Adam())

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# Training
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
]

model.fit(train_dataset, epochs=10, callbacks=callbacks)
```

---

## Data Streaming

### 1. Streaming with Generators

```python
def data_stream(file_path, batch_size=1000):
    """Stream data from file"""
    with open(file_path, 'r') as f:
        batch = []
        for line in f:
            # Parse line
            sample = parse_line(line)
            batch.append(sample)
            
            if len(batch) >= batch_size:
                yield process_batch(batch)
                batch = []
        
        # Yield remaining
        if batch:
            yield process_batch(batch)

# Use stream
for batch in data_stream('large_file.txt'):
    model.partial_fit(batch['X'], batch['y'])
```

### 2. Kafka for Real-time Data Streaming

```python
from kafka import KafkaConsumer
import json

# Create consumer
consumer = KafkaConsumer(
    'ml_training_topic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Stream and train
batch = []
for message in consumer:
    sample = message.value
    batch.append(sample)
    
    if len(batch) >= 1000:
        # Train on batch
        X_batch = [s['features'] for s in batch]
        y_batch = [s['label'] for s in batch]
        model.partial_fit(X_batch, y_batch)
        batch = []
```

---

## Testing Large Datasets

### 1. Stratified Sampling for Testing

```python
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# For large datasets, use stratified sampling
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

### 2. Incremental Testing

```python
def test_in_batches(model, X_test, y_test, batch_size=10000):
    """Test model on large dataset in batches"""
    n_samples = len(X_test)
    n_batches = n_samples // batch_size
    
    all_predictions = []
    all_probas = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        
        # Predict
        predictions = model.predict(X_batch)
        probas = model.predict_proba(X_batch)
        
        all_predictions.extend(predictions)
        all_probas.extend(probas)
    
    return np.array(all_predictions), np.array(all_probas)

# Calculate metrics
predictions, probas = test_in_batches(model, X_test, y_test)
accuracy = accuracy_score(y_test, predictions)
```

### 3. Parallel Testing

```python
from joblib import Parallel, delayed

def test_chunk(model, X_chunk, y_chunk):
    predictions = model.predict(X_chunk)
    return accuracy_score(y_chunk, predictions)

# Split test set into chunks
n_chunks = 10
chunk_size = len(X_test) // n_chunks

chunks = [
    (X_test[i*chunk_size:(i+1)*chunk_size], 
     y_test[i*chunk_size:(i+1)*chunk_size])
    for i in range(n_chunks)
]

# Test in parallel
accuracies = Parallel(n_jobs=-1)(
    delayed(test_chunk)(model, X_chunk, y_chunk)
    for X_chunk, y_chunk in chunks
)

overall_accuracy = np.mean(accuracies)
```

---

## Performance Optimization

### 1. Use Appropriate Data Types

```python
# Use float32 instead of float64 (half the memory, often same accuracy)
X = X.astype('float32')

# Use int8 for small integers
y = y.astype('int8')
```

### 2. Cache Intermediate Results

```python
from joblib import Memory

# Create cache directory
memory = Memory('./cache', verbose=0)

@memory.cache
def expensive_computation(data):
    # Expensive operation
    return processed_data

# First call: computes and caches
result1 = expensive_computation(data)

# Second call: loads from cache
result2 = expensive_computation(data)
```

### 3. Use GPU for Large Computations

```python
import cupy as cp  # GPU-accelerated NumPy

# Convert to GPU array
X_gpu = cp.asarray(X)

# Operations run on GPU
result_gpu = cp.dot(X_gpu, weights_gpu)

# Convert back to CPU if needed
result_cpu = cp.asnumpy(result_gpu)
```

---

## Complete Examples

### Example 1: Training on 10M+ Rows

```python
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load data in chunks
chunk_size = 100000
scaler = StandardScaler()
model = SGDClassifier(loss='log_loss', learning_rate='adaptive')

# Fit scaler on first chunk
first_chunk = True
for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
    X_chunk = chunk.drop('target', axis=1)
    y_chunk = chunk['target']
    
    if first_chunk:
        scaler.fit(X_chunk)
        first_chunk = False
    
    # Transform and train
    X_scaled = scaler.transform(X_chunk)
    model.partial_fit(X_scaled, y_chunk, classes=np.unique(y_chunk))

# Save model
joblib.dump(model, 'trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

### Example 2: Neural Network with Data Generator

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import h5py

# Create data generator
class HDF5Generator(Sequence):
    def __init__(self, file_path, batch_size=32):
        self.file_path = file_path
        self.batch_size = batch_size
        with h5py.File(file_path, 'r') as f:
            self.n_samples = f['X'].shape[0]
    
    def __len__(self):
        return self.n_samples // self.batch_size
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, self.n_samples)
            
            X_batch = f['X'][start_idx:end_idx]
            y_batch = f['y'][start_idx:end_idx]
        
        return X_batch, y_batch

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train with generator
train_gen = HDF5Generator('train_data.h5', batch_size=32)
val_gen = HDF5Generator('val_data.h5', batch_size=32)

model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    workers=4,
    use_multiprocessing=True
)
```

---

## Key Takeaways

1. **Chunk processing**: Process data in batches
2. **Memory optimization**: Use appropriate data types
3. **Efficient formats**: Use Parquet, HDF5
4. **Incremental learning**: Use partial_fit
5. **Distributed training**: Use multiple GPUs/machines
6. **Generators**: For streaming data

---

## Next Steps

- **[15_Train_Test_Split_Complete_Guide.md](15_Train_Test_Split_Complete_Guide.md)** - Complete train/test strategies
- **[16_Library_Comparison_Benchmarks.md](16_Library_Comparison_Benchmarks.md)** - Library comparisons

---

**Master large dataset handling to work with real-world data!**

