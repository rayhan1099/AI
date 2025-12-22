# MLOps and Model Deployment

## 📖 Table of Contents
1. [Introduction to MLOps](#introduction-to-mlops)
2. [Model Serialization](#model-serialization)
3. [API Development](#api-development)
4. [Containerization with Docker](#containerization-with-docker)
5. [Cloud Deployment](#cloud-deployment)
6. [Model Monitoring](#model-monitoring)
7. [CI/CD for ML](#cicd-for-ml)
8. [Complete Examples](#complete-examples)

---

## Introduction to MLOps

### What is MLOps?
- **ML + DevOps**: Applying DevOps practices to ML systems
- **End-to-end lifecycle**: From development to production
- **Automation**: Automate ML workflows
- **Monitoring**: Track model performance in production

### MLOps Components
1. **Version Control**: Code, data, models
2. **Experiment Tracking**: Track experiments and results
3. **Model Registry**: Store and version models
4. **Model Serving**: Deploy models as APIs
5. **Monitoring**: Monitor model performance
6. **CI/CD**: Automated testing and deployment

---

## Model Serialization

### Scikit-learn Models

```python
import joblib
import pickle

# Save model
joblib.dump(model, 'model.pkl')

# Load model
model = joblib.load('model.pkl')

# Alternative with pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### TensorFlow/Keras Models

```python
# Save entire model
model.save('model.h5')
model.save('model_directory/')  # SavedModel format

# Load model
model = keras.models.load_model('model.h5')

# Save only weights
model.save_weights('weights.h5')

# Load weights
model.load_weights('weights.h5')

# Save as TensorFlow Lite (for mobile)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### PyTorch Models

```python
# Save entire model
torch.save(model, 'model.pth')

# Load model
model = torch.load('model.pth')

# Save state dict (recommended)
torch.save(model.state_dict(), 'model_state.pth')

# Load state dict
model = MyModel()
model.load_state_dict(torch.load('model_state.pth'))
model.eval()

# Save for inference (TorchScript)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pth')
```

### ONNX (Cross-platform)

```python
# Convert to ONNX
import onnx
import onnxruntime

# From TensorFlow
import tf2onnx
onnx_model, _ = tf2onnx.convert.from_keras(model, output_path='model.onnx')

# From PyTorch
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx')

# Load and run ONNX model
session = onnxruntime.InferenceSession('model.onnx')
outputs = session.run(None, {'input': input_data})
```

---

## API Development

### FastAPI - Modern Python API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Create app
app = FastAPI(title="ML Model API")

# Define request/response models
class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

# Health check
@app.get("/")
def read_root():
    return {"message": "ML Model API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Preprocess
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0].max()
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Batch prediction
@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    predictions = []
    for request in requests:
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        predictions.append(float(prediction))
    return {"predictions": predictions}

# Run server
# uvicorn main:app --host 0.0.0.0 --port 8000
```

### Flask - Simple API

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### TensorFlow Serving

```python
# Save model in SavedModel format
model.save('saved_model/my_model')

# Start TensorFlow Serving (Docker)
# docker run -p 8501:8501 \
#   --mount type=bind,source=/path/to/saved_model,target=/models/my_model \
#   -e MODEL_NAME=my_model -t tensorflow/serving

# Client code
import requests
import json

data = {
    "instances": [[1.0, 2.0, 3.0, 4.0]]
}

response = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    data=json.dumps(data)
)
predictions = response.json()['predictions']
```

---

## Containerization with Docker

### Dockerfile for ML API

```dockerfile
# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy model files
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.pkl
    restart: unless-stopped
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=ml_db
      - POSTGRES_USER=ml_user
      - POSTGRES_PASSWORD=ml_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Building and Running

```bash
# Build image
docker build -t ml-api .

# Run container
docker run -p 8000:8000 ml-api

# Run with docker-compose
docker-compose up -d
```

---

## Cloud Deployment

### AWS SageMaker

```python
import sagemaker
from sagemaker.sklearn import SKLearn

# Create estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=sagemaker.get_execution_role(),
    instance_type='ml.m5.large',
    framework_version='0.24-1',
    py_version='py3'
)

# Train
sklearn_estimator.fit({'training': 's3://bucket/training-data'})

# Deploy
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Predict
result = predictor.predict(data)
```

### Google Cloud AI Platform

```python
from google.cloud import aiplatform

# Deploy model
model = aiplatform.Model('projects/PROJECT_ID/models/MODEL_ID')
endpoint = model.deploy(
    machine_type='n1-standard-2',
    min_replica_count=1,
    max_replica_count=3
)

# Predict
predictions = endpoint.predict(instances=data)
```

### Azure ML

```python
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice

# Deploy model
ws = Workspace.from_config()
model = Model(ws, name='my_model')

service = Model.deploy(
    ws,
    'my-service',
    [model],
    inference_config,
    deployment_config,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
```

### Heroku Deployment

```python
# Procfile
web: uvicorn main:app --host 0.0.0.0 --port $PORT

# requirements.txt
fastapi==0.68.0
uvicorn==0.15.0
scikit-learn==0.24.2
joblib==1.0.1

# Deploy
# git push heroku main
```

---

## Model Monitoring

### Logging Predictions

```python
import logging
from datetime import datetime

logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def predict_with_logging(request):
    prediction = model.predict(request.features)
    
    # Log prediction
    logging.info(f"Prediction: {prediction}, Features: {request.features}")
    
    return prediction
```

### Performance Monitoring

```python
import time
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        prediction = model.predict(request.features)
        prediction_counter.inc()
        return {"prediction": prediction}
    finally:
        prediction_latency.observe(time.time() - start_time)

# Start metrics server
start_http_server(8000)
```

### Data Drift Detection

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Compare production data with training data
report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=train_data,
    current_data=production_data,
    column_mapping=column_mapping
)

# Check for drift
if report.as_dict()['metrics'][0]['result']['dataset_drift']:
    print("Data drift detected!")
```

---

## CI/CD for ML

### GitHub Actions Workflow

```yaml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/
    - name: Run linting
      run: |
        flake8 .
  
  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Train model
      run: |
        python train.py
    - name: Upload model
      uses: actions/upload-artifact@v2
      with:
        name: model
        path: model.pkl
  
  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        # Deployment script
        ./deploy.sh
```

### MLflow for Experiment Tracking

```python
import mlflow
import mlflow.sklearn

# Start experiment
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Register model
    mlflow.register_model("runs:/<run_id>/model", "ProductionModel")
```

---

## Complete Examples

### Example 1: Complete FastAPI Deployment

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

app = FastAPI()
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        logger.info(f"Prediction: {prediction}")
        
        return {"prediction": float(prediction)}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
```

### Example 2: Dockerized ML Service

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Key Takeaways

1. **Serialization**: Save models properly for deployment
2. **APIs**: Use FastAPI or Flask for model serving
3. **Docker**: Containerize for consistent deployment
4. **Monitoring**: Track model performance in production
5. **CI/CD**: Automate testing and deployment

---

## Next Steps

- **[10_Practical_Projects_Guide.md](10_Practical_Projects_Guide.md)** - Build end-to-end projects
- **[11_Libraries_Cheat_Sheet.md](11_Libraries_Cheat_Sheet.md)** - Quick reference

---

**Master MLOps to deploy models in production successfully!**

