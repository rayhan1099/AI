# Practical AI Projects - Flask, Django & FastAPI

## 📖 Table of Contents
1. [Project 1: Sentiment Analysis API](#project-1-sentiment-analysis-api)
2. [Project 2: Image Classification API](#project-2-image-classification-api)
3. [Project 3: Text Summarization API](#project-3-text-summarization-api)
4. [Project 4: Price Prediction API](#project-4-price-prediction-api)
5. [Project 5: Chatbot API](#project-5-chatbot-api)
6. [Framework Comparison](#framework-comparison)
7. [Deployment Guide](#deployment-guide)

---

## Project 1: Sentiment Analysis API

### FastAPI Version (Recommended)

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from transformers import pipeline
import uvicorn

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Load model (using transformers for simplicity)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API", "endpoints": ["/predict", "/health"]}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    try:
        # Analyze sentiment
        result = sentiment_analyzer(request.text)[0]
        
        # Format response
        sentiment = result['label'].lower()
        confidence = result['score']
        
        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    results = []
    for text in texts:
        result = sentiment_analyzer(text)[0]
        results.append({
            "text": text,
            "sentiment": result['label'].lower(),
            "confidence": result['score']
        })
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Flask Version

```python
# app.py
from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Load model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "endpoints": ["/predict", "/health"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text")
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # Analyze sentiment
        result = sentiment_analyzer(text)[0]
        
        return jsonify({
            "text": text,
            "sentiment": result['label'].lower(),
            "confidence": result['score']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json()
        texts = data.get("texts", [])
        
        results = []
        for text in texts:
            result = sentiment_analyzer(text)[0]
            results.append({
                "text": text,
                "sentiment": result['label'].lower(),
                "confidence": result['score']
            })
        
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

### Django Version

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@csrf_exempt
@require_http_methods(["GET"])
def home(request):
    return JsonResponse({
        "message": "Sentiment Analysis API",
        "endpoints": ["/predict", "/health"]
    })

@csrf_exempt
@require_http_methods(["GET"])
def health(request):
    return JsonResponse({"status": "healthy"})

@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    try:
        data = json.loads(request.body)
        text = data.get("text")
        
        if not text:
            return JsonResponse({"error": "Text is required"}, status=400)
        
        result = sentiment_analyzer(text)[0]
        
        return JsonResponse({
            "text": text,
            "sentiment": result['label'].lower(),
            "confidence": result['score']
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("health/", views.health, name="health"),
    path("predict/", views.predict, name="predict"),
]
```

### Requirements

```txt
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
flask==3.0.0
django==4.2.7
transformers==4.35.0
torch==2.1.0
pydantic==2.5.0
```

### Testing

```python
# test_api.py
import requests

# Test FastAPI
response = requests.post("http://localhost:8000/predict", json={"text": "I love this product!"})
print(response.json())

# Test Flask
response = requests.post("http://localhost:5000/predict", json={"text": "This is terrible!"})
print(response.json())
```

---

## Project 2: Image Classification API

### FastAPI Version

```python
# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms, models
import uvicorn

app = FastAPI(title="Image Classification API")

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ImageNet class labels
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Format results
        results = []
        for i in range(5):
            results.append({
                "class": classes[top5_idx[i]],
                "confidence": float(top5_prob[i])
            })
        
        return JSONResponse({
            "filename": file.filename,
            "predictions": results
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Flask Version

```python
# app.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import torch
from torchvision import transforms, models
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model
model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            
            # Preprocess
            image_tensor = transform(image).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top5_prob, top5_idx = torch.topk(probabilities, 5)
            
            # Format results
            results = []
            for i in range(5):
                results.append({
                    "class": classes[top5_idx[i]],
                    "confidence": float(top5_prob[i])
                })
            
            return jsonify({
                "filename": file.filename,
                "predictions": results
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

---

## Project 3: Text Summarization API

### FastAPI Version

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI(title="Text Summarization API")

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class TextRequest(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30

class SummaryResponse(BaseModel):
    original_text: str
    summary: str
    original_length: int
    summary_length: int

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: TextRequest):
    try:
        if len(request.text) < request.min_length:
            raise HTTPException(status_code=400, detail="Text is too short")
        
        # Summarize
        result = summarizer(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=False
        )
        
        summary = result[0]['summary_text']
        
        return SummaryResponse(
            original_text=request.text,
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Project 4: Price Prediction API (House Prices)

### FastAPI Version

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn

app = FastAPI(title="House Price Prediction API")

# Load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int

class PricePrediction(BaseModel):
    features: HouseFeatures
    predicted_price: float
    confidence_interval: dict

@app.post("/predict", response_model=PricePrediction)
async def predict_price(features: HouseFeatures):
    try:
        # Convert to array
        feature_array = np.array([[
            features.bedrooms,
            features.bathrooms,
            features.sqft_living,
            features.sqft_lot,
            features.floors,
            features.waterfront,
            features.view,
            features.condition,
            features.grade,
            features.sqft_above,
            features.sqft_basement,
            features.yr_built,
            features.yr_renovated
        ]])
        
        # Scale features
        features_scaled = scaler.transform(feature_array)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Calculate confidence interval (simplified)
        std_error = prediction * 0.1  # 10% error estimate
        lower_bound = prediction - 1.96 * std_error
        upper_bound = prediction + 1.96 * std_error
        
        return PricePrediction(
            features=features,
            predicted_price=float(prediction),
            confidence_interval={
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Training Script

```python
# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("house_prices.csv")

# Select features
features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    'floors', 'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'
]

X = df[features]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save model
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved!")
```

---

## Project 5: Chatbot API

### FastAPI Version

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import uvicorn

app = FastAPI(title="Chatbot API")

# Load chatbot model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

class ChatMessage(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    response: str
    conversation_history: list

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # Encode conversation
        conversation = message.conversation_history + [message.message]
        full_text = " ".join(conversation)
        
        # Tokenize
        inputs = tokenizer.encode(full_text + tokenizer.eos_token, return_tensors='pt')
        
        # Generate response
        outputs = model.generate(
            inputs,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(full_text):].strip()
        
        # Update conversation history
        new_history = conversation + [response]
        
        return ChatResponse(
            response=response,
            conversation_history=new_history
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Framework Comparison

### FastAPI (Recommended for AI)

**Pros:**
- ⚡ **Fast**: Built on Starlette, very fast
- 📝 **Auto Documentation**: Swagger UI included
- 🔒 **Type Safety**: Pydantic models
- 🚀 **Async Support**: Native async/await
- 📊 **Modern**: Built for modern Python

**Best For:**
- AI/ML APIs
- High-performance APIs
- When you need async
- Modern Python projects

**Example Use Case:**
```python
# FastAPI is best for AI because:
# 1. Fast performance
# 2. Easy to integrate ML models
# 3. Auto-generated API docs
# 4. Type validation with Pydantic
```

### Flask

**Pros:**
- 🎯 **Simple**: Easy to learn
- 🔧 **Flexible**: Minimal, customizable
- 📚 **Large Community**: Lots of resources
- 🛠️ **Mature**: Battle-tested

**Best For:**
- Simple APIs
- Quick prototypes
- When you need flexibility
- Learning web development

**Example Use Case:**
```python
# Flask is good for:
# 1. Simple ML APIs
# 2. Quick prototypes
# 3. When you don't need async
# 4. Learning projects
```

### Django

**Pros:**
- 🏗️ **Full Framework**: Complete solution
- 🔐 **Security**: Built-in security features
- 📦 **Batteries Included**: Admin, ORM, etc.
- 🏢 **Enterprise**: Used by large companies

**Best For:**
- Full web applications
- When you need admin panel
- Complex projects
- Enterprise applications

**Example Use Case:**
```python
# Django is best for:
# 1. Full web apps with ML features
# 2. When you need admin interface
# 3. Complex applications
# 4. Enterprise projects
```

### Comparison Table

| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Learning Curve | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Async Support | ✅ | ❌ | ✅ |
| Auto Docs | ✅ | ❌ | ❌ |
| Type Safety | ✅ | ❌ | ❌ |
| Flexibility | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Best for AI | ✅ | ⚠️ | ⚠️ |

---

## Deployment Guide

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models
    restart: unless-stopped
```

### Heroku Deployment

```bash
# Procfile
web: uvicorn main:app --host 0.0.0.0 --port $PORT

# Deploy
git init
heroku create your-app-name
git push heroku main
```

### AWS Deployment

```python
# Use AWS Lambda with Serverless Framework
# serverless.yml
service: ai-api

provider:
  name: aws
  runtime: python3.9

functions:
  api:
    handler: main.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
```

---

## Complete Project Structure

```
ai-project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── models.py        # ML models
│   ├── schemas.py       # Pydantic models
│   └── utils.py         # Utilities
├── models/
│   ├── trained_model.pkl
│   └── scaler.pkl
├── tests/
│   ├── test_api.py
│   └── test_models.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Best Practices

### 1. Error Handling

```python
from fastapi import HTTPException

@app.post("/predict")
async def predict(data: Request):
    try:
        # Your code
        pass
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 2. Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(data: Request):
    logger.info(f"Received prediction request: {data}")
    # Your code
```

### 3. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: Request):
    # Your code
```

### 4. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_computation(input_data):
    # Expensive operation
    return result
```

---

## Testing Your API

### Unit Tests

```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict():
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    assert "sentiment" in response.json()
```

### Load Testing

```python
# load_test.py
import requests
import time
from concurrent.futures import ThreadPoolExecutor

def make_request():
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": "Test message"}
    )
    return response.status_code

# Test with 100 concurrent requests
with ThreadPoolExecutor(max_workers=100) as executor:
    start = time.time()
    results = list(executor.map(make_request, range(100)))
    end = time.time()
    
print(f"Time: {end - start:.2f}s")
print(f"Success rate: {results.count(200) / len(results) * 100:.2f}%")
```

---

## Key Takeaways

1. **FastAPI is best for AI APIs** - Fast, modern, async support
2. **Flask is good for simple projects** - Easy to learn, flexible
3. **Django is for full applications** - When you need complete framework
4. **Always handle errors** - Proper error handling is crucial
5. **Test your APIs** - Write unit and integration tests
6. **Deploy with Docker** - Makes deployment easier

---

## Next Steps

1. **Choose a project** from above
2. **Set up environment** - Install dependencies
3. **Run locally** - Test on your machine
4. **Deploy** - Use Docker or cloud platform
5. **Monitor** - Track API performance

---

**Start building AI APIs and gain practical experience!**

